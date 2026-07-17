from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class IncrementalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]], append_to_cache: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = x.size(0)
        qkv = self.qkv(x)
        q, k_new, v_new = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k_new = split_heads(k_new)
        v_new = split_heads(v_new)
        if cache is None:
            k_all = k_new
            v_all = v_new
        else:
            k_all = torch.cat([cache["k"], k_new], dim=2)
            v_all = torch.cat([cache["v"], v_new], dim=2)
        att = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = self.dropout(F.softmax(att, dim=-1))
        y = torch.matmul(att, v_all)
        y = y.transpose(1, 2).contiguous().view(bsz, 1, self.d_model)
        y = self.out(y)
        if append_to_cache:
            return y, {"k": k_all, "v": v_all}
        return y, cache if cache is not None else {"k": k_new, "v": v_new}


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = IncrementalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]], append_to_cache: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y, cache = self.attn(self.ln1(x), cache, append_to_cache=append_to_cache)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x, cache


class BaseDepthRecurrentModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.num_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _embed_token(self, token_ids: torch.Tensor, pos: int) -> torch.Tensor:
        pos_ids = torch.full_like(token_ids, pos)
        return self.token_emb(token_ids).unsqueeze(1) + self.pos_emb(pos_ids).unsqueeze(1)

    def _empty_caches(self) -> List[Optional[Dict[str, torch.Tensor]]]:
        return [None for _ in range(self.cfg.num_layers)]


class CausalTransformerBaseline(BaseDepthRecurrentModel):
    """Standard causal Transformer baseline with no depth recurrence."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        caches = self._empty_caches()
        hidden_states = []
        for pos in range(seq_len):
            x = self._embed_token(input_ids[:, pos], pos)
            for layer_idx, block in enumerate(self.blocks):
                x, caches[layer_idx] = block(x, caches[layer_idx], append_to_cache=True)
            hidden_states.append(x)
        hidden = torch.cat(hidden_states, dim=1)
        return self.token_head(self.final_ln(hidden))


class DepthFeedbackRatio1Model(BaseDepthRecurrentModel):
    """Depth recurrence, ratio = 1: previous deep state feeds current shallow layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.source_idx = cfg.ratio1_feedback_source_layer - 1
        self.target_idx = cfg.ratio1_feedback_target_layer - 1
        if not (0 <= self.target_idx < self.source_idx < cfg.num_layers):
            raise ValueError("Require target layer < source layer within model depth")
        self.feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.feedback_gate_logit = nn.Parameter(torch.tensor(float(cfg.ratio1_feedback_gate_init)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        caches = self._empty_caches()
        prev_deep_state = input_ids.new_zeros((bsz, 1, self.cfg.d_model), dtype=torch.float32)
        hidden_states = []
        for pos in range(seq_len):
            x = self._embed_token(input_ids[:, pos], pos)
            for layer_idx, block in enumerate(self.blocks):
                if layer_idx == self.target_idx:
                    x = x + torch.sigmoid(self.feedback_gate_logit) * self.feedback_proj(prev_deep_state)
                x, caches[layer_idx] = block(x, caches[layer_idx], append_to_cache=True)
                if layer_idx == self.source_idx:
                    prev_deep_state = x
            hidden_states.append(x)
        hidden = torch.cat(hidden_states, dim=1)
        return self.token_head(self.final_ln(hidden))


class DepthLoopRatioLt1Model(BaseDepthRecurrentModel):
    """Depth recurrence, ratio < 1.

    Previous token deep state is injected into the current token's loop entry,
    then the middle/deep stack is run K times before advancing to the next token.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.entry_end = cfg.ratiolt1_entry_layers
        self.loop_start = cfg.ratiolt1_loop_start_layer - 1
        self.loop_end = cfg.ratiolt1_loop_end_layer - 1
        self.num_loops = cfg.ratiolt1_num_loops
        if self.entry_end != self.loop_start:
            raise ValueError("entry layers should end exactly before loop_start")
        if not (0 <= self.loop_start <= self.loop_end < cfg.num_layers):
            raise ValueError("Invalid loop layer range")
        if self.num_loops < 2:
            raise ValueError("ratio < 1 requires at least two recurrence loops per token")
        self.feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.feedback_gate_logit = nn.Parameter(torch.tensor(float(cfg.ratiolt1_feedback_gate_init)))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        caches = self._empty_caches()
        prev_deep_state = input_ids.new_zeros((bsz, 1, self.cfg.d_model), dtype=torch.float32)
        hidden_states = []

        for pos in range(seq_len):
            x = self._embed_token(input_ids[:, pos], pos)
            for layer_idx in range(self.entry_end):
                x, caches[layer_idx] = self.blocks[layer_idx](x, caches[layer_idx], append_to_cache=True)

            gate = torch.sigmoid(self.feedback_gate_logit)
            loop_state = x + gate * self.feedback_proj(prev_deep_state)

            for loop_idx in range(self.num_loops):
                x = loop_state
                for layer_idx in range(self.loop_start, self.loop_end + 1):
                    append = self.cfg.append_internal_steps_to_cache or loop_idx == self.num_loops - 1
                    x, caches[layer_idx] = self.blocks[layer_idx](x, caches[layer_idx], append_to_cache=append)
                if loop_idx < self.num_loops - 1:
                    loop_state = loop_state + gate * self.feedback_proj(x)

            prev_deep_state = x
            hidden_states.append(x)

        hidden = torch.cat(hidden_states, dim=1)
        return self.token_head(self.final_ln(hidden))


def make_model(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "baseline":
        return CausalTransformerBaseline(cfg)
    if kind == "ratio1":
        return DepthFeedbackRatio1Model(cfg)
    if kind == "ratiolt1":
        return DepthLoopRatioLt1Model(cfg)
    raise ValueError(f"Unknown model kind: {kind}")
