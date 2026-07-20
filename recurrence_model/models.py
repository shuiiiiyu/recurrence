from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
except ImportError:
    OfficialMamba = None

try:
    from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig as OfficialGatedDeltaProductConfig
    from fla.models.gated_deltaproduct.modeling_gated_deltaproduct import GatedDeltaProductBlock as OfficialGatedDeltaProductBlock
except ImportError:
    sys.path.insert(0, "/data/shencanyu/repos/flash-linear-attention")
    try:
        from fla.models.gated_deltaproduct.configuration_gated_deltaproduct import GatedDeltaProductConfig as OfficialGatedDeltaProductConfig
        from fla.models.gated_deltaproduct.modeling_gated_deltaproduct import GatedDeltaProductBlock as OfficialGatedDeltaProductBlock
    except ImportError:
        OfficialGatedDeltaProductConfig = None
        OfficialGatedDeltaProductBlock = None


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

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]], append_to_cache: bool = True, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=att.device, dtype=torch.bool)
            cache_len = k_all.size(2)
            if key_padding_mask.size(1) < cache_len:
                pad_value = key_padding_mask[:, -1:].expand(-1, cache_len - key_padding_mask.size(1))
                key_padding_mask = torch.cat([key_padding_mask, pad_value], dim=1)
            elif key_padding_mask.size(1) > cache_len:
                key_padding_mask = key_padding_mask[:, :cache_len]
            att = att.masked_fill(~key_padding_mask.view(bsz, 1, 1, cache_len), torch.finfo(att.dtype).min)
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

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]], append_to_cache: bool = True, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y, cache = self.attn(self.ln1(x), cache, append_to_cache=append_to_cache, key_padding_mask=key_padding_mask)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x, cache


class FullSequenceCausalSelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        allowed_mask = causal_mask.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            key_mask = attention_mask.to(device=x.device, dtype=torch.bool).view(bsz, 1, 1, seq_len)
            allowed_mask = allowed_mask & key_mask
        att = att.masked_fill(~allowed_mask, torch.finfo(att.dtype).min)
        att = self.dropout(F.softmax(att, dim=-1))
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out(y)


class FullSequenceTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FullSequenceCausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


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


class BaseFullSequenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([FullSequenceTransformerBlock(cfg.d_model, cfg.num_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _embed_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.token_emb(input_ids) + self.pos_emb(pos_ids)


class CausalTransformerBaseline(BaseFullSequenceModel):
    """Standard decoder-only causal Transformer baseline."""

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._embed_sequence(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.token_head(self.final_ln(x))


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

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        caches = self._empty_caches()
        prev_deep_state = input_ids.new_zeros((bsz, 1, self.cfg.d_model), dtype=torch.float32)
        hidden_states = []
        for pos in range(seq_len):
            x = self._embed_token(input_ids[:, pos], pos)
            for layer_idx, block in enumerate(self.blocks):
                if layer_idx == self.target_idx:
                    x = x + torch.sigmoid(self.feedback_gate_logit) * self.feedback_proj(prev_deep_state)
                key_padding_mask = attention_mask[:, : pos + 1] if attention_mask is not None else None
                x, caches[layer_idx] = block(x, caches[layer_idx], append_to_cache=True, key_padding_mask=key_padding_mask)
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

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        caches = self._empty_caches()
        prev_deep_state = input_ids.new_zeros((bsz, 1, self.cfg.d_model), dtype=torch.float32)
        hidden_states = []

        for pos in range(seq_len):
            x = self._embed_token(input_ids[:, pos], pos)
            for layer_idx in range(self.entry_end):
                key_padding_mask = attention_mask[:, : pos + 1] if attention_mask is not None else None
                x, caches[layer_idx] = self.blocks[layer_idx](x, caches[layer_idx], append_to_cache=True, key_padding_mask=key_padding_mask)

            gate = torch.sigmoid(self.feedback_gate_logit)
            loop_state = x + gate * self.feedback_proj(prev_deep_state)

            for loop_idx in range(self.num_loops):
                x = loop_state
                for layer_idx in range(self.loop_start, self.loop_end + 1):
                    append = self.cfg.append_internal_steps_to_cache or loop_idx == self.num_loops - 1
                    key_padding_mask = attention_mask[:, : pos + 1] if attention_mask is not None else None
                    x, caches[layer_idx] = self.blocks[layer_idx](x, caches[layer_idx], append_to_cache=append, key_padding_mask=key_padding_mask)
                if loop_idx < self.num_loops - 1:
                    loop_state = loop_state + gate * self.feedback_proj(x)

            prev_deep_state = x
            hidden_states.append(x)

        hidden = torch.cat(hidden_states, dim=1)
        return self.token_head(self.final_ln(hidden))


class DepthLoopRatioGt1Model(BaseFullSequenceModel):
    """Depth recurrence, ratio > 1.

    The whole token sequence is processed in parallel with causal attention.
    A middle layer range is looped multiple times in depth, without cross-token
    deep-to-shallow feedback.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.entry_end = cfg.ratiogt1_entry_layers
        self.loop_start = cfg.ratiogt1_loop_start_layer - 1
        self.loop_end = cfg.ratiogt1_loop_end_layer - 1
        self.num_loops = cfg.ratiogt1_num_loops
        if self.entry_end != self.loop_start:
            raise ValueError("entry layers should end exactly before loop_start")
        if not (0 <= self.loop_start <= self.loop_end < cfg.num_layers):
            raise ValueError("Invalid loop layer range")
        if self.num_loops < 2:
            raise ValueError("ratio > 1 depth loop requires at least two loops")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._embed_sequence(input_ids)
        for layer_idx in range(self.entry_end):
            x = self.blocks[layer_idx](x, attention_mask=attention_mask)

        for _ in range(self.num_loops):
            for layer_idx in range(self.loop_start, self.loop_end + 1):
                x = self.blocks[layer_idx](x, attention_mask=attention_mask)

        for layer_idx in range(self.loop_end + 1, self.cfg.num_layers):
            x = self.blocks[layer_idx](x, attention_mask=attention_mask)

        return self.token_head(self.final_ln(x))






class PrefixDeepFeedbackModel(BaseFullSequenceModel):
    """Prefix recurrent depth feedback.

    At recurrence step t the visible prefix 1..t is recomputed with causal
    attention. The only state carried across recurrence steps is each token's
    previous source-layer hidden state, which is added to that same token's
    target layer in the next update.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        num_updates_per_token: int,
        source_layer: int,
        target_layer: int,
        gate_init: float,
    ):
        super().__init__(cfg)
        self.num_updates_per_token = num_updates_per_token
        self.source_idx = source_layer - 1
        self.target_idx = target_layer - 1
        if self.num_updates_per_token < 1:
            raise ValueError("num_updates_per_token must be >= 1")
        if not (0 <= self.target_idx < self.source_idx < cfg.num_layers):
            raise ValueError("Require target layer < source layer within model depth")
        self.feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.feedback_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def _embed_prefix(self, input_ids: torch.Tensor, end: int) -> torch.Tensor:
        prefix_ids = input_ids[:, :end]
        bsz, prefix_len = prefix_ids.shape
        pos_ids = torch.arange(prefix_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        return self.token_emb(prefix_ids) + self.pos_emb(pos_ids)

    def _update_prefix(
        self,
        prefix_x: torch.Tensor,
        source_prev: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = prefix_x
        source_cur = None
        gate = torch.sigmoid(self.feedback_gate_logit)
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx == self.target_idx:
                x = x + gate * self.feedback_proj(source_prev)
            x = block(x, attention_mask=attention_mask)
            if layer_idx == self.source_idx:
                source_cur = x
        return x, x if source_cur is None else source_cur

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        source_prev = None
        outputs = []

        for pos in range(seq_len):
            prefix_len = pos + 1
            prefix_mask = attention_mask[:, :prefix_len] if attention_mask is not None else None
            prefix_x0 = self._embed_prefix(input_ids, prefix_len)
            if source_prev is None:
                source_work = prefix_x0.new_zeros((bsz, 0, self.cfg.d_model))
            else:
                source_work = source_prev

            new_source = prefix_x0.new_zeros((bsz, 1, self.cfg.d_model))
            source_work = torch.cat([source_work, new_source], dim=1)

            final_hidden = None
            for _ in range(self.num_updates_per_token):
                prefix_x = self._embed_prefix(input_ids, prefix_len)
                x, source_work = self._update_prefix(prefix_x, source_work.to(dtype=prefix_x.dtype), prefix_mask)
                final_hidden = x

            source_prev = source_work
            outputs.append(final_hidden[:, -1:, :])

        hidden = torch.cat(outputs, dim=1)
        return self.token_head(self.final_ln(hidden))


class DepthRatio1Model(PrefixDeepFeedbackModel):
    """Depth recurrence, ratio = 1: one prefix recurrent update per new token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(
            cfg,
            num_updates_per_token=1,
            source_layer=cfg.depthratio1_feedback_source_layer,
            target_layer=cfg.depthratio1_feedback_target_layer,
            gate_init=cfg.depthratio1_feedback_gate_init,
        )


class DepthRatioLt1Model(PrefixDeepFeedbackModel):
    """Depth recurrence, ratio < 1: multiple prefix recurrent updates per new token."""

    def __init__(self, cfg: ModelConfig):
        if cfg.depthratiolt1_num_steps < 2:
            raise ValueError("depthRatiolt1 requires depthratiolt1_num_steps >= 2")
        super().__init__(
            cfg,
            num_updates_per_token=cfg.depthratiolt1_num_steps,
            source_layer=cfg.depthratiolt1_feedback_source_layer,
            target_layer=cfg.depthratiolt1_feedback_target_layer,
            gate_init=cfg.depthratiolt1_feedback_gate_init,
        )



class DepthStepMemoryGt1Model(nn.Module):
    """Depth+step recurrence, ratio > 1 with chunk-level memory tokens.

    Each recurrence step processes a chunk of ordinary tokens. A read-memory
    state from the previous chunk is prepended, write-memory placeholders are
    appended, and only ordinary token positions are returned for the task loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = cfg.depthstepgt1_chunk_size
        self.memory_tokens = cfg.depthstepgt1_memory_tokens
        self.source_idx = cfg.depthstepgt1_feedback_source_layer - 1
        self.target_idx = cfg.depthstepgt1_feedback_target_layer - 1
        if self.chunk_size < 1:
            raise ValueError("depthstepgt1_chunk_size must be >= 1")
        if self.memory_tokens < 1:
            raise ValueError("depthstepgt1_memory_tokens must be >= 1")
        if not (0 <= self.target_idx < self.source_idx < cfg.num_layers):
            raise ValueError("Require target layer < source layer within model depth")
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 2 * self.memory_tokens, cfg.d_model)
        self.read_memory_init = nn.Parameter(torch.empty(self.memory_tokens, cfg.d_model).normal_(std=0.02))
        self.write_memory_init = nn.Parameter(torch.empty(self.memory_tokens, cfg.d_model).normal_(std=0.02))
        self.feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.feedback_gate_logit = nn.Parameter(torch.tensor(float(cfg.depthstepgt1_feedback_gate_init)))
        self.blocks = nn.ModuleList([
            FullSequenceTransformerBlock(cfg.d_model, cfg.num_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _chunk_token_embeddings(self, input_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
        bsz = input_ids.size(0)
        pos_ids = torch.arange(start, end, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        return self.token_emb(input_ids[:, start:end]) + self.pos_emb(pos_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        read_memory = self.read_memory_init.unsqueeze(0).expand(bsz, -1, -1).to(device=input_ids.device)
        outputs = []
        gate = torch.sigmoid(self.feedback_gate_logit)

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            token_x = self._chunk_token_embeddings(input_ids, start, end)
            write_memory = self.write_memory_init.unsqueeze(0).expand(bsz, -1, -1).to(device=input_ids.device, dtype=token_x.dtype)
            x = torch.cat([read_memory.to(dtype=token_x.dtype), token_x, write_memory], dim=1)

            chunk_len = end - start
            if attention_mask is None:
                chunk_token_mask = torch.ones(bsz, chunk_len, device=input_ids.device, dtype=torch.long)
            else:
                chunk_token_mask = attention_mask[:, start:end]
            memory_mask = torch.ones(bsz, self.memory_tokens, device=input_ids.device, dtype=chunk_token_mask.dtype)
            chunk_attention_mask = torch.cat([memory_mask, chunk_token_mask, memory_mask], dim=1)

            next_memory = None
            for layer_idx, block in enumerate(self.blocks):
                if layer_idx == self.target_idx:
                    feedback = self.feedback_proj(read_memory.mean(dim=1, keepdim=True).to(dtype=x.dtype))
                    x[:, self.memory_tokens :, :] = x[:, self.memory_tokens :, :] + gate * feedback
                x = block(x, attention_mask=chunk_attention_mask)
                if layer_idx == self.source_idx:
                    next_memory = x[:, -self.memory_tokens :, :]

            if next_memory is None:
                next_memory = x[:, -self.memory_tokens :, :]
            read_memory = next_memory
            token_hidden = x[:, self.memory_tokens : self.memory_tokens + chunk_len, :]
            if attention_mask is not None:
                token_hidden = token_hidden * chunk_token_mask.to(device=token_hidden.device, dtype=token_hidden.dtype).unsqueeze(-1)
            outputs.append(token_hidden)

        hidden = torch.cat(outputs, dim=1)
        return self.token_head(self.final_ln(hidden))



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        bsz, query_len, _ = query.shape
        key_len = key.size(1)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = torch.ones(bsz, 1, query_len, key_len, device=query.device, dtype=torch.bool)
        if key_mask is not None:
            allowed = allowed & key_mask.to(device=query.device, dtype=torch.bool).view(bsz, 1, 1, key_len)
        if causal:
            if query_len != key_len:
                raise ValueError("causal attention expects query_len == key_len")
            causal_mask = torch.ones(query_len, key_len, device=query.device, dtype=torch.bool).tril()
            allowed = allowed & causal_mask.view(1, 1, query_len, key_len)
        att = att.masked_fill(~allowed, torch.finfo(att.dtype).min)
        weights = self.dropout(F.softmax(att, dim=-1))
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(bsz, query_len, self.d_model)
        return self.out(y)


class GatedStateUpdate(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float, gate_type: str, single_gate: bool, skip_ffn: bool):
        super().__init__()
        if gate_type not in {"bias", "lstm"}:
            raise ValueError("brt_gate_type must be 'bias' or 'lstm'")
        if single_gate and skip_ffn:
            raise ValueError("brt_single_gate and brt_skip_ffn cannot both be true")
        self.gate_type = gate_type
        self.single_gate = single_gate
        self.skip_ffn = skip_ffn
        self.post_attn = nn.Linear(2 * d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        if gate_type == "bias":
            self.attn_gate_bias = nn.Parameter(torch.empty(d_model).normal_(std=0.1))
            self.ffn_gate_bias = nn.Parameter(torch.empty(d_model).normal_(std=0.1))
        else:
            self.attn_input_gate = nn.Linear(d_model, d_model)
            self.attn_forget_gate = nn.Linear(d_model, d_model)
            self.ffn_input_gate = nn.Linear(d_model, d_model)
            self.ffn_forget_gate = nn.Linear(d_model, d_model)
            for layer in [self.attn_input_gate, self.attn_forget_gate, self.ffn_input_gate, self.ffn_forget_gate]:
                nn.init.normal_(layer.bias, std=0.1)
                nn.init.xavier_uniform_(layer.weight, gain=0.1)

    def _gate(self, state: torch.Tensor, update: torch.Tensor, hidden: torch.Tensor, stage: str) -> torch.Tensor:
        if self.gate_type == "bias":
            bias = self.attn_gate_bias if stage == "attn" else self.ffn_gate_bias
            gate = torch.sigmoid(bias).view(1, 1, -1)
            return state * gate + update * (1.0 - gate)
        if stage == "attn":
            input_gate = self.attn_input_gate
            forget_gate = self.attn_forget_gate
        else:
            input_gate = self.ffn_input_gate
            forget_gate = self.ffn_forget_gate
        fg = torch.sigmoid(forget_gate(hidden) + 1.0)
        ig = torch.sigmoid(input_gate(hidden) - 1.0)
        return state * fg + update * ig

    def forward(self, state: torch.Tensor, state_self: torch.Tensor, state_cross: torch.Tensor) -> torch.Tensor:
        attn_update = self.post_attn(torch.cat([state_self, state_cross], dim=-1))
        if self.single_gate:
            return self._gate(state, torch.tanh(self.ffn(self.ln(attn_update))), state, "ffn")
        state = self._gate(state, attn_update, state, "attn")
        if self.skip_ffn:
            return state
        ffn_update = self.ffn(self.ln(state))
        return self._gate(state, ffn_update, state, "ffn")


class BlockRecurrentTransformerBlock(nn.Module):
    """Official-style BRT layer with fixed recurrent state vectors.

    The layer follows Meliad's original memory TransformerLayer: tokens in a
    block cross-attend to the previous recurrent state, while the recurrent
    state self-attends and cross-attends to the current token block, then uses a
    bias/LSTM gate to produce the state for the next block.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.brt_block_size < 1:
            raise ValueError("brt_block_size must be >= 1")
        if cfg.brt_num_states < 1:
            raise ValueError("brt_num_states must be >= 1")
        self.cfg = cfg
        self.block_size = cfg.brt_block_size
        self.num_states = cfg.brt_num_states
        self.state_init = nn.Parameter(torch.empty(cfg.brt_num_states, cfg.d_model).normal_(std=0.1))
        self.state_pos = nn.Parameter(torch.empty(cfg.brt_num_states, cfg.d_model).normal_(std=1.0))

        self.token_ln = nn.LayerNorm(cfg.d_model)
        self.token_self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.token_state_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.token_attn_proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.token_ffn_ln = nn.LayerNorm(cfg.d_model)
        self.token_ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        self.state_ln = nn.LayerNorm(cfg.d_model)
        self.state_self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.state_token_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.state_update = GatedStateUpdate(
            cfg.d_model,
            cfg.ffn_mult,
            cfg.dropout,
            cfg.brt_gate_type,
            cfg.brt_single_gate,
            cfg.brt_skip_ffn,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        state = self.state_init.unsqueeze(0).expand(bsz, -1, -1).to(dtype=x.dtype)
        valid_all = (
            torch.ones(bsz, seq_len, device=x.device, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.to(device=x.device, dtype=torch.bool)
        )
        outputs = []
        for start in range(0, seq_len, self.block_size):
            end = min(start + self.block_size, seq_len)
            chunk = x[:, start:end, :]
            chunk_mask = valid_all[:, start:end]

            token_in = self.token_ln(chunk)
            state_for_read = self.state_ln(state + self.state_pos.unsqueeze(0).to(dtype=x.dtype))
            token_local = self.token_self_attn(token_in, token_in, token_in, key_mask=chunk_mask, causal=True)
            token_memory = self.token_state_attn(token_in, state_for_read, state_for_read)
            token_update = self.token_attn_proj(torch.cat([token_local, token_memory], dim=-1))
            chunk_out = chunk + token_update
            chunk_out = chunk_out + self.token_ffn(self.token_ffn_ln(chunk_out))
            if attention_mask is not None:
                chunk_out = chunk_out * chunk_mask.to(dtype=x.dtype).unsqueeze(-1)
            outputs.append(chunk_out)

            state_in = self.state_ln(state + self.state_pos.unsqueeze(0).to(dtype=x.dtype))
            state_self = self.state_self_attn(state_in, state_in, state_in)
            state_cross = self.state_token_attn(state_in, token_in, token_in, key_mask=chunk_mask)
            state = self.state_update(state, state_self, state_cross)

        return torch.cat(outputs, dim=1)


class BlockRecurrentTransformerModel(nn.Module):
    """Step recurrence, ratio > 1 with BRT memory at selected layers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.brt_layer_indices = self._parse_layer_indices(cfg.brt_layer_indices, cfg.num_layers)
        self.blocks = nn.ModuleList([
            BlockRecurrentTransformerBlock(cfg)
            if layer_idx in self.brt_layer_indices
            else FullSequenceTransformerBlock(cfg.d_model, cfg.num_heads, cfg.ffn_mult, cfg.dropout)
            for layer_idx in range(cfg.num_layers)
        ])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def _parse_layer_indices(spec: str, num_layers: int) -> set[int]:
        indices: set[int] = set()
        for raw in str(spec).split(","):
            raw = raw.strip()
            if not raw:
                continue
            idx = int(raw)
            if idx < 0:
                idx = num_layers + idx
            else:
                idx = idx - 1
            if not 0 <= idx < num_layers:
                raise ValueError(f"BRT layer index {raw} is out of range for num_layers={num_layers}")
            indices.add(idx)
        if not indices:
            raise ValueError("brt_layer_indices must select at least one layer")
        return indices

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        if attention_mask is not None:
            x = x * attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.token_head(self.final_ln(x))



class DeltaProductBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        if OfficialGatedDeltaProductConfig is None or OfficialGatedDeltaProductBlock is None:
            raise ImportError(
                "fla is required for --model deltaproduct; expected local checkout at "
                "/data/shencanyu/repos/flash-linear-attention"
            )
        if cfg.d_model % cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.block = OfficialGatedDeltaProductBlock(
            OfficialGatedDeltaProductConfig(
                hidden_size=cfg.d_model,
                num_heads=cfg.num_heads,
                head_dim=cfg.d_model // cfg.num_heads,
                hidden_ratio=cfg.ffn_mult,
                num_hidden_layers=cfg.num_layers,
                vocab_size=cfg.vocab_size,
                use_output_gate=cfg.deltaproduct_use_output_gate,
                use_forget_gate=cfg.deltaproduct_use_forget_gate,
                use_short_conv=cfg.deltaproduct_use_short_conv,
                conv_size=cfg.deltaproduct_conv_size,
                allow_neg_eigval=cfg.deltaproduct_allow_neg_eigval,
                num_householder=cfg.deltaproduct_num_householder,
            ),
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y, _attn, _cache, _attnres = self.block(
                    hidden_states=x,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                )
        else:
            y, _attn, _cache, _attnres = self.block(
                hidden_states=x,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
            )
        y = y.to(dtype=x.dtype)
        if attention_mask is not None:
            y = y * attention_mask.to(device=y.device, dtype=y.dtype).unsqueeze(-1)
        return y


class DeltaProductModel(nn.Module):
    """Step recurrence, ratio < 1: multiple state updates per input token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DeltaProductBlock(cfg, layer_idx) for layer_idx in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.token_head(self.final_ln(x))


class MambaRatio1Block(nn.Module):
    """Step recurrence, ratio = 1 block using the official Mamba layer."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        if OfficialMamba is None:
            raise ImportError("mamba-ssm is required for --model mamba; install the official mamba_ssm package")
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mamba = OfficialMamba(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            use_fast_path=cfg.mamba_use_fast_path,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * mask
        x = x + self.mamba(self.ln1(x))
        if attention_mask is not None:
            x = x * mask
        return x


class MambaRatio1Model(nn.Module):
    """Step recurrence, ratio = 1: one Mamba selective SSM update per token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaRatio1Block(cfg, layer_idx) for layer_idx in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.token_head(self.final_ln(x))


class MambaRatioLt1Block(nn.Module):
    """Step recurrence, ratio < 1: K Mamba state updates per input token.

    The implementation expands each token into K identical internal steps and
    runs the official Mamba scan over the expanded sequence. The block returns
    only the last internal output for each original token.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        if OfficialMamba is None:
            raise ImportError("mamba-ssm is required for --model mamba_lt1; install the official mamba_ssm package")
        if cfg.mamba_lt1_internal_steps < 2:
            raise ValueError("mamba_lt1 requires mamba_lt1_internal_steps >= 2")
        self.internal_steps = cfg.mamba_lt1_internal_steps
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mamba = OfficialMamba(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            use_fast_path=cfg.mamba_use_fast_path,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * mask
        expanded = self.ln1(x).repeat_interleave(self.internal_steps, dim=1)
        if attention_mask is not None:
            expanded_mask = attention_mask.to(device=x.device, dtype=x.dtype).repeat_interleave(self.internal_steps, dim=1).unsqueeze(-1)
            expanded = expanded * expanded_mask
        expanded_y = self.mamba(expanded)
        y = expanded_y.view(bsz, seq_len, self.internal_steps, d_model)[:, :, -1, :]
        x = x + y
        if attention_mask is not None:
            x = x * mask
        return x


class MambaRatioLt1Model(nn.Module):
    """Step recurrence, ratio < 1: multiple Mamba updates per token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaRatioLt1Block(cfg, layer_idx) for layer_idx in range(cfg.num_layers)])
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.token_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.token_head(self.final_ln(x))

def make_model(kind: str, cfg: ModelConfig) -> nn.Module:
    if kind == "baseline":
        return CausalTransformerBaseline(cfg)
    if kind == "ratio1":
        return DepthFeedbackRatio1Model(cfg)
    if kind == "ratiolt1":
        return DepthLoopRatioLt1Model(cfg)
    if kind == "ratiogt1":
        return DepthLoopRatioGt1Model(cfg)
    if kind == "depthstep_gt1":
        return DepthStepMemoryGt1Model(cfg)
    if kind == "depthRatio1":
        return DepthRatio1Model(cfg)
    if kind == "depthRatiolt1":
        return DepthRatioLt1Model(cfg)
    if kind == "deltaproduct":
        return DeltaProductModel(cfg)
    if kind == "brt":
        return BlockRecurrentTransformerModel(cfg)
    if kind == "mamba":
        return MambaRatio1Model(cfg)
    if kind == "mamba_lt1":
        return MambaRatioLt1Model(cfg)
    raise ValueError(f"Unknown model kind: {kind}")
