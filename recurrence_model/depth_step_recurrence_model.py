from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .basic_model import BaseDepthRecurrentModel, FullSequenceTransformerBlock

class DepthStepRatio1Model(BaseDepthRecurrentModel):
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


class DepthStepRatioLt1Model(BaseDepthRecurrentModel):
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


class DepthStepRatioGt1Model(nn.Module):
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

