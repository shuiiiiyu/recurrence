from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .basic_model import BaseFullSequenceModel

class DepthRatioGt1Model(BaseFullSequenceModel):
    """Depth recurrence, ratio > 1.

    The whole token sequence is processed in parallel with causal attention.
    A configured layer range is looped multiple times in depth. With the default
    full-stack setting, the output hidden states of one depth pass become the
    input hidden states of the next pass over the full sequence.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.entry_end = cfg.ratiogt1_entry_layers
        self.loop_start = cfg.ratiogt1_loop_start_layer - 1
        self.loop_end = cfg.ratiogt1_loop_end_layer - 1
        self.num_loops = cfg.ratiogt1_num_loops
        if not (0 <= self.entry_end <= cfg.num_layers):
            raise ValueError("Invalid number of entry layers")
        if not (0 <= self.loop_start <= self.loop_end < cfg.num_layers):
            raise ValueError("Invalid loop layer range")
        if self.entry_end > self.loop_start:
            raise ValueError("entry layers cannot overlap the loop range")
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

