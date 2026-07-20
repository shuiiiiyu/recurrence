from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
except ImportError:  # pragma: no cover - handled at model construction time
    OfficialMamba = None


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


class DeltaProductStateLayer(nn.Module):
    """Naive PyTorch DeltaProduct recurrent state layer.

    This follows the FLA GatedDeltaProduct layout: q has one vector per token,
    while k/v/beta are expanded into num_householder internal updates per token.
    The fused FLA kernels are intentionally not used so the model remains easy
    to inspect and works in the current project environment.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        num_householder: int,
        use_output_gate: bool,
        use_forget_gate: bool,
        allow_neg_eigval: bool,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_householder < 1:
            raise ValueError("deltaproduct_num_householder must be >= 1")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_householder = num_householder
        self.use_output_gate = use_output_gate
        self.use_forget_gate = use_forget_gate
        self.allow_neg_eigval = allow_neg_eigval

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model * num_householder, bias=False)
        self.v_proj = nn.Linear(d_model, d_model * num_householder, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads * num_householder, bias=False)
        if use_forget_gate:
            self.a_proj = nn.Linear(d_model, num_heads, bias=False)
            a = torch.empty(num_heads, dtype=torch.float32).uniform_(0.0, 16.0)
            self.a_log = nn.Parameter(torch.log(a))
            dt = torch.empty(num_heads, dtype=torch.float32).uniform_(0.001, 0.1)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        if use_output_gate:
            self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = F.normalize(self._split_heads(self.q_proj(x)), p=2, dim=-1)
        k = self.k_proj(x).view(bsz, seq_len, self.num_householder, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_householder, self.num_heads, self.head_dim)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).view(bsz, seq_len, self.num_householder, self.num_heads)
        if self.allow_neg_eigval:
            beta = 2.0 * beta

        if attention_mask is None:
            valid_mask = x.new_ones(bsz, seq_len, 1, 1, 1)
        else:
            valid_mask = attention_mask.to(device=x.device, dtype=x.dtype).view(bsz, seq_len, 1, 1, 1)

        if self.use_forget_gate:
            forget = -self.a_log.float().exp().view(1, 1, self.num_heads) * F.softplus(self.a_proj(x).float() + self.dt_bias.view(1, 1, self.num_heads))
        else:
            forget = None

        state = x.new_zeros(bsz, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32)
        outputs = []
        for pos in range(seq_len):
            old_state = state
            if forget is not None:
                state = state * torch.exp(forget[:, pos]).to(dtype=state.dtype).view(bsz, self.num_heads, 1, 1)
            for householder_idx in range(self.num_householder):
                k_t = k[:, pos, householder_idx].float()
                v_t = v[:, pos, householder_idx].float()
                beta_t = beta[:, pos, householder_idx].float()
                pred = (state * k_t[..., None]).sum(dim=-2)
                state = state + (v_t - pred).unsqueeze(-2) * k_t[..., None] * beta_t[..., None, None]
            valid = valid_mask[:, pos].float()
            state = valid * state + (1.0 - valid) * old_state
            q_t = q[:, pos].float()
            out_t = (state * q_t[..., None]).sum(dim=-2)
            outputs.append(out_t.to(dtype=x.dtype))

        y = torch.stack(outputs, dim=1).reshape(bsz, seq_len, self.d_model)
        if self.use_output_gate:
            y = y * F.silu(self.g_proj(x))
        return self.out(self.dropout(y))


class DeltaProductBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.delta = DeltaProductStateLayer(
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            num_householder=cfg.deltaproduct_num_householder,
            use_output_gate=cfg.deltaproduct_use_output_gate,
            use_forget_gate=cfg.deltaproduct_use_forget_gate,
            allow_neg_eigval=cfg.deltaproduct_allow_neg_eigval,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.delta(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DeltaProductModel(nn.Module):
    """Step recurrence, ratio < 1: multiple state updates per input token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DeltaProductBlock(cfg) for _ in range(cfg.num_layers)])
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


class MambaRatioGt1Block(nn.Module):
    """Step recurrence, ratio > 1: one Mamba state update per token chunk.

    A non-recurrent chunk projection summarizes C tokens into one vector. The
    official Mamba scan then runs over chunk summaries, so the recurrent state
    advances once per C input tokens. The chunk-level output is broadcast back to
    every token in the chunk.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        if OfficialMamba is None:
            raise ImportError("mamba-ssm is required for --model mamba_gt1; install the official mamba_ssm package")
        if cfg.mamba_gt1_chunk_size < 2:
            raise ValueError("mamba_gt1 requires mamba_gt1_chunk_size >= 2")
        self.chunk_size = cfg.mamba_gt1_chunk_size
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.chunk_proj = nn.Linear(cfg.d_model * self.chunk_size, cfg.d_model)
        self.mamba = OfficialMamba(
            d_model=cfg.d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            use_fast_path=cfg.mamba_use_fast_path,
            layer_idx=layer_idx,
        )
        self.broadcast_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        if attention_mask is not None:
            token_mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            x = x * token_mask
        else:
            token_mask = None
        pad_len = (-seq_len) % self.chunk_size
        x_norm = self.ln1(x)
        if pad_len:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_len))
            if token_mask is not None:
                token_mask_padded = F.pad(token_mask, (0, 0, 0, pad_len))
            else:
                token_mask_padded = None
        else:
            token_mask_padded = token_mask
        padded_len = x_norm.size(1)
        num_chunks = padded_len // self.chunk_size
        if token_mask_padded is not None:
            x_norm = x_norm * token_mask_padded
        chunks = x_norm.view(bsz, num_chunks, self.chunk_size, d_model)
        chunk_summary = self.chunk_proj(chunks.reshape(bsz, num_chunks, self.chunk_size * d_model))
        if token_mask_padded is not None:
            chunk_mask = token_mask_padded.view(bsz, num_chunks, self.chunk_size, 1).amax(dim=2)
            chunk_summary = chunk_summary * chunk_mask
        chunk_y = self.mamba(chunk_summary)
        token_y = chunk_y.unsqueeze(2).expand(-1, -1, self.chunk_size, -1).reshape(bsz, padded_len, d_model)
        token_y = token_y[:, :seq_len, :]
        x = x + self.broadcast_proj(token_y)
        if token_mask is not None:
            x = x * token_mask
        return x


class MambaRatioGt1Model(nn.Module):
    """Step recurrence, ratio > 1: one Mamba update per token chunk."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MambaRatioGt1Block(cfg, layer_idx) for layer_idx in range(cfg.num_layers)])
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
    if kind == "deltaproduct":
        return DeltaProductModel(cfg)
    if kind == "mamba":
        return MambaRatio1Model(cfg)
    if kind == "mamba_lt1":
        return MambaRatioLt1Model(cfg)
    if kind == "mamba_gt1":
        return MambaRatioGt1Model(cfg)
    raise ValueError(f"Unknown model kind: {kind}")
