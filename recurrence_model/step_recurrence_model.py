from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .basic_model import FullSequenceCausalSelfAttention, FullSequenceTransformerBlock

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

