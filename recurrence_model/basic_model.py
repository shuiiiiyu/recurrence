from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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

