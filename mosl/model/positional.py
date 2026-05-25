"""Sinusoidal positional encoding (Vaswani et al. 2017, eqs. 5).

Verbatim from "Attention is All You Need", §3.5:
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Pre-computed up to `max_len` and added to input embeddings (after the
`sqrt(d_model)` scaling that the paper applies to the embeddings themselves).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer (not parameter): saves with state_dict, moves with .to(device).
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to a (B, L, d_model) input.  L must be ≤ max_len."""
        L = x.size(1)
        if L > self.pe.size(1):
            raise ValueError(
                f"sequence length {L} exceeds max_len {self.pe.size(1)}; "
                "increase SinusoidalPositionalEncoding(max_len=...)"
            )
        return x + self.pe[:, :L, :]
