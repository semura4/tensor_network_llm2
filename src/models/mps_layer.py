"""MPS-based sequence modeling layer.

This layer replaces the self-attention mechanism in a Transformer with
an MPS/Tensor Train-inspired recurrence. It uses multi-head MPS recurrence
for sequence modeling with pre-norm residual blocks.
"""

import torch
import torch.nn as nn

from .mps_core import MultiHeadMPSRecurrence


class MPSSequenceLayer(nn.Module):
    """MPS sequence layer: a drop-in replacement for self-attention.

    Architecture:
        input (B, T, d_model)
        -> LayerNorm -> Multi-Head MPS Recurrence -> residual
        -> LayerNorm -> FFN -> residual
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_ff: int = None,
        **kwargs,  # absorb unused params like max_seq_len
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 2

        # MPS recurrence sub-layer
        self.mps = MultiHeadMPSRecurrence(d_model, d_hidden, num_heads)

        # Feed-forward sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MPS sub-layer
        residual = x
        out = self.mps(self.norm1(x))
        x = residual + self.dropout(out)

        # FFN sub-layer
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x
