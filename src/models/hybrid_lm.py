"""Hybrid model: Transformer with attention replaced by MPS recurrence.

The standard Transformer block has:
    LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual

This model keeps the exact same structure but swaps Self-Attention
for Multi-Head MPS Recurrence, preserving the FFN, norms, and residuals.
"""

import torch
import torch.nn as nn

from .embedding import TokenAndPositionEmbedding
from .mps_core import MultiHeadMPSRecurrence


class HybridTransformerBlock(nn.Module):
    """Transformer block with MPS recurrence instead of self-attention."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mps_attn = MultiHeadMPSRecurrence(d_model, d_hidden, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mps_attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridMPSTransformerLM(nn.Module):
    """Transformer LM with self-attention replaced by MPS recurrence.

    Architecture:
        Token + Position Embedding
        -> N x HybridTransformerBlock (pre-norm, MPS recurrence + FFN)
        -> LayerNorm
        -> Linear LM head (weight-tied with token embedding)
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 512,
        bond_dim: int = 64,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_size, d_model, max_seq_len, dropout)

        d_hidden = bond_dim * 4  # match MPS-LM scaling

        self.layers = nn.ModuleList([
            HybridTransformerBlock(d_model, d_hidden, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        logits = logits / (self.embedding.d_model ** 0.5)
        return logits
