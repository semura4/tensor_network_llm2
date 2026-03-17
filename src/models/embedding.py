"""Shared embedding module for both MPS and Transformer models."""

import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    """Token + positional embedding used by both model architectures."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        tok = self.token_emb(x)
        pos = self.pos_emb(positions)
        return self.dropout(tok + pos)
