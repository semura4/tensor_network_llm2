"""Full MERA Language Model."""

import torch
import torch.nn as nn

from .embedding import TokenAndPositionEmbedding
from .mera_layer import MERABlock


class MERALanguageModel(nn.Module):
    """Language model using MERA (Multi-scale Entanglement Renormalization Ansatz).

    Architecture:
        Token + Position Embedding
        -> N x MERABlock (hierarchical encoder-decoder with MPS at top)
        -> LayerNorm
        -> Linear LM head (weight-tied with token embedding)
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        bond_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_scales: int = 3,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
        d_ff: int = None,
    ):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_size, d_model, max_seq_len, dropout)

        d_hidden = bond_dim * 4
        if d_ff is None:
            d_ff = d_model * 2

        self.layers = nn.ModuleList([
            MERABlock(d_model, d_hidden, num_heads, num_scales, d_ff, dropout)
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
