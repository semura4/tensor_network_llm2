"""Full MPS Language Model."""

import torch
import torch.nn as nn

from .embedding import TokenAndPositionEmbedding
from .mps_layer import MPSSequenceLayer


class MPSLanguageModel(nn.Module):
    """Language model using MPS/Tensor Train-inspired recurrence layers.

    Architecture:
        Token + Position Embedding
        -> N x MPSSequenceLayer (with residual + LayerNorm)
        -> LayerNorm
        -> Linear LM head (weight-tied with token embedding)
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        d_local: int = 32,  # unused, kept for config compat
        bond_dim: int = 64,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
        d_ff: int = None,
    ):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Use bond_dim as hidden dimension for MPS recurrence
        d_hidden = bond_dim * 4  # scale up for capacity
        num_heads = 4

        self.layers = nn.ModuleList([
            MPSSequenceLayer(d_model, d_hidden, num_heads, dropout, d_ff=d_ff)
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
        # Scale logits to prevent overconfident initial predictions with tied weights
        logits = logits / (self.embedding.d_model ** 0.5)
        return logits
