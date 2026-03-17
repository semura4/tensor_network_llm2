"""GPT-scale models: Standard Transformer and Tensor Network (MPS) variants.

Supports gradient checkpointing, Flash Attention, proper initialization,
and configurable scale from 85M to 350M+ parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mps_core import MultiHeadMPSRecurrence


class GPTEmbedding(nn.Module):
    """Token + learned positional embedding (GPT-2 style)."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        return self.dropout(x)


class GPTAttentionBlock(nn.Module):
    """Standard GPT Transformer block with Flash Attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        h = self.norm1(x)
        B, T, C = h.shape
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, T, D)

        # Flash Attention (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        x = x + self.resid_dropout(self.attn_proj(attn_out))

        # FFN sub-layer
        x = x + self.ffn(self.norm2(x))
        return x


class GPTTNBlock(nn.Module):
    """GPT block with MPS recurrence replacing attention."""

    def __init__(
        self, d_model: int, d_hidden: int, num_heads: int, d_ff: int,
        dropout: float = 0.1, scan_chunk_size: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mps = MultiHeadMPSRecurrence(
            d_model, d_hidden, num_heads, scan_chunk_size=scan_chunk_size,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.mps(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """GPT language model supporting both standard attention and TN variants.

    Args:
        vocab_size: vocabulary size
        d_model: model dimension
        num_heads: number of attention/MPS heads
        d_ff: feed-forward dimension
        num_layers: number of transformer blocks
        max_seq_len: maximum sequence length
        dropout: dropout rate
        tie_weights: tie embedding and LM head weights
        model_type: "transformer" or "tn" (tensor network)
        d_hidden: MPS hidden dim (only for model_type="tn")
        scan_chunk_size: chunk size for parallel scan (0 = no chunking)
        gradient_checkpointing: enable gradient checkpointing
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        num_heads: int = 16,
        d_ff: int = 4096,
        num_layers: int = 24,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        tie_weights: bool = True,
        model_type: str = "transformer",
        d_hidden: int = 768,
        scan_chunk_size: int = 128,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.model_type = model_type

        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_len, dropout)

        if model_type == "transformer":
            self.layers = nn.ModuleList([
                GPTAttentionBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])
        elif model_type == "tn":
            self.layers = nn.ModuleList([
                GPTTNBlock(d_model, d_hidden, num_heads, d_ff, dropout, scan_chunk_size)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.token_emb.weight

        self._init_weights(num_layers)

    def _init_weights(self, num_layers: int):
        """GPT-2 style weight initialization."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name and "W_decay" not in name:
                nn.init.zeros_(p)

        # Residual scaling: 1/sqrt(2*num_layers) on output projections
        scale = 1.0 / math.sqrt(2 * num_layers)
        for layer in self.layers:
            if hasattr(layer, "attn_proj"):
                layer.attn_proj.weight.data *= scale
            if hasattr(layer, "mps"):
                layer.mps.out_proj.weight.data *= scale
            # FFN output projection
            layer.ffn[-2].weight.data *= scale  # Linear before Dropout

        # Re-initialize multi-scale decay biases for TN model
        if self.model_type == "tn":
            for layer in self.layers:
                layer.mps._init_multi_scale()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.final_norm(x)
        return self.lm_head(x)


def build_gpt_model(config: dict) -> GPTModel:
    """Build a GPT model from config dict."""
    mc = config["model"]
    return GPTModel(
        vocab_size=mc.get("vocab_size", 50257),
        d_model=mc["d_model"],
        num_heads=mc["num_heads"],
        d_ff=mc["d_ff"],
        num_layers=mc["num_layers"],
        max_seq_len=mc.get("max_seq_len", 1024),
        dropout=mc.get("dropout", 0.1),
        tie_weights=mc.get("tie_weights", True),
        model_type=mc["type"],
        d_hidden=mc.get("d_hidden", mc["d_model"]),
        scan_chunk_size=mc.get("scan_chunk_size", 128),
        gradient_checkpointing=mc.get("gradient_checkpointing", False),
    )
