"""MERA (Multi-scale Entanglement Renormalization Ansatz) layer for sequence modeling.

Implements a hierarchical tensor network with:
- Causal disentanglers (local pairwise interactions via gated conv1d)
- Causal isometries (strided causal conv for coarse-graining)
- U-Net encoder-decoder structure with skip connections
- MPS recurrence at the coarsest scale

IMPORTANT: All operations must be strictly causal — position t's output
must never depend on positions t+1, t+2, ....
"""

import torch
import torch.nn as nn

from .mps_core import MultiHeadMPSRecurrence


class CausalDisentangler(nn.Module):
    """Causal disentangler: removes local entanglement via gated convolution.

    Uses two causal conv1d (kernel=2) with grouped convolutions for efficiency.
    Output = sigmoid(conv_gate(x)) * conv_val(x)
    """

    def __init__(self, d_model: int, num_groups: int = 8):
        super().__init__()
        self.conv_gate = nn.Conv1d(
            d_model, d_model, kernel_size=2, padding=1, groups=num_groups,
        )
        self.conv_val = nn.Conv1d(
            d_model, d_model, kernel_size=2, padding=1, groups=num_groups,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xt = x.transpose(1, 2)  # (B, D, T)
        gate = torch.sigmoid(self.conv_gate(xt)[:, :, :T])  # causal trim
        val = self.conv_val(xt)[:, :, :T]
        out = (gate * val).transpose(1, 2)  # (B, T, D)
        return self.norm(out)


class CausalIsometry(nn.Module):
    """Causal isometry: coarse-grains sequence by factor 2.

    Uses a causal strided conv1d (kernel=2, stride=2) so that coarse
    position k depends only on fine positions 2k-1 and 2k (not 2k+1).
    This ensures strict causality.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Causal conv: pad left by kernel_size-1, then stride=2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=2, stride=2, padding=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Left-pad by 1 for causal (position k sees k-1 and k, not k+1)
        xt = x.transpose(1, 2)  # (B, D, T)
        xt = torch.nn.functional.pad(xt, (1, 0))  # (B, D, T+1)
        out = self.conv(xt)  # (B, D, ceil((T+1)/2))
        # Take first T//2 positions (if T is odd, we drop the remainder)
        out_len = T // 2
        out = out[:, :, :out_len].transpose(1, 2)  # (B, T//2, D)
        return self.norm(out)


class CausalInverseIsometry(nn.Module):
    """Causal inverse isometry: expands sequence by factor 2.

    Each coarse position is simply repeated twice — no mixing of
    future information. A learned linear transforms after repeating.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat each position: (B, T, D) -> (B, 2T, D)
        x = x.unsqueeze(2).expand(-1, -1, 2, -1).reshape(x.shape[0], x.shape[1] * 2, x.shape[2])
        return self.norm(self.proj(x))


class MERABlock(nn.Module):
    """One MERA encoder-decoder block.

    Encoder: disentangle -> causal pool at each scale (descending)
    Top: MPS recurrence at coarsest scale
    Decoder: causal un-pool -> disentangle at each scale (ascending) with skip connections
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_heads: int,
        num_scales: int,
        d_ff: int,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        self.num_scales = num_scales

        # Encoder path
        self.enc_disentanglers = nn.ModuleList([
            CausalDisentangler(d_model, num_groups) for _ in range(num_scales)
        ])
        self.enc_isometries = nn.ModuleList([
            CausalIsometry(d_model) for _ in range(num_scales)
        ])

        # Top-level MPS recurrence
        self.top_mps = MultiHeadMPSRecurrence(d_model, d_hidden, num_heads)

        # Decoder path
        self.dec_inv_isometries = nn.ModuleList([
            CausalInverseIsometry(d_model) for _ in range(num_scales)
        ])
        self.dec_disentanglers = nn.ModuleList([
            CausalDisentangler(d_model, num_groups) for _ in range(num_scales)
        ])

        # Skip connection projections
        self.skip_projs = nn.ModuleList([
            nn.Linear(2 * d_model, d_model) for _ in range(num_scales)
        ])

        # FFN sub-layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MERA sub-layer with pre-norm residual
        residual = x
        x = self.norm1(x)

        # --- Encoder (descending) ---
        skips = []
        lengths = []
        for i in range(self.num_scales):
            x = self.enc_disentanglers[i](x)
            skips.append(x)
            lengths.append(x.shape[1])
            x = self.enc_isometries[i](x)

        # --- Top-level MPS recurrence ---
        x = self.top_mps(x)

        # --- Decoder (ascending) ---
        for i in range(self.num_scales - 1, -1, -1):
            x = self.dec_inv_isometries[i](x)
            # Trim to match skip length
            x = x[:, :lengths[i], :]
            # Skip connection: concat + project
            x = self.skip_projs[i](torch.cat([x, skips[i]], dim=-1))
            x = self.dec_disentanglers[i](x)

        x = residual + self.dropout(x)

        # FFN sub-layer
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x
