"""Matrix Product State (MPS) / Tensor Train core operations.

Parallel implementation with selective state transitions (Mamba-style):
- Input-dependent decay: a_t = sigmoid(W_decay(x_t)) instead of fixed exp(log_a)
- Multi-scale head initialization: different decay rates per head
- Output gating: SiLU-gated output path
- Causal depthwise conv1d for local context
- Parallel prefix scan in O(log T) steps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_scan import parallel_scan_simple


class MultiHeadMPSRecurrence(nn.Module):
    """Multi-head MPS recurrence with selective state transitions.

    Improvements over basic diagonal recurrence:
    1. Input-dependent decay (selective scan): decay rate adapts to input
    2. Multi-scale initialization: heads specialize in different time-scales
    3. Output gating: multiplicative SiLU gate on output
    4. Causal conv1d: local context before recurrence
    """

    def __init__(self, d_model: int, d_hidden: int, num_heads: int = 4,
                 conv_kernel: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_hidden // num_heads
        self.d_hidden = d_hidden

        # Input projection
        self.W_ih = nn.Linear(d_model, d_hidden)

        # Input gate
        self.W_gate = nn.Linear(d_model, d_hidden)

        # Selective (input-dependent) decay
        self.W_decay = nn.Linear(d_model, d_hidden)

        # Output gate (SiLU-gated)
        self.W_z = nn.Linear(d_model, d_hidden)

        # Causal depthwise conv1d for local context
        self.conv1d = nn.Conv1d(
            d_hidden, d_hidden, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_hidden,
        )

        # Layer norm + output projections
        self.ln = nn.LayerNorm(d_hidden)
        self.W_out = nn.Linear(d_hidden, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_model)

        self._init_multi_scale()

    def _init_multi_scale(self):
        """Initialize W_decay bias for multi-scale time constants.

        Head 0: fast decay (local patterns)  — sigmoid bias ~0.73
        Head 1: medium decay                 — sigmoid bias ~0.92
        Head 2: slow decay (long-range)      — sigmoid bias ~0.98
        Head 3: very slow decay (memory)     — sigmoid bias ~0.996
        """
        scales = torch.linspace(1.0, 5.5, self.num_heads)
        bias = torch.zeros(self.d_hidden)
        for i, s in enumerate(scales):
            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            bias[start:end] = s
        self.W_decay.bias.data.copy_(bias)
        # Small weight init so initial decay is dominated by bias
        nn.init.normal_(self.W_decay.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Input projection + causal conv1d
        ih = self.W_ih(x)                                  # (B, T, D)
        ih = self.conv1d(ih.transpose(1, 2))[:, :, :T].transpose(1, 2)  # causal trim

        # Gates
        gate = torch.sigmoid(self.W_gate(x))               # (B, T, D)

        # Selective (input-dependent) decay
        a = torch.sigmoid(self.W_decay(x))                  # (B, T, D)

        # Output gate
        z = F.silu(self.W_z(x))                             # (B, T, D)

        # Recurrence coefficients: h_t = alpha_t * h_{t-1} + beta_t
        alpha = (1.0 - gate) * a                            # (B, T, D)
        beta = gate * ih                                     # (B, T, D)

        # Parallel scan
        outputs = parallel_scan_simple(alpha, beta)          # (B, T, D)

        # Gated output
        outputs = z * self.ln(outputs)
        outputs = self.W_out(outputs)
        return self.out_proj(outputs)
