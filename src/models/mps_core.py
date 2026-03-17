"""Matrix Product State (MPS) / Tensor Train core operations.

Parallel implementation using diagonal state transitions and prefix scan.
The recurrence h_t = a_t * h_{t-1} + b_t is computed in O(log T) steps
instead of O(T) sequential steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_scan import parallel_scan_simple


class MPSRecurrence(nn.Module):
    """MPS-based recurrence with parallel scan.

    Uses diagonal state transition (element-wise) instead of full matrix,
    enabling parallel prefix scan computation in O(log T) sequential steps.

    The recurrence is:
        h_t = (1 - g_t) * diag(a) * h_{t-1} + g_t * W_ih(x_t)

    This is reformulated as: h_t = alpha_t * h_{t-1} + beta_t
    where alpha_t = (1 - g_t) * a,  beta_t = g_t * W_ih(x_t)
    and solved via parallel scan.
    """

    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        super().__init__()
        self.d_hidden = d_hidden

        # Input-to-hidden projection
        self.W_ih = nn.Linear(d_input, d_hidden)

        # Diagonal state transition: learnable per-dimension decay
        # Initialized near 1 for stable long-range propagation
        self.log_a = nn.Parameter(torch.zeros(d_hidden) - 0.1)

        # Input gate
        self.W_gate = nn.Linear(d_input, d_hidden)

        # Output projection
        self.W_out = nn.Linear(d_hidden, d_output)

        # Layer norm
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_input)
        Returns:
            (batch, seq_len, d_output)
        """
        B, T, _ = x.shape

        # Pre-compute all projections in parallel (no sequential dependency)
        ih = self.W_ih(x)                        # (B, T, d_hidden)
        gate = torch.sigmoid(self.W_gate(x))     # (B, T, d_hidden)

        # Diagonal state transition coefficient
        a = torch.exp(self.log_a)                 # (d_hidden,) values near 1

        # Recurrence coefficients:
        #   h_t = alpha_t * h_{t-1} + beta_t
        alpha = (1.0 - gate) * a.unsqueeze(0).unsqueeze(0)  # (B, T, d_hidden)
        beta = gate * ih                                      # (B, T, d_hidden)

        # Parallel scan: compute all h_t in O(log T) steps
        outputs = parallel_scan_simple(alpha, beta)  # (B, T, d_hidden)

        # Layer norm + output projection
        outputs = self.ln(outputs)
        return self.W_out(outputs)


class MultiHeadMPSRecurrence(nn.Module):
    """Multi-head MPS recurrence with parallel scan.

    All heads are computed in a single fused pass (no separate loops).
    """

    def __init__(self, d_model: int, d_hidden: int, num_heads: int = 4):
        super().__init__()
        assert d_hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_hidden // num_heads
        self.d_hidden = d_hidden

        # Fused projections for all heads
        self.W_ih = nn.Linear(d_model, d_hidden)
        self.W_gate = nn.Linear(d_model, d_hidden)

        # Per-dimension diagonal decay (shared structure, independent per head)
        self.log_a = nn.Parameter(torch.zeros(d_hidden) - 0.1)

        # Per-head layer norms
        self.ln = nn.LayerNorm(d_hidden)

        # Output projections
        self.W_out = nn.Linear(d_hidden, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Fused input projections for all heads
        ih = self.W_ih(x)                         # (B, T, d_hidden)
        gate = torch.sigmoid(self.W_gate(x))      # (B, T, d_hidden)

        # Diagonal state transition
        a = torch.exp(self.log_a)

        # Recurrence coefficients
        alpha = (1.0 - gate) * a.unsqueeze(0).unsqueeze(0)
        beta = gate * ih

        # Single parallel scan over all heads (fused)
        outputs = parallel_scan_simple(alpha, beta)

        # Layer norm + output
        outputs = self.ln(outputs)
        outputs = self.W_out(outputs)
        return self.out_proj(outputs)
