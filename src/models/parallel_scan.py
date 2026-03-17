"""Parallel scan (Blelloch prefix sum) for linear recurrences.

Computes the linear recurrence:
    h_t = a_t * h_{t-1} + b_t

in O(log T) sequential steps instead of O(T), using the associativity of
the composition (a1, b1) ⊗ (a2, b2) = (a1 * a2, a2 * b1 + b2).

All operations are element-wise, so a_t and b_t can be arbitrary-shaped
tensors as long as they share the same shape across the sequence dimension.
"""

import torch


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute linear recurrence h_t = a_t * h_{t-1} + b_t via parallel scan.

    Args:
        a: (batch, seq_len, d_hidden) — multiplicative coefficients
        b: (batch, seq_len, d_hidden) — additive inputs
        Both must have the same shape.

    Returns:
        h: (batch, seq_len, d_hidden) — recurrence outputs h_1, h_2, ..., h_T
           where h_0 = 0 (zero initial state).
    """
    # Up-sweep (reduce) phase: combine pairs bottom-up
    # Down-sweep phase: propagate results top-down
    # We implement the "work-efficient" Blelloch scan.
    #
    # For simplicity and GPU efficiency, we use the recursive doubling
    # approach which is slightly less work-efficient but more parallelism-friendly.

    B, T, D = a.shape

    # Pad to next power of 2 for clean recursive doubling
    log2T = (T - 1).bit_length()
    T_pad = 1 << log2T

    if T_pad > T:
        pad = T_pad - T
        a = torch.cat([a, a.new_ones(B, pad, D)], dim=1)
        b = torch.cat([b, b.new_zeros(B, pad, D)], dim=1)

    # Clone to avoid in-place issues with autograd
    aa = a.clone()
    bb = b.clone()

    # Up-sweep: combine elements at increasing strides
    for d in range(log2T):
        stride = 1 << d
        # Indices: combine (i - stride) into (i) for i = stride*2-1, stride*2*2-1, ...
        # Simpler: at each level, combine pairs
        idx = torch.arange(stride, T_pad, stride * 2, device=a.device)
        idx_prev = idx - stride

        # (a_prev, b_prev) ⊗ (a_cur, b_cur) = (a_prev * a_cur, a_cur * b_prev + b_cur)
        a_prev = aa[:, idx_prev]
        b_prev = bb[:, idx_prev]
        a_cur = aa[:, idx]
        b_cur = bb[:, idx]

        new_a = a_prev * a_cur
        new_b = a_cur * b_prev + b_cur

        aa = aa.clone()
        bb = bb.clone()
        aa[:, idx] = new_a
        bb[:, idx] = new_b

    # Down-sweep: propagate partial results
    # Set root to identity
    aa_down = aa.clone()
    bb_down = bb.clone()
    aa_down[:, T_pad - 1] = 1.0
    bb_down[:, T_pad - 1] = 0.0

    for d in range(log2T - 1, -1, -1):
        stride = 1 << d
        idx = torch.arange(stride, T_pad, stride * 2, device=a.device)
        idx_prev = idx - stride

        # Save current values
        a_left = aa_down[:, idx_prev].clone()
        b_left = bb_down[:, idx_prev].clone()
        a_right = aa_down[:, idx].clone()
        b_right = bb_down[:, idx].clone()

        # Left child gets parent's value
        aa_down_new = aa_down.clone()
        bb_down_new = bb_down.clone()
        aa_down_new[:, idx_prev] = a_right
        bb_down_new[:, idx_prev] = b_right

        # Right child gets combination
        aa_down_new[:, idx] = a_left * a_right
        bb_down_new[:, idx] = a_right * b_left + b_right

        aa_down = aa_down_new
        bb_down = bb_down_new

    # Final result: apply the scan coefficients to original values
    h = aa_down * b + bb_down

    # Remove padding
    return h[:, :T]


def parallel_scan_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simpler parallel scan using recursive doubling (Hillis-Steele style).

    Less memory-efficient but cleaner implementation, better for autograd.

    Args:
        a: (batch, seq_len, d_hidden) — multiplicative coefficients
        b: (batch, seq_len, d_hidden) — additive inputs

    Returns:
        h: (batch, seq_len, d_hidden) — recurrence outputs
    """
    B, T, D = a.shape

    # We'll compute prefix products of a and use them to combine b values.
    # h_t = sum_{k=1}^{t} (prod_{j=k+1}^{t} a_j) * b_k
    #
    # Using the recursive doubling approach:
    # At each step d, we combine element i with element i - 2^d
    # After log2(T) steps, each element has accumulated all predecessors.

    aa = a.clone()
    bb = b.clone()

    for d in range(T.bit_length()):
        stride = 1 << d
        if stride >= T:
            break

        # Elements at position stride and beyond get combined with predecessors
        aa_shifted = torch.cat([aa.new_ones(B, stride, D), aa[:, :-stride]], dim=1)
        bb_shifted = torch.cat([bb.new_zeros(B, stride, D), bb[:, :-stride]], dim=1)

        # Combine: (aa_shifted, bb_shifted) ⊗ (aa, bb)
        # = (aa_shifted * aa, aa * bb_shifted + bb)
        new_bb = aa * bb_shifted + bb
        new_aa = aa_shifted * aa

        # Only update positions that have a valid predecessor at this stride
        mask = torch.arange(T, device=a.device) >= stride
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        aa = torch.where(mask, new_aa, aa)
        bb = torch.where(mask, new_bb, bb)

    return bb
