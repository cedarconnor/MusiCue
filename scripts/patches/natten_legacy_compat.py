"""Legacy NATTEN <0.17 compatibility shim.

Provides pure-PyTorch implementations of:
  natten1dqkrpb, natten1dav, natten2dqkrpb, natten2dav
which were removed in natten >= 0.17. These match the original
neighborhood-attention semantics: each position attends to a window of
`kernel_size` neighbors, with edge positions getting their window shifted
inward (NOT zero-padded).

Memory-efficient: uses index_select to gather neighbors per query position
without materializing the dense (L, L) cross-position tensor — important for
long sequences (e.g. ~6000-frame song spectrograms).
"""
from __future__ import annotations

import torch
from torch import Tensor


def _neighbor_indices_1d(L: int, kernel_size: int, dilation: int, device) -> Tensor:
    """Return (L, kernel_size) tensor of neighbor positions per query index.

    NATTEN's "neighborhood" semantics: when kernel_size*dilation extends beyond
    a boundary, the window slides inward so each query gets exactly kernel_size
    dilated neighbors clamped to [0, L).
    """
    half = kernel_size // 2
    base = torch.arange(L, device=device)
    win_start = base - half * dilation
    win_max_start = (L - 1) - (kernel_size - 1) * dilation
    win_start = torch.clamp(win_start, min=0, max=max(int(win_max_start), 0))
    idx = win_start.unsqueeze(1) + torch.arange(kernel_size, device=device).unsqueeze(0) * dilation
    return idx.clamp_(0, L - 1)


def _rpb_indices_1d(kernel_size: int, dilation: int, L: int, device) -> Tensor:
    """Map each (query_pos, neighbor_idx_in_window) -> bias index in [0, 2k-2]."""
    half = kernel_size // 2
    base = torch.arange(L, device=device)
    win_max_start = (L - 1) - (kernel_size - 1) * dilation
    win_start = torch.clamp(base - half * dilation, min=0, max=max(int(win_max_start), 0))
    abs_neighbors = win_start.unsqueeze(1) + torch.arange(kernel_size, device=device).unsqueeze(0) * dilation
    rel = (abs_neighbors - base.unsqueeze(1)) // max(dilation, 1)
    return (rel + (kernel_size - 1)).clamp_(0, 2 * kernel_size - 2)


def _gather_neighbors_1d(tensor: Tensor, idx: Tensor) -> Tensor:
    """tensor: (B, H, L, D), idx: (L, k) flat indices into L axis.

    Returns (B, H, L, k, D) without ever expanding to (B, H, L, L, D)."""
    B, H, L, D = tensor.shape
    k = idx.shape[-1]
    # Flatten idx -> (L*k,) once, then index_select along the L dim, then reshape.
    flat_idx = idx.reshape(-1)  # (L*k,)
    gathered = tensor.index_select(2, flat_idx)  # (B, H, L*k, D)
    return gathered.view(B, H, L, k, D)


def natten1dqkrpb(query: Tensor, key: Tensor, rpb: Tensor | None,
                  kernel_size: int, dilation: int) -> Tensor:
    """1D neighborhood QK^T + relative positional bias.

    query, key: (B, H, L, D)
    rpb: (H, 2*kernel_size - 1) or None
    Returns: (B, H, L, kernel_size) attention scores (pre-softmax).
    """
    B, H, L, D = query.shape
    device = query.device
    idx = _neighbor_indices_1d(L, kernel_size, dilation, device)  # (L, k)
    key_neighbors = _gather_neighbors_1d(key, idx)  # (B, H, L, k, D)
    scores = (query.unsqueeze(3) * key_neighbors).sum(dim=-1)  # (B, H, L, k)
    if rpb is not None:
        rpb_idx = _rpb_indices_1d(kernel_size, dilation, L, device)  # (L, k)
        bias = rpb[:, rpb_idx]  # (H, L, k)
        scores = scores + bias.unsqueeze(0)
    return scores


def natten1dav(attn: Tensor, value: Tensor, kernel_size: int, dilation: int) -> Tensor:
    """1D neighborhood attention * V.

    attn: (B, H, L, kernel_size) softmaxed weights
    value: (B, H, L, D)
    Returns: (B, H, L, D)
    """
    B, H, L, k = attn.shape
    device = value.device
    idx = _neighbor_indices_1d(L, kernel_size, dilation, device)  # (L, k)
    val_neighbors = _gather_neighbors_1d(value, idx)  # (B, H, L, k, D)
    return (attn.unsqueeze(-1) * val_neighbors).sum(dim=3)


def _gather_neighbors_2d(tensor: Tensor, row_idx: Tensor, col_idx: Tensor) -> Tensor:
    """tensor: (B, head, Hh, Ww, D)
    row_idx: (Hh, k), col_idx: (Ww, k) per-axis neighbor indices.

    Returns (B, head, Hh, Ww, k*k, D) without expanding to (Hh*Ww, Hh*Ww)."""
    B, head, Hh, Ww, D = tensor.shape
    k_h = row_idx.shape[-1]
    k_w = col_idx.shape[-1]
    # Step 1: gather along H axis -> (B, head, Hh, k_h, Ww, D)
    flat_row = row_idx.reshape(-1)  # (Hh * k_h,)
    g = tensor.index_select(2, flat_row)  # (B, head, Hh*k_h, Ww, D)
    g = g.view(B, head, Hh, k_h, Ww, D)
    # Step 2: gather along W axis -> (B, head, Hh, k_h, Ww, k_w, D)
    flat_col = col_idx.reshape(-1)  # (Ww * k_w,)
    g = g.index_select(4, flat_col)  # (B, head, Hh, k_h, Ww*k_w, D)
    g = g.view(B, head, Hh, k_h, Ww, k_w, D)
    # Permute to (B, head, Hh, Ww, k_h, k_w, D) then collapse k_h*k_w
    g = g.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return g.view(B, head, Hh, Ww, k_h * k_w, D)


def natten2dqkrpb(query: Tensor, key: Tensor, rpb: Tensor | None,
                  kernel_size: int, dilation: int) -> Tensor:
    """2D neighborhood QK^T + relative positional bias.

    query, key: (B, head, Hh, Ww, D)
    rpb: (head, 2k-1, 2k-1) or None
    Returns: (B, head, Hh, Ww, k*k)
    """
    B, head, Hh, Ww, D = query.shape
    device = query.device
    row_idx = _neighbor_indices_1d(Hh, kernel_size, dilation, device)  # (Hh, k)
    col_idx = _neighbor_indices_1d(Ww, kernel_size, dilation, device)  # (Ww, k)
    key_neighbors = _gather_neighbors_2d(key, row_idx, col_idx)  # (B, head, Hh, Ww, k*k, D)
    q_exp = query.unsqueeze(4)  # (B, head, Hh, Ww, 1, D)
    scores = (q_exp * key_neighbors).sum(dim=-1)  # (B, head, Hh, Ww, k*k)
    if rpb is not None:
        rpb_row = _rpb_indices_1d(kernel_size, dilation, Hh, device)  # (Hh, k)
        rpb_col = _rpb_indices_1d(kernel_size, dilation, Ww, device)  # (Ww, k)
        ri = rpb_row.view(Hh, 1, kernel_size, 1).expand(Hh, Ww, kernel_size, kernel_size)
        ci = rpb_col.view(1, Ww, 1, kernel_size).expand(Hh, Ww, kernel_size, kernel_size)
        bias = rpb[:, ri, ci]  # (head, Hh, Ww, k, k)
        bias = bias.reshape(head, Hh, Ww, kernel_size * kernel_size)
        scores = scores + bias.unsqueeze(0)
    return scores


def natten2dav(attn: Tensor, value: Tensor, kernel_size: int, dilation: int) -> Tensor:
    """2D neighborhood attention * V.

    attn: (B, head, Hh, Ww, k*k)
    value: (B, head, Hh, Ww, D)
    Returns: (B, head, Hh, Ww, D)
    """
    B, head, Hh, Ww, kk = attn.shape
    device = value.device
    row_idx = _neighbor_indices_1d(Hh, kernel_size, dilation, device)  # (Hh, k)
    col_idx = _neighbor_indices_1d(Ww, kernel_size, dilation, device)  # (Ww, k)
    val_neighbors = _gather_neighbors_2d(value, row_idx, col_idx)  # (B, head, Hh, Ww, k*k, D)
    return (attn.unsqueeze(-1) * val_neighbors).sum(dim=4)


_legacy_natten_compat = True
