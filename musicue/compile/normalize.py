"""Shared robust normalization for energy-like curves."""
from __future__ import annotations

import numpy as np


def percentile_normalize(
    values: list[float],
    low: float = 5.0,
    high: float = 95.0,
    floor: float | None = None,
    flat_value: float = 0.0,
) -> list[float]:
    """Map the ``low``..``high`` percentile band of ``values`` onto [0, 1].

    Values outside the band are clipped. When ``floor`` is given, the
    percentiles are taken over values strictly above it only (e.g. non-silent
    LUFS frames), and values at or below it map to 0.0 — so a long silent
    intro can't drag the reference down and compress everything else toward
    1. ``flat_value`` is used for (non-floor) values when the band collapses.
    """
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    ref = arr[arr > floor] if floor is not None else arr
    if ref.size == 0:
        return [0.0] * len(values)
    lo = float(np.percentile(ref, low))
    hi = float(np.percentile(ref, high))
    if hi - lo < 1e-9:
        out = np.full(arr.shape, float(flat_value))
    else:
        out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if floor is not None:
        out[arr <= floor] = 0.0
    return [float(v) for v in out]
