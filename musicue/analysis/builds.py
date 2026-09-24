"""Section energy ranks and anticipation ("build") windows.

Shared by the analysis pipeline (section-transition ramps) and the bundle
builder (``sections[*].energy_rank`` and the ``build`` control curve), so
both agree on which boundaries are builds and where each build starts.

Everything here is pure: plain floats / lists in, plain floats / lists out.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# LUFS frames at or below this are treated as silence (the LUFS curve
# floors digital silence at -70).
LUFS_SILENCE = -60.0

# A boundary is a build when the next section's energy_rank exceeds the
# current one's by at least this much ...
BUILD_MIN_RANK_RISE = 0.2
# ... or when the next section is near the top of the song's range and
# still louder than the current one.
BUILD_HIGH_RANK = 0.75
# Maximum build length, in bars before the boundary.
BUILD_BARS = 8


def section_lufs(
    lufs_values: Sequence[float], lufs_hop: float, start: float, end: float
) -> float | None:
    """Mean LUFS over the section's non-silent frames.

    Silent (-70 floor) frames are excluded so a section with a pause in it
    isn't dragged toward -70; a fully silent section reports the floor.
    """
    if lufs_hop <= 0 or not lufs_values:
        return None
    i0 = max(0, int(start / lufs_hop))
    i1 = min(len(lufs_values), int(end / lufs_hop))
    if i1 <= i0:
        return None
    window = [v for v in lufs_values[i0:i1] if v > LUFS_SILENCE]
    if not window:
        return float(min(lufs_values[i0:i1]))
    return float(sum(window) / len(window))


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def energy_ranks(section_lufs_values: Sequence[float | None]) -> list[float]:
    """Min-max rank (0..1) of per-section LUFS; unknown sections get 0.5."""
    known = [v for v in section_lufs_values if v is not None]
    normalized = iter(_minmax(known))
    return [next(normalized) if v is not None else 0.5 for v in section_lufs_values]


def is_build(current_rank: float, next_rank: float) -> bool:
    """True when the boundary into ``next_rank`` should get a build ramp."""
    return (next_rank - current_rank >= BUILD_MIN_RANK_RISE - 1e-9) or (
        next_rank >= BUILD_HIGH_RANK and next_rank > current_rank
    )


def build_window_start(
    tb: float,
    section_start: float,
    downbeats: Sequence[float],
    bpm: float,
    beats_per_bar: int = 4,
    bars: int = BUILD_BARS,
) -> float:
    """Start of the build window ending at boundary ``tb``.

    ``bars`` bars before ``tb`` measured on the local beat grid: the
    ``bars``-th downbeat before the boundary. A downbeat within half a beat
    before ``tb`` is treated as the boundary's own downbeat (boundaries and
    beat trackers disagree by a few frames). When the grid has fewer than
    ``bars`` downbeats before ``tb``, the local bar length (median spacing of
    the nearest downbeats) is extrapolated; with no usable grid the nominal
    ``bars * beats_per_bar * 60 / bpm`` seconds is used. Never earlier than
    ``section_start``.
    """
    beat_sec = 60.0 / bpm if bpm > 0 else 0.5
    db = sorted(float(t) for t in downbeats)
    before = [t for t in db if t < tb - 0.5 * beat_sec]
    if len(before) >= bars:
        t0 = before[-bars]
    else:
        local = before[-(bars + 1):]
        spacing = np.diff(local) if len(local) >= 2 else np.array([])
        if spacing.size:
            t0 = tb - bars * float(np.median(spacing))
        else:
            t0 = tb - bars * beats_per_bar * beat_sec
    return max(section_start, t0)


def build_windows(
    sections: Sequence[tuple[float, float, float]],
    downbeats: Sequence[float],
    bpm: float,
    beats_per_bar: int = 4,
) -> list[tuple[float, float]]:
    """``(t0, tb)`` build windows for every qualifying section boundary.

    ``sections`` are ``(start, end, energy_rank)`` in time order; boundary
    ``i`` sits at ``sections[i + 1].start``.
    """
    out: list[tuple[float, float]] = []
    for cur, nxt in zip(sections, sections[1:]):
        cur_start, _cur_end, cur_rank = cur
        tb, _next_end, next_rank = nxt
        if not is_build(cur_rank, next_rank):
            continue
        t0 = build_window_start(tb, cur_start, downbeats, bpm, beats_per_bar)
        if tb - t0 > 1e-6:
            out.append((t0, tb))
    return out


def build_curve(
    windows: Sequence[tuple[float, float]], hop_sec: float, n: int
) -> list[float]:
    """Render build windows onto a dense grid (frame ``i`` at ``i * hop_sec``).

    Inside ``[t0, tb)`` the value is ``((t - t0) / (tb - t0)) ** 2``; it
    drops to 0 at ``tb``. Overlapping windows take the max; 0 elsewhere.
    """
    if hop_sec <= 0 or n <= 0:
        return []
    t = np.arange(n) * hop_sec
    out = np.zeros(n)
    for t0, tb in windows:
        span = tb - t0
        if span <= 0:
            continue
        mask = (t >= t0) & (t < tb)
        out[mask] = np.maximum(out[mask], ((t[mask] - t0) / span) ** 2)
    return [float(v) for v in np.clip(out, 0.0, 1.0)]
