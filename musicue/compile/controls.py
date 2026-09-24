"""Dense 0..1 control curves for bundle schema 1.3 (``bundle.controls``).

MusiCue measures, CedarToy shapes: every control here is already
normalized to 0..1 and sampled on one shared grid (frame ``i`` at
``i * hop_sec``, ``n = int(duration / hop_sec)`` frames). See
``docs/specs/bundle-1.3-contract.md``.

All functions are pure (lists / floats in, lists out).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from musicue.compile.normalize import percentile_normalize

# dBFS at or below this is silence (energy_fast / brightness masks).
DB_SILENCE = -60.0
# Centered smoothing window for brightness.
BRIGHTNESS_SMOOTH_SEC = 0.15
# Spectral-centroid floor before taking log2 (keeps log finite and > 0 so
# 0.0 can serve as the silence sentinel for percentile_normalize).
_CENTROID_FLOOR_HZ = 20.0
# Drum classes counted by onset_density.
ONSET_DENSITY_CLASSES = ("kick", "snare", "hat")


def resample(values: Sequence[float], src_hop: float, hop_sec: float, n: int) -> np.ndarray:
    """Linearly resample a curve (frame ``j`` at ``j * src_hop``) onto the
    shared grid; ends are held. Empty input → empty array."""
    if not values or src_hop <= 0 or hop_sec <= 0 or n <= 0:
        return np.zeros(0)
    src = np.asarray(values, dtype=float)
    src_t = np.arange(src.size) * src_hop
    return np.interp(np.arange(n) * hop_sec, src_t, src)


def rms_to_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=float), 1e-10))


def energy_fast(rms_db: np.ndarray) -> list[float]:
    """Short-window mix loudness: 5th→0 / 95th→1 over non-silent frames.

    ``rms_db`` is the (already resampled) 100 ms mix RMS in dBFS; frames at
    or below -60 dBFS map to 0.
    """
    return percentile_normalize(
        [float(v) for v in rms_db], 5.0, 95.0, floor=DB_SILENCE, flat_value=0.5
    )


def _centered_nanmean(values: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average of odd width ``k`` ignoring NaNs."""
    if k <= 1:
        return values.copy()
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    kernel = np.ones(k)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full(values.shape, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def brightness(
    centroid_hz: np.ndarray, silent: np.ndarray, hop_sec: float
) -> list[float]:
    """Spectral centroid on a log-frequency axis, smoothed, 5/95-normalized.

    ``silent`` is a boolean mask of silent frames: they are excluded from the
    smoothing and the percentiles, and map to 0.
    """
    if centroid_hz.size == 0:
        return []
    log_c = np.log2(np.maximum(centroid_hz, _CENTROID_FLOOR_HZ))
    log_c[silent] = np.nan
    # Odd box width closest to ~150 ms (3 frames = 120 ms at a 40 ms hop).
    k = 2 * int(round((BRIGHTNESS_SMOOTH_SEC / hop_sec - 1) / 2)) + 1 if hop_sec > 0 else 1
    smoothed = _centered_nanmean(log_c, max(1, k))
    # log2(20 Hz) > 4, so 0.0 is a safe "silent" sentinel under floor=0.
    sentinel = np.where(silent | np.isnan(smoothed), 0.0, smoothed)
    return percentile_normalize(
        [float(v) for v in sentinel], 5.0, 95.0, floor=0.0, flat_value=0.5
    )


def onset_density(
    onset_times: Sequence[float],
    beat_times: Sequence[float],
    hop_sec: float,
    n: int,
    beats_per_bar: int = 4,
) -> list[float]:
    """Drum onsets per beat, smoothed over ~1 bar, 95th-percentile normalized.

    1. Count onsets in each beat interval ``[b_i, b_{i+1})`` (the last beat
       gets the median beat length).
    2. Smooth with a centered window spanning exactly one bar
       (``beats_per_bar + 1`` taps with half-weight ends for even bars,
       ``beats_per_bar`` taps for odd).
    3. Divide by the 95th percentile of the per-beat values, clip to 0..1.
    4. Linearly interpolate between beat-interval midpoints onto the dense
       grid; 0 outside the beat grid.

    Fewer than two beats → all zeros (no grid to count against).
    """
    if hop_sec <= 0 or n <= 0:
        return []
    beats = np.sort(np.asarray(beat_times, dtype=float))
    if beats.size < 2:
        return [0.0] * n
    period = float(np.median(np.diff(beats)))
    edges = np.append(beats, beats[-1] + period)
    counts, _ = np.histogram(np.asarray(onset_times, dtype=float), bins=edges)
    counts = counts.astype(float)

    bpb = max(1, int(beats_per_bar))
    if bpb % 2 == 0:
        # Even bar length: bpb + 1 taps with half-weight ends spans
        # exactly one bar and stays centered.
        kernel = np.ones(bpb + 1)
        kernel[0] = kernel[-1] = 0.5
    else:
        kernel = np.ones(bpb)
    kernel /= kernel.sum()
    # Reflect-pad so the first/last beats aren't pulled toward 0 (edge
    # padding would instead repeat a lone downbeat hit several times).
    pad = min(kernel.size // 2, counts.size - 1)
    padded = np.pad(counts, pad, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="same")[pad:pad + counts.size]

    ref = float(np.percentile(smoothed, 95.0))
    if ref <= 1e-9:
        ref = float(smoothed.max())
    if ref <= 1e-9:
        return [0.0] * n
    per_beat = np.clip(smoothed / ref, 0.0, 1.0)

    mids = (edges[:-1] + edges[1:]) / 2.0
    t = np.arange(n) * hop_sec
    out = np.interp(t, mids, per_beat)
    out[(t < edges[0]) | (t >= edges[-1])] = 0.0
    return [float(v) for v in out]


def grid_length(duration_sec: float, hop_sec: float) -> int:
    if hop_sec <= 0 or not math.isfinite(duration_sec) or duration_sec <= 0:
        return 0
    # Epsilon: 10.0 / 0.04 is 249.999... in floating point.
    return int(duration_sec / hop_sec + 1e-6)
