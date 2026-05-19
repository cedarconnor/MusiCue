"""Color choices for cue-video lanes and section labels.

All colors are RGB tuples in [0, 1], matplotlib-compatible.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

RGB = Tuple[float, float, float]


_SECTION_PALETTE: dict[str, RGB] = {
    "intro": (0.35, 0.55, 0.95),
    "verse": (0.40, 0.80, 0.45),
    "chorus": (0.95, 0.85, 0.30),
    "bridge": (0.70, 0.45, 0.90),
    "drop": (0.95, 0.40, 0.40),
    "outro": (0.55, 0.55, 0.55),
}

_GREY: RGB = (0.55, 0.55, 0.55)


_TYPE_TINT: dict[str, RGB] = {
    "impulse": (0.10, 0.10, 0.12),
    "envelope": (0.16, 0.12, 0.10),
    "step": (0.10, 0.13, 0.16),
    "ramp": (0.14, 0.10, 0.16),
    "continuous": (0.12, 0.12, 0.12),
}


def track_color(name: str) -> RGB:
    """Stable, vivid color hashed from a track name (HSV with fixed s/v)."""
    h = hashlib.md5(name.encode("utf-8")).digest()
    hue = h[0] / 255.0
    s = 0.75
    v = 0.95
    i = int(hue * 6.0)
    f = hue * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def section_palette(label: str) -> RGB:
    """Map a section label to a fixed color. Fuzzy prefix match."""
    key = (label or "").strip().lower()
    if not key:
        return _GREY
    for prefix, color in _SECTION_PALETTE.items():
        if key.startswith(prefix):
            return color
    return _GREY


def type_tint(track_type: str) -> RGB:
    """Background tint for a lane based on its CueTrack.type."""
    return _TYPE_TINT.get(track_type, (0.10, 0.10, 0.10))
