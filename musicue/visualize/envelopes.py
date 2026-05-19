"""Pure-math helpers for sampling ADSR envelopes and easing curves."""
from __future__ import annotations

from typing import Mapping


def sample_adsr(env: Mapping[str, float], t_since_event: float) -> float:
    """Sample an ADSR envelope at time `t_since_event` after the event fired.

    Returns a value in [0, 1]. Outside the envelope (negative or after
    release) returns 0.0. When release is 0, the envelope ends at the
    end of decay (impulse style).
    """
    if t_since_event < 0.0:
        return 0.0
    a = float(env.get("a", 0.0))
    d = float(env.get("d", 0.0))
    s = float(env.get("s", 0.0))
    r = float(env.get("r", 0.0))

    if t_since_event <= a:
        return t_since_event / a if a > 0 else 1.0
    t = t_since_event - a
    if t <= d:
        return 1.0 - (1.0 - s) * (t / d) if d > 0 else s
    t -= d
    if r <= 0.0:
        return 0.0
    if t <= r:
        return s * (1.0 - t / r)
    return 0.0


def ease(shape: str, x: float) -> float:
    """Apply a named easing function to x in [0, 1]. Unknown -> linear."""
    x = max(0.0, min(1.0, x))
    if shape == "linear":
        return x
    if shape == "ease_in":
        return x * x
    if shape == "ease_out":
        return 1.0 - (1.0 - x) * (1.0 - x)
    if shape == "ease_in_out":
        if x < 0.5:
            return 2.0 * x * x
        return 1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    return x
