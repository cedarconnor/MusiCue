"""ADSR envelope rendering and named ramp/easing curves.

This module provides utilities used by the compiler for generating control
signals from envelope and ramp track definitions.

Exports:
    RAMP_SHAPES: dict mapping shape name -> easing function on [0, 1] -> [0, 1].
        Shapes: linear, ease_in, ease_out, ease_in_out, s_curve, exp_in, exp_out.
    adsr_to_dict: pack ADSR parameters into a dict.
    render_adsr: render an ADSR envelope to a numpy array.
    render_ramp: render a named easing curve to a numpy array of length n.
"""

from __future__ import annotations

import math

import numpy as np

RAMP_SHAPES = {
    "linear":      lambda x: x,
    "ease_in":     lambda x: x * x,
    "ease_out":    lambda x: 1 - (1 - x) ** 2,
    "ease_in_out": lambda x: x * x * (3 - 2 * x),
    "s_curve":     lambda x: x * x * x * (x * (x * 6 - 15) + 10),
    "exp_in":      lambda x: (math.exp(x * 3) - 1) / (math.exp(3) - 1),
    "exp_out":     lambda x: 1 - (math.exp((1 - x) * 3) - 1) / (math.exp(3) - 1),
}


def adsr_to_dict(a: float, d: float, s: float, r: float) -> dict:
    return {"a": a, "d": d, "s": s, "r": r}


def render_adsr(a: float, d: float, s: float, r: float, sr: float = 100.0) -> np.ndarray:
    """Render ADSR envelope to a numpy array at `sr` samples/sec."""
    attack_n = max(1, int(a * sr))
    decay_n = max(1, int(d * sr))
    sustain_hold = max(0, int(0.1 * sr))  # brief sustain window
    release_n = max(1, int(r * sr))
    total = attack_n + decay_n + sustain_hold + release_n
    env = np.zeros(total)
    env[:attack_n] = np.linspace(0, 1, attack_n)
    env[attack_n : attack_n + decay_n] = np.linspace(1, s, decay_n)
    env[attack_n + decay_n : attack_n + decay_n + sustain_hold] = s
    env[attack_n + decay_n + sustain_hold :] = np.linspace(s, 0, release_n)
    return env


def render_ramp(shape: str, n: int) -> np.ndarray:
    fn = RAMP_SHAPES.get(shape, RAMP_SHAPES["linear"])
    return np.array([fn(x) for x in np.linspace(0.0, 1.0, n)])
