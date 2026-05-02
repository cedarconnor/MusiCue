"""Derive section-transition ramps from spectral flux and LUFS curves.

For each section boundary (sections[i].start for i >= 1), we look back a short
window (default 1.5 s) and measure two pieces of evidence: the peak spectral
flux relative to the track-wide mean (a proxy for "build" intensity), and the
LUFS rise across the window (a proxy for loudness growth into the boundary).
The output is a list of transition dicts shaped to match the M1
``SectionTransition`` schema -- note the dict key ``"from"`` matches the
schema's alias-based parsing on the pydantic side.
"""
from __future__ import annotations

import numpy as np


def derive_transitions(
    sections: list[dict],
    spectral_flux: dict,
    lufs: dict,
    lookback_sec: float = 1.5,
) -> list[dict]:
    """Emit one transition dict per section boundary.

    Parameters
    ----------
    sections:
        Macro-timescale section dicts with ``start``/``end``/``label``.
        Boundaries are taken at ``sections[i]["start"]`` for ``i >= 1``.
    spectral_flux, lufs:
        Curve dicts shaped ``{"hop_sec": float, "values": list[float]}``.
    lookback_sec:
        Width of the analysis window placed immediately before each boundary.

    Returns
    -------
    list[dict]
        One dict per transition, with keys ``t``, ``from``, ``to``, ``ramp``,
        and ``ramp_evidence``. Returns ``[]`` when fewer than two sections
        are supplied.
    """
    if len(sections) < 2:
        return []

    flux_hop = spectral_flux["hop_sec"]
    flux_vals = np.array(spectral_flux["values"])
    lufs_hop = lufs["hop_sec"]
    lufs_vals = np.array(lufs["values"])

    transitions = []
    for i in range(1, len(sections)):
        t = sections[i]["start"]
        lookback_frames = int(lookback_sec / flux_hop)
        t_idx = min(int(t / flux_hop), len(flux_vals) - 1)
        start_idx = max(0, t_idx - lookback_frames)

        window_flux = flux_vals[start_idx:t_idx] if t_idx > start_idx else np.array([0.0])
        flux_rise = (
            float(np.max(window_flux) / (np.mean(flux_vals) + 1e-9))
            if len(window_flux) > 0
            else 0.0
        )
        flux_rise = float(np.clip(flux_rise, 0.0, 1.0))

        lufs_t_idx = min(int(t / lufs_hop), len(lufs_vals) - 1)
        lufs_start_idx = max(0, lufs_t_idx - int(lookback_sec / lufs_hop))
        window_lufs = lufs_vals[lufs_start_idx:lufs_t_idx]
        lufs_rise = float(window_lufs[-1] - window_lufs[0]) if len(window_lufs) >= 2 else 0.0

        ramp_start = max(0.0, t - lookback_sec * 0.8)
        transitions.append({
            "t": float(t),
            "from": sections[i - 1]["label"],
            "to": sections[i]["label"],
            "ramp": {"t_start": ramp_start, "t_end": float(t), "shape": "ease_in"},
            "ramp_evidence": {
                "spectral_flux_rise": flux_rise,
                "lufs_rise_db": lufs_rise,
            },
        })
    return transitions
