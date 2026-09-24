"""Derive section-transition ramps from spectral flux and LUFS curves.

For each section boundary (sections[i].start for i >= 1) we measure two pieces
of evidence:

* ``spectral_flux_rise`` -- how much busier the track gets across the
  boundary: (mean flux in the ``flux_window_sec`` after the boundary - mean
  flux in the ``flux_window_sec`` before) / track-wide flux std, squashed to
  [0, 1] with ``0.5 + 0.5 * tanh(z)``. 0.5 = no change, > 0.5 = the new
  section is more active, < 0.5 = it drops back.
* ``lufs_rise_db`` -- LUFS rise across the ``lookback_sec`` window placed
  immediately before the boundary (loudness growth into the boundary).

The ramp itself is still a fixed ``ease_in`` over the last
``0.8 * lookback_sec`` before the boundary; it is not yet shaped by the
evidence (a build that starts 8 bars early still gets a ~1.2 s ramp).
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
    flux_window_sec: float = 2.0,
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
        Width of the LUFS analysis window placed immediately before each
        boundary (also sets the ramp length).
    flux_window_sec:
        Width of the before/after windows compared for ``spectral_flux_rise``.

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

    flux_std = float(np.std(flux_vals)) if len(flux_vals) else 0.0
    flux_win = max(1, int(round(flux_window_sec / flux_hop)))

    transitions = []
    for i in range(1, len(sections)):
        t = sections[i]["start"]
        t_idx = min(int(t / flux_hop), len(flux_vals))
        before = flux_vals[max(0, t_idx - flux_win):t_idx]
        after = flux_vals[t_idx:t_idx + flux_win]
        if len(before) and len(after) and flux_std > 1e-9:
            z = (float(np.mean(after)) - float(np.mean(before))) / flux_std
            flux_rise = 0.5 + 0.5 * float(np.tanh(z))
        else:
            flux_rise = 0.5
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
