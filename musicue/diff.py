from __future__ import annotations

import numpy as np

from musicue.schemas import CueSheet, CueTrack


def _event_times(track: CueTrack) -> list[float]:
    return [
        float(e.get("t") if e.get("t") is not None else e.get("t_start", 0))
        for e in track.events
    ]


def _match_events(times_a: list[float], times_b: list[float], tol: float = 0.05) -> dict:
    matched_b: set[int] = set()
    matched_a: set[int] = set()
    for i, ta in enumerate(times_a):
        for j, tb in enumerate(times_b):
            if j not in matched_b and abs(ta - tb) <= tol:
                matched_a.add(i)
                matched_b.add(j)
                break
    return {
        "matched": len(matched_a),
        "removed": len(times_a) - len(matched_a),
        "added": len(times_b) - len(matched_b),
    }


def _diff_continuous(ta: CueTrack | None, tb: CueTrack | None) -> dict:
    """Diff two continuous tracks: report length and mean absolute difference.

    The result includes ``count_a``/``count_b`` aliases (equal to
    ``length_a``/``length_b``) so callers that consume the per-track stats
    uniformly across track types do not have to special-case the schema.
    """
    vals_a = list(ta.values) if ta is not None and ta.values is not None else []
    vals_b = list(tb.values) if tb is not None and tb.values is not None else []
    len_a, len_b = len(vals_a), len(vals_b)

    # MAE over the overlap; if either side is empty, MAE is undefined.
    if len_a == 0 or len_b == 0:
        mae: float | None = None
    else:
        n = min(len_a, len_b)
        mae = float(np.mean(np.abs(np.array(vals_a[:n]) - np.array(vals_b[:n]))))

    return {
        "type": "continuous",
        "count_a": len_a,
        "count_b": len_b,
        "length_a": len_a,
        "length_b": len_b,
        "mean_abs_diff": mae,
    }


def diff_cuesheets(cs_a: CueSheet, cs_b: CueSheet, tol: float = 0.05) -> dict:
    """Compare two cuesheets track-by-track.

    Impulse / envelope / step / ramp tracks are diffed as event-time sets
    with a ``tol`` tolerance, returning ``count_a``, ``count_b``,
    ``matched``, ``added``, ``removed``. Continuous tracks (no
    timestamped events, just a ``values`` array on a uniform ``hop_sec``)
    are diffed by length and mean absolute difference over the overlap.
    """
    tracks_a = {t.name: t for t in cs_a.tracks}
    tracks_b = {t.name: t for t in cs_b.tracks}
    all_names = set(tracks_a) | set(tracks_b)

    result: dict = {}
    for name in sorted(all_names):
        ta = tracks_a.get(name)
        tb = tracks_b.get(name)
        type_a = ta.type if ta else None
        type_b = tb.type if tb else None

        if type_a == "continuous" or type_b == "continuous":
            result[name] = _diff_continuous(ta, tb)
            continue

        times_a = _event_times(ta) if ta else []
        times_b = _event_times(tb) if tb else []
        m = _match_events(times_a, times_b, tol=tol)
        result[name] = {
            "type": "impulse",
            "count_a": len(times_a),
            "count_b": len(times_b),
            "matched": m["matched"],
            "added": m["added"],
            "removed": m["removed"],
        }
    return result
