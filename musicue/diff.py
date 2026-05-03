from __future__ import annotations

from musicue.schemas import CueSheet


def _event_times(track) -> list[float]:
    return [
        float(e.get("t") if e.get("t") is not None else e.get("t_start", 0))
        for e in track.events
    ]


def _match_events(times_a: list[float], times_b: list[float], tol: float = 0.05) -> dict:
    matched_b = set()
    matched_a = set()
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


def diff_cuesheets(cs_a: CueSheet, cs_b: CueSheet, tol: float = 0.05) -> dict:
    tracks_a = {t.name: t for t in cs_a.tracks}
    tracks_b = {t.name: t for t in cs_b.tracks}
    all_names = set(tracks_a) | set(tracks_b)

    result = {}
    for name in sorted(all_names):
        times_a = _event_times(tracks_a[name]) if name in tracks_a else []
        times_b = _event_times(tracks_b[name]) if name in tracks_b else []
        m = _match_events(times_a, times_b, tol=tol)
        result[name] = {
            "count_a": len(times_a),
            "count_b": len(times_b),
            "matched": m["matched"],
            "added": m["added"],
            "removed": m["removed"],
        }
    return result
