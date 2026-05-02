from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from musicue.schemas import CueSheet, CueTrack


def _time_grid(cuesheet: CueSheet, default_hop: float = 0.04) -> np.ndarray:
    hops = [track.hop_sec for track in cuesheet.tracks
            if track.type == "continuous" and track.hop_sec]
    hop = min(hops) if hops else default_hop
    n = max(1, int(np.ceil(cuesheet.duration_sec / hop)))
    return np.linspace(0.0, cuesheet.duration_sec, n, endpoint=False)


def _continuous_col(track: CueTrack, times: np.ndarray) -> list[float]:
    if not track.values or not track.hop_sec:
        return [0.0] * len(times)
    src_t = np.arange(len(track.values)) * track.hop_sec
    return list(np.interp(times, src_t, track.values))


def _impulse_col(track: CueTrack, times: np.ndarray) -> list[float]:
    col = np.zeros(len(times))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    for event in track.events:
        t = event.get("t") or event.get("t_start", 0.0)
        idx = int(round(float(t) / hop))
        if 0 <= idx < len(col):
            col[idx] = float(event.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    times = _time_grid(cuesheet)
    columns: dict[str, list[float]] = {"time_sec": list(times)}
    for track in cuesheet.tracks:
        if track.type == "continuous":
            columns[track.name] = _continuous_col(track, times)
        else:
            columns[track.name] = _impulse_col(track, times)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for i in range(len(times)):
            writer.writerow({k: v[i] for k, v in columns.items()})
