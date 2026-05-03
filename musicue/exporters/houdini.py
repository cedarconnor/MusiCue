"""Houdini CHOP-compatible CSV exporter.

Format:
- Two leading comment lines (`#`) describe rate/start/end so a Houdini
  artist importing via the File CHOP knows the sample rate
- Header row: ``time,<track1>,<track2>,...``
- Data rows: float values per channel

For impulse/envelope/step/ramp tracks, the corresponding column carries the
event's strength on the matching grid frame; for continuous tracks, the
column carries linearly-interpolated values.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from musicue.schemas import CueSheet, CueTrack


def _time_grid(cuesheet: CueSheet, default_hop: float = 0.04) -> np.ndarray:
    hop = default_hop
    for track in cuesheet.tracks:
        if track.type == "continuous" and track.hop_sec:
            hop = min(hop, track.hop_sec)
    # np.arange preserves hop_sec exactly (np.linspace drifts when duration/hop
    # is not an integer)
    return np.arange(0.0, cuesheet.duration_sec, hop)


def _to_column(track: CueTrack, times: np.ndarray) -> list[float]:
    if track.type == "continuous":
        if not track.values or not track.hop_sec:
            return [0.0] * len(times)
        src_t = np.arange(len(track.values)) * track.hop_sec
        return list(np.interp(times, src_t, track.values))
    # impulse / step / ramp / envelope -> trigger column
    col = np.zeros(len(times))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    for ev in track.events:
        raw_t = ev.get("t")
        t = float(raw_t if raw_t is not None else ev.get("t_start", 0.0))
        idx = min(int(round(t / hop)), len(col) - 1)
        if idx >= 0:
            col[idx] = float(ev.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    times = _time_grid(cuesheet)
    rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 25.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        f.write("# MusiCue Houdini CHOP Export\n")
        f.write(f"# rate={rate:.4f} start=0 end={cuesheet.duration_sec:.4f}\n")
        writer = csv.writer(f)
        headers = ["time"] + [track.name for track in cuesheet.tracks]
        writer.writerow(headers)
        columns = [_to_column(track, times) for track in cuesheet.tracks]
        for i, t in enumerate(times):
            row = [f"{t:.6f}"] + [f"{col[i]:.6f}" for col in columns]
            writer.writerow(row)
