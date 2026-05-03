"""TouchDesigner CHOP CSV + events CSV exporter.

TouchDesigner consumes timed data via two operators:

* ``CHOP File In`` reads a CSV where the first column is named ``time``
  (TD's convention) and remaining columns are sample-aligned channels.
  We emit a uniform time grid sampled at the finest continuous-track hop
  (defaulting to 0.04s / 25 fps), with one channel per cuesheet track.
  Continuous tracks are linearly interpolated; impulse tracks are stamped
  at the nearest grid index with their ``strength``.

* ``Table DAT`` consumes a separate "events" CSV with columns
  ``track, t, t_end, strength, tags``. Discrete tracks (impulse, envelope,
  step) emit one row per event so timeline-driven logic in TD scripts can
  trigger on exact times rather than reading the resampled CHOP grid.

The CHOP CSV is written to ``<out_path>``; the events CSV is written
beside it as ``<out_path.stem>_events.csv``.
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
    return np.arange(0.0, cuesheet.duration_sec, hop)


def _continuous_col(track: CueTrack, times: np.ndarray) -> list[float]:
    if not track.values or not track.hop_sec:
        return [0.0] * len(times)
    src_t = np.arange(len(track.values)) * track.hop_sec
    return list(np.interp(times, src_t, track.values))


def _impulse_col(track: CueTrack, times: np.ndarray) -> list[float]:
    col = np.zeros(len(times))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    for event in track.events:
        raw_t = event.get("t")
        t = float(raw_t if raw_t is not None else event.get("t_start", 0.0))
        idx = min(int(round(t / hop)), len(col) - 1)
        if idx >= 0:
            col[idx] = float(event.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    """Write a TouchDesigner CHOP CSV and a sibling events CSV."""
    times = _time_grid(cuesheet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[float]] = {"time": list(times)}
    for track in cuesheet.tracks:
        if track.type == "continuous":
            columns[track.name] = _continuous_col(track, times)
        else:
            columns[track.name] = _impulse_col(track, times)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for i in range(len(times)):
            writer.writerow({k: v[i] for k, v in columns.items()})

    events_path = out_path.parent / (out_path.stem + "_events.csv")
    event_rows: list[dict] = []
    for track in cuesheet.tracks:
        if track.type in ("impulse", "envelope"):
            for ev in track.events:
                raw_t = ev.get("t")
                t = float(raw_t if raw_t is not None else ev.get("t_start", 0.0))
                t_end = float(
                    ev.get("t_end", t + float(ev.get("envelope", {}).get("d", 0.1)))
                )
                event_rows.append({
                    "track": track.name,
                    "t": f"{t:.4f}",
                    "t_end": f"{t_end:.4f}",
                    "strength": f"{float(ev.get('strength', 1.0)):.4f}",
                    "tags": "|".join(ev.get("tags", [])),
                })
        elif track.type == "step":
            for ev in track.events:
                event_rows.append({
                    "track": track.name,
                    "t": f"{float(ev['t']):.4f}",
                    "t_end": f"{float(ev['t']):.4f}",
                    "strength": "1.0",
                    "tags": str(ev.get("label", "")),
                })

    if event_rows:
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["track", "t", "t_end", "strength", "tags"]
            )
            writer.writeheader()
            writer.writerows(event_rows)
