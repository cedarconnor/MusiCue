"""disguise / DMX cue list exporter.

Writes a CSV with columns:
- timecode (HH:MM:SS:FF at the configured fps)
- cue_name (auto-generated, e.g. ``kick_0001``)
- track (source track name)
- type (impulse | step | ramp | envelope)
- intensity (0..1)
- duration_frames
- label

Sortable in the d3 cue list editor; rows are timecode-ordered.
"""
from __future__ import annotations

import csv
from pathlib import Path

from musicue.schemas import CueSheet


def _seconds_to_timecode(t: float, fps: float = 25.0) -> str:
    fps_int = max(1, int(fps))
    total_frames = int(round(t * fps))
    ff = total_frames % fps_int
    total_seconds = total_frames // fps_int
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def export(cuesheet: CueSheet, out_path: Path, fps: float = 25.0, **opts) -> None:
    rows: list[dict] = []

    for track in cuesheet.tracks:
        if track.type == "impulse":
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                env = ev.get("envelope", {})
                duration_frames = max(1, int(float(env.get("d", 0.1)) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t, fps),
                    "cue_name": f"{track.name}_{i+1:04d}",
                    "track": track.name,
                    "type": "impulse",
                    "intensity": f"{strength:.4f}",
                    "duration_frames": str(duration_frames),
                    "label": "|".join(ev.get("tags", [])),
                })
        elif track.type == "step":
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                label = str(ev.get("label", ""))
                rows.append({
                    "timecode": _seconds_to_timecode(t, fps),
                    "cue_name": f"section_{label}_{i+1:04d}",
                    "track": track.name,
                    "type": "step",
                    "intensity": "1.0",
                    "duration_frames": "1",
                    "label": label,
                })
        elif track.type == "ramp":
            for i, ev in enumerate(track.events):
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                duration_frames = max(1, int((t_end - t_start) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t_start, fps),
                    "cue_name": f"{track.name}_ramp_{i+1:04d}",
                    "track": track.name,
                    "type": "fade",
                    "intensity": f"{float(ev.get('to', 1.0)):.4f}",
                    "duration_frames": str(duration_frames),
                    "label": str(ev.get("label", "")),
                })
        elif track.type == "envelope":
            for i, ev in enumerate(track.events):
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                duration_frames = max(1, int((t_end - t_start) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t_start, fps),
                    "cue_name": f"{track.name}_{i+1:04d}",
                    "track": track.name,
                    "type": "envelope",
                    "intensity": f"{strength:.4f}",
                    "duration_frames": str(duration_frames),
                    "label": "|".join(ev.get("tags", [])),
                })

    rows.sort(key=lambda r: r["timecode"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timecode", "cue_name", "track", "type", "intensity",
                           "duration_frames", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
