"""Premiere Pro marker CSV exporter.

Premiere's stock marker CSV column set:
    Marker Name, Description, In, Out, Duration, Marker Type

Tested against Premiere 2024 — the importer is happy with HH:MM:SS:FF
timecode (no semicolon) and a Marker Type of "Comment" / "Chapter" / etc.
"""
from __future__ import annotations

import csv
from pathlib import Path

from musicue.exporters._editorial import extract_markers
from musicue.schemas import CueSheet
from musicue.timecode import t_to_timecode


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    fps = float(opts.get("fps") or cuesheet.fps or 24.0)
    drop_frame = bool(opts.get("drop_frame", cuesheet.drop_frame))
    marker_sources = opts.get("marker_sources") or {"section", "transition"}
    if isinstance(marker_sources, str):
        marker_sources = set(marker_sources.split(","))
    else:
        marker_sources = set(marker_sources)
    impulse_names = opts.get("impulse_track_names")

    markers = extract_markers(cuesheet, marker_sources, impulse_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Premiere accepts CRLF + UTF-8 (no BOM). csv.writer emits \r\n on its own
    # when newline="" suppresses the default newline translation.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"])
        for m in markers:
            tc_in = t_to_timecode(m.t_start, fps, drop_frame)
            tc_out = t_to_timecode(m.t_end, fps, drop_frame)
            tc_dur = t_to_timecode(max(0.0, m.t_end - m.t_start), fps, drop_frame)
            writer.writerow(
                [m.name, m.note, tc_in, tc_out, tc_dur, "Comment"]
            )
