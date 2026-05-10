"""DaVinci Resolve marker CSV exporter.

Resolve's marker import CSV columns (Edit page → Timeline → Import → Markers):
    #, Source In, Source Out, Track, Type, Note, Color, Source File

Resolve is finicky about CRLF and BOM. We write CRLF + UTF-8-with-BOM
specifically — that's the combo that imports cleanly in Resolve 18+.
"""
from __future__ import annotations

import csv
from pathlib import Path

from musicue.exporters._editorial import extract_markers
from musicue.schemas import CueSheet
from musicue.timecode import t_to_timecode

# Resolve accepts these named colors. Map our category color back to Resolve's
# canonical names (which are exactly the Marker Color picker labels).
_RESOLVE_COLORS = {
    "Blue": "Blue",
    "Red": "Red",
    "Green": "Green",
    "Yellow": "Yellow",
    "Purple": "Purple",
    "Cyan": "Cyan",
    "Pink": "Pink",
    "Cream": "Cream",
    "Lavender": "Lavender",
    "Sky": "Sky",
    "Mint": "Mint",
    "Lemon": "Lemon",
    "Sand": "Sand",
    "Cocoa": "Cocoa",
    "Rose": "Rose",
}


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
    # Resolve wants UTF-8 BOM + CRLF. csv.writer emits \r\n by default when
    # newline="" disables Python's newline translation.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "#", "Source In", "Source Out", "Track", "Type", "Note", "Color", "Source File"
        ])
        for i, m in enumerate(markers, 1):
            tc_in = t_to_timecode(m.t_start, fps, drop_frame)
            tc_out = t_to_timecode(m.t_end, fps, drop_frame)
            color = _RESOLVE_COLORS.get(m.color, "Blue")
            writer.writerow([
                i, tc_in, tc_out, "V1", m.category.title(), m.note, color, ""
            ])
