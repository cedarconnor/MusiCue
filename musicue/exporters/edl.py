"""CMX 3600 EDL exporter — point-event markers via 1-frame clips.

The CMX 3600 EDL is the universal interchange format for editorial markers.
Avid, Resolve, and Premiere all read it. Each marker is encoded as a 1-frame
"clip" with `* FROM CLIP NAME`, `* COMMENT`, and `* COLOR` lines beneath.
"""
from __future__ import annotations

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

    fcm = "DROP FRAME" if drop_frame else "NON-DROP FRAME"
    lines: list[str] = [
        f"TITLE: MusiCue cuesheet — grammar={cuesheet.grammar}",
        f"FCM: {fcm}",
        "",
    ]

    for i, m in enumerate(markers, 1):
        # CMX 3600 needs IN ≠ OUT, so always advance OUT by 1 frame.
        tc_in = t_to_timecode(m.t_start, fps, drop_frame)
        # Add 1 frame for OUT — easier than re-doing DF math, just nudge by 1/fps.
        tc_out = t_to_timecode(m.t_start + (1.0 / fps), fps, drop_frame)
        # CMX 3600 line: <event#> <reel> <track> <transition> <src_in> <src_out> <rec_in> <rec_out>
        lines.append(
            f"{i:03d}  MUSICUE  V     C        {tc_in} {tc_out} {tc_in} {tc_out}"
        )
        lines.append(f"* FROM CLIP NAME: {m.name}")
        lines.append(f"* COLOR: {m.color}")
        lines.append(f"* COMMENT: {m.note}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
