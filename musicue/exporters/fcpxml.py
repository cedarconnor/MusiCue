"""FCPXML 1.10 exporter — markers attached to a placeholder asset.

FCPXML is the interchange format for Final Cut Pro X and DaVinci Resolve.
Resolve 18+ imports it natively as a media-pool asset with markers visible
on the clip.

We emit a single <asset-clip> spanning the song's duration, with one
<marker> per editorial event. Color is encoded via name prefix
(`[Blue] verse`) since the marker XML attribute set varies by reader.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from musicue.exporters._editorial import extract_markers
from musicue.schemas import CueSheet


def _rational_seconds(t: float) -> str:
    """FCPXML uses rational time strings like `4500/24000s`. We use the
    cuesheet's seconds directly with a /1 denominator — most readers accept
    integer-numerator strings. For sub-second precision we multiply by a
    denominator and use the integer result."""
    if t == 0.0:
        return "0s"
    # Use a 24000 denominator to handle 23.976 / 29.97 / 59.94 cleanly.
    num = int(round(t * 24000))
    return f"{num}/24000s"


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    fps = float(opts.get("fps") or cuesheet.fps or 24.0)
    marker_sources = opts.get("marker_sources") or {"section", "transition"}
    if isinstance(marker_sources, str):
        marker_sources = set(marker_sources.split(","))
    else:
        marker_sources = set(marker_sources)
    impulse_names = opts.get("impulse_track_names")
    include_audio = bool(opts.get("include_audio", False))

    markers = extract_markers(cuesheet, marker_sources, impulse_names)

    duration = cuesheet.duration_sec
    # Frame duration string for the format. Standard FCPXML conventions:
    # 24fps → 1/24s, 23.976 → 1001/24000s, 25 → 1/25s, 29.97 → 1001/30000s.
    frame_durations = {
        23.976: "1001/24000s",
        24.0: "1/24s",
        25.0: "1/25s",
        29.97: "1001/30000s",
        30.0: "1/30s",
        50.0: "1/50s",
        59.94: "1001/60000s",
        60.0: "1/60s",
    }
    frame_dur = frame_durations.get(round(fps, 3), f"1/{int(round(fps))}s")

    fcpxml = ET.Element("fcpxml", {"version": "1.10"})
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        {
            "id": "r1",
            "name": f"FFVideoFormat_{int(round(fps * 100)) / 100}",
            "frameDuration": frame_dur,
            "width": "1920",
            "height": "1080",
        },
    )
    ET.SubElement(
        resources,
        "asset",
        {
            "id": "r2",
            "name": "MusiCue song",
            "duration": _rational_seconds(duration),
            "format": "r1",
            "hasAudio": "1" if include_audio else "0",
            "hasVideo": "0",
        },
    )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": "MusiCue markers"})
    project = ET.SubElement(event, "project", {"name": cuesheet.grammar})
    sequence = ET.SubElement(
        project,
        "sequence",
        {
            "format": "r1",
            "duration": _rational_seconds(duration),
            "tcStart": "0s",
        },
    )
    spine = ET.SubElement(sequence, "spine")
    asset_clip = ET.SubElement(
        spine,
        "asset-clip",
        {
            "ref": "r2",
            "offset": "0s",
            "duration": _rational_seconds(duration),
            "name": cuesheet.grammar,
            "format": "r1",
        },
    )

    for m in markers:
        # Color via name prefix — Resolve and FCPX both ignore unknown attrs
        # so this is the most portable encoding.
        marker_name = f"[{m.color}] {m.name}" if m.name else f"[{m.color}]"
        marker_dur = _rational_seconds(max(m.t_end - m.t_start, 1.0 / fps))
        ET.SubElement(
            asset_clip,
            "marker",
            {
                "start": _rational_seconds(m.t_start),
                "duration": marker_dur,
                "value": marker_name,
                "note": m.note,
            },
        )

    # Write with XML declaration + DOCTYPE.
    tree_str = ET.tostring(fcpxml, encoding="unicode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE fcpxml>\n" + tree_str,
        encoding="utf-8",
    )
