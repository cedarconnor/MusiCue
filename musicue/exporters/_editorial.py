"""Shared marker extraction for editorial exporters (EDL/FCPXML/Premiere/Resolve).

Each editorial format needs the same source data — a flat list of `Marker`
objects with name, time range, category, and color — formatted differently.
This module produces that list from a CueSheet so the format-specific
exporters stay tiny.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from musicue.schemas import CueSheet
from musicue.timecode import t_to_frame, t_to_timecode

MarkerCategory = Literal["section", "transition", "impulse", "envelope"]

# Default colors per category. Editorial formats encode color differently;
# each exporter maps these to its own scheme (Resolve named colors, FCPXML
# name-prefix convention, EDL * COLOR comments, Premiere none).
DEFAULT_COLORS: dict[MarkerCategory, str] = {
    "section": "Blue",
    "transition": "Red",
    "impulse": "Green",
    "envelope": "Yellow",
}


@dataclass
class Marker:
    name: str
    note: str
    t_start: float
    t_end: float  # equals t_start for point markers
    category: MarkerCategory
    color: str

    def frame_start(self, fps: float) -> int:
        return t_to_frame(self.t_start, fps)

    def frame_end(self, fps: float) -> int:
        return t_to_frame(self.t_end, fps)

    def timecode_start(self, fps: float, drop_frame: bool) -> str:
        return t_to_timecode(self.t_start, fps, drop_frame)

    def timecode_end(self, fps: float, drop_frame: bool) -> str:
        return t_to_timecode(self.t_end, fps, drop_frame)


def extract_markers(
    cs: CueSheet,
    marker_sources: set[str] | None = None,
    impulse_track_names: set[str] | None = None,
) -> list[Marker]:
    """Extract a flat list of editorial markers from a cuesheet.

    Args:
        cs: The cuesheet to read.
        marker_sources: Categories to include. Default: {"section", "transition"}.
            Add "impulse" or "envelope" to include per-event markers from those
            track types.
        impulse_track_names: When `marker_sources` includes "impulse", only emit
            markers from these track names (default: all impulse tracks).
            Useful to limit to e.g. {"drop", "downbeat", "kick_pulse"} so the
            timeline doesn't get flooded with one marker per beat.

    Returns:
        A list of markers sorted by `t_start`.
    """
    if marker_sources is None:
        marker_sources = {"section", "transition"}

    markers: list[Marker] = []

    for track in cs.tracks:
        if track.type == "step" and "section" in marker_sources:
            for ev in track.events:
                t = float(ev.get("t", 0.0))
                label = str(ev.get("label", track.name))
                markers.append(
                    Marker(
                        name=label,
                        note=f"Section: {label}",
                        t_start=t,
                        t_end=t,
                        category="section",
                        color=DEFAULT_COLORS["section"],
                    )
                )
        elif track.type == "ramp" and "transition" in marker_sources:
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start))
                label = str(ev.get("label", track.name))
                shape = str(ev.get("shape", "linear"))
                markers.append(
                    Marker(
                        name=label or track.name,
                        note=f"Transition ({shape}): {label}",
                        t_start=t_start,
                        t_end=t_end,
                        category="transition",
                        color=DEFAULT_COLORS["transition"],
                    )
                )
        elif track.type == "impulse" and "impulse" in marker_sources:
            if impulse_track_names is not None and track.name not in impulse_track_names:
                continue
            for ev in track.events:
                t = float(ev.get("t", 0.0))
                markers.append(
                    Marker(
                        name=track.name,
                        note=f"Impulse: {track.name}",
                        t_start=t,
                        t_end=t,
                        category="impulse",
                        color=DEFAULT_COLORS["impulse"],
                    )
                )
        elif track.type == "envelope" and "envelope" in marker_sources:
            for ev in track.events:
                t_start = float(ev.get("t_start", ev.get("t", 0.0)))
                t_end = float(ev.get("t_end", t_start))
                label = str(ev.get("label", track.name))
                markers.append(
                    Marker(
                        name=label or track.name,
                        note=f"Envelope: {label}",
                        t_start=t_start,
                        t_end=t_end,
                        category="envelope",
                        color=DEFAULT_COLORS["envelope"],
                    )
                )

    markers.sort(key=lambda m: m.t_start)
    return markers
