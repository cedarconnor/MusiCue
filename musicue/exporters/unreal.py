"""Unreal Sequencer JSON exporter.

Output schema (version 1.0): a single dict with::

  {
    "schema_version": "1.0",
    "generator": "MusiCue",
    "grammar": <name>,
    "duration_sec": <float>,
    "tempo_map": [...],
    "tracks": [<track>, ...]
  }

Each track is one of:
- ``event_track`` -- discrete events (impulse, envelope, step) with metadata
- ``float_curve`` -- keyed values (ramp, continuous) for Sequencer Float Tracks

A small Unreal Python helper (not shipped) reads this JSON and uses
``unreal.SequencerScriptingExtensions`` to populate a Level Sequence.
"""
from __future__ import annotations

import json
from pathlib import Path

from musicue.exporters._common import non_empty_tracks
from musicue.schemas import CueSheet


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    tracks = []

    for track in non_empty_tracks(cuesheet.tracks):
        if track.type == "impulse":
            events = [
                {
                    "time": float(ev["t"]),
                    "strength": float(ev.get("strength", 1.0)),
                    "envelope": ev.get("envelope", {}),
                    "tags": ev.get("tags", []),
                }
                for ev in track.events
            ]
            tracks.append(
                {
                    "name": track.name,
                    "type": "event_track",
                    "timescale": track.timescale,
                    "events": events,
                }
            )

        elif track.type == "envelope":
            events = [
                {
                    "time_start": float(ev.get("t_start", 0.0)),
                    "time_end": float(ev.get("t_end", 0.0)),
                    "strength": float(ev.get("strength", 0.8)),
                    "envelope": ev.get("envelope", {}),
                }
                for ev in track.events
            ]
            tracks.append(
                {
                    "name": track.name,
                    "type": "event_track",
                    "timescale": track.timescale,
                    "events": events,
                }
            )

        elif track.type == "step":
            events = [
                {
                    "time": float(ev["t"]),
                    "value": ev.get("value", 1),
                    "label": str(ev.get("label", "")),
                }
                for ev in track.events
            ]
            tracks.append(
                {
                    "name": track.name,
                    "type": "event_track",
                    "timescale": track.timescale,
                    "events": events,
                }
            )

        elif track.type == "ramp":
            keys = []
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                from_val = float(ev.get("from", 0.0))
                to_val = float(ev.get("to", 1.0))
                shape = str(ev.get("shape", "ease_in_out"))
                keys.append({"time": t_start, "value": from_val, "interp": "linear"})
                keys.append({"time": t_end, "value": to_val, "interp": shape})
            tracks.append(
                {
                    "name": track.name,
                    "type": "float_curve",
                    "timescale": track.timescale,
                    "keys": keys,
                }
            )

        elif track.type == "continuous" and track.values and track.hop_sec:
            hop = track.hop_sec
            target_hz = 25.0
            step = max(1, int(round(1.0 / (hop * target_hz))))
            keys = [
                {"time": i * hop, "value": float(track.values[i]), "interp": "linear"}
                for i in range(0, len(track.values), step)
            ]
            tracks.append(
                {
                    "name": track.name,
                    "type": "float_curve",
                    "timescale": track.timescale,
                    "keys": keys,
                }
            )

    payload = {
        "schema_version": "1.0",
        "generator": "MusiCue",
        "grammar": cuesheet.grammar,
        "duration_sec": cuesheet.duration_sec,
        "tempo_map": cuesheet.tempo_map,
        "tracks": tracks,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
