"""MIDI exporter for MusiCue cuesheets.

Maps each CueTrack type onto Standard MIDI File constructs:

- ``impulse``    -> note_on/note_off pairs. Channel 10 (GM drums) for known drum
                    track names (kick/snare/hat/...); melodic channel otherwise.
                    ``strength`` -> velocity, ``envelope.d`` -> note duration.
- ``envelope``   -> sustained note_on/note_off spanning ``t_start``..``t_end``.
                    ``strength`` -> velocity.
- ``step``       -> Meta marker events with the event ``label`` text.
- ``ramp``       -> currently emitted as nothing (ramps are visual-domain shapes;
                    DAWs receive surrounding step/continuous data).
- ``continuous`` -> CC74 (filter cutoff, repurposed as energy follower) at ~10 Hz.

Tempo is taken from ``cuesheet.tempo_map[0].bpm`` (default 120 BPM) and written
once as a ``set_tempo`` meta event. A type-1 MIDI file is produced with one
meta track plus one track per CueTrack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mido

from musicue.schemas import CueSheet

# MIDI channel assignments by track name
_CHANNEL_MAP: dict[str, int] = {
    "kick": 9,       # channel 10 (0-indexed: 9) = GM drums
    "snare": 9,
    "hat": 9,
    "hihat": 9,
    "downbeat": 9,
    "downbeat_pulse": 9,
}
# GM drum note numbers
_NOTE_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hat": 42,
    "hihat": 42,
    "downbeat": 75,
    "downbeat_pulse": 75,
    "vocal_phrase": 64,
    "drop": 37,
    "impact": 39,
}
_DEFAULT_NOTE = 60
_ENERGY_CC = 74  # CC74: filter cutoff -- repurposed as energy follower


def _ticks(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    beats = seconds * 1_000_000 / tempo_us
    return max(0, int(round(beats * ticks_per_beat)))


def _rescale_to_unit(values: list[float]) -> list[float]:
    """Rescale values to [0, 1] based on observed min/max.

    Continuous tracks may arrive already-normalized (e.g. percentile-normalized
    in the grammar) or as raw signal (e.g. LUFS in [-70, 0]). Auto-rescaling per
    track handles both without an explicit range hint.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def export(cuesheet: CueSheet, out_path: Path, ticks_per_beat: int = 480, **opts) -> None:
    bpm = 120.0
    if cuesheet.tempo_map:
        bpm = float(cuesheet.tempo_map[0].get("bpm", 120.0))
    tempo_us = int(60_000_000 / bpm)

    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    for track in cuesheet.tracks:
        midi_track = mido.MidiTrack()
        mid.tracks.append(midi_track)
        midi_track.append(mido.MetaMessage("track_name", name=track.name, time=0))

        channel = _CHANNEL_MAP.get(track.name, 0)
        note = _NOTE_MAP.get(track.name, _DEFAULT_NOTE)
        msgs: list[tuple[int, Any]] = []

        if track.type == "impulse":
            for event in track.events:
                t = float(event["t"])
                velocity = max(1, min(127, int(float(event.get("strength", 0.8)) * 127)))
                tick = _ticks(t, ticks_per_beat, tempo_us)
                env = event.get("envelope", {})
                dur = float(env.get("d", 0.1))
                off_tick = _ticks(t + dur, ticks_per_beat, tempo_us)
                msgs.append((tick, mido.Message("note_on", channel=channel, note=note,
                                                velocity=velocity, time=0)))
                msgs.append((off_tick, mido.Message("note_off", channel=channel, note=note,
                                                    velocity=0, time=0)))

        elif track.type == "envelope":
            for event in track.events:
                t_start = float(event.get("t_start", 0.0))
                t_end = float(event.get("t_end", t_start + 1.0))
                velocity = max(1, min(127, int(float(event.get("strength", 0.8)) * 127)))
                tick_on = _ticks(t_start, ticks_per_beat, tempo_us)
                tick_off = _ticks(t_end, ticks_per_beat, tempo_us)
                msgs.append((tick_on, mido.Message("note_on", channel=channel, note=note,
                                                   velocity=velocity, time=0)))
                msgs.append((tick_off, mido.Message("note_off", channel=channel, note=note,
                                                    velocity=0, time=0)))

        elif track.type == "step":
            for event in track.events:
                t = float(event["t"])
                tick = _ticks(t, ticks_per_beat, tempo_us)
                label = str(event.get("label", ""))
                msgs.append((tick, mido.MetaMessage("marker", text=label, time=0)))

        elif track.type == "continuous" and track.values and track.hop_sec:
            hop = track.hop_sec
            target_hz = 10.0
            # Stride to keep target_hz samples/sec from a 1/hop samples/sec source.
            # When source rate <= target rate (hop * target_hz >= 1), emit every frame.
            step = max(1, int(round(1.0 / (target_hz * hop))))
            unit_values = _rescale_to_unit(track.values)
            for i in range(0, len(track.values), step):
                t = i * hop
                cc_val = max(0, min(127, int(unit_values[i] * 127)))
                tick = _ticks(t, ticks_per_beat, tempo_us)
                msgs.append((tick, mido.Message("control_change", channel=channel,
                                                control=_ENERGY_CC, value=cc_val, time=0)))

        msgs.sort(key=lambda x: x[0])
        prev_tick = 0
        for abs_tick, msg in msgs:
            delta = abs_tick - prev_tick
            msg.time = delta
            midi_track.append(msg)
            prev_tick = abs_tick

        midi_track.append(mido.MetaMessage("end_of_track", time=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
