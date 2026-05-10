"""Stamp frame numbers and SMPTE timecode onto analysis and cuesheet events.

Both analysis events (typed Pydantic models) and cuesheet events (dicts) carry
the same `frame` / `timecode` fields. Two functions here cover both shapes.

These helpers are pure: they return new objects rather than mutating in place,
so a caller can re-stamp at a different FPS without leaking state.
"""
from __future__ import annotations

from musicue.schemas import AnalysisResult, CueSheet, CueTrack
from musicue.timecode import t_to_frame, t_to_timecode


def _stamp_event_dict(ev: dict, fps: float, drop_frame: bool) -> dict:
    """Add `frame` and `timecode` to a cuesheet event dict.

    Cuesheet events use either `t` (impulse/envelope) or `t_start` / `t_end`
    (ramp). We stamp whichever is present.
    """
    out = dict(ev)
    if "t" in out:
        t = float(out["t"])
        out["frame"] = t_to_frame(t, fps)
        out["timecode"] = t_to_timecode(t, fps, drop_frame)
    if "t_start" in out:
        ts = float(out["t_start"])
        out["frame_start"] = t_to_frame(ts, fps)
        out["timecode_start"] = t_to_timecode(ts, fps, drop_frame)
    if "t_end" in out:
        te = float(out["t_end"])
        out["frame_end"] = t_to_frame(te, fps)
        out["timecode_end"] = t_to_timecode(te, fps, drop_frame)
    return out


def populate_analysis_frames(
    analysis: AnalysisResult,
    fps: float,
    drop_frame: bool = False,
) -> AnalysisResult:
    """Return a copy of `analysis` with frame/timecode stamped on every event."""
    a = analysis.model_copy(deep=True)
    a.analysis_config.fps = fps
    a.analysis_config.drop_frame = drop_frame

    for b in a.beats:
        b.frame = t_to_frame(b.t, fps)
        b.timecode = t_to_timecode(b.t, fps, drop_frame)

    for s in a.sections:
        s.frame_start = t_to_frame(s.start, fps)
        s.frame_end = t_to_frame(s.end, fps)
        s.timecode_start = t_to_timecode(s.start, fps, drop_frame)
        s.timecode_end = t_to_timecode(s.end, fps, drop_frame)

    for tr in a.section_transitions:
        tr.frame = t_to_frame(tr.t, fps)
        tr.timecode = t_to_timecode(tr.t, fps, drop_frame)

    for stem_onsets in a.onsets.values():
        for o in stem_onsets:
            o.frame = t_to_frame(o.t, fps)
            o.timecode = t_to_timecode(o.t, fps, drop_frame)

    for stem_notes in a.midi.values():
        for n in stem_notes:
            n.frame = t_to_frame(n.t, fps)
            n.timecode = t_to_timecode(n.t, fps, drop_frame)

    for stem_phrases in a.phrases.values():
        for p in stem_phrases:
            p.frame_start = t_to_frame(p.t_start, fps)
            p.frame_end = t_to_frame(p.t_end, fps)
            p.timecode_start = t_to_timecode(p.t_start, fps, drop_frame)
            p.timecode_end = t_to_timecode(p.t_end, fps, drop_frame)

    return a


def populate_cuesheet_frames(
    cs: CueSheet,
    fps: float,
    drop_frame: bool = False,
) -> CueSheet:
    """Return a copy of `cs` with frame/timecode stamped on every track event."""
    out = cs.model_copy(deep=True)
    out.fps = fps
    out.drop_frame = drop_frame
    new_tracks: list[CueTrack] = []
    for tr in out.tracks:
        new_events = [_stamp_event_dict(ev, fps, drop_frame) for ev in tr.events]
        new_tracks.append(tr.model_copy(update={"events": new_events}))
    out.tracks = new_tracks
    return out
