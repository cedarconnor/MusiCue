"""Render compiled cuesheets as MP4 video previews.

Public API:
    render_cue_video(cuesheet, audio_path, out_path, ...)
"""
from __future__ import annotations

import bisect
from typing import Sequence

NTSC_FPS = 30000.0 / 1001.0


def plan_frames(duration_sec: float, fps: float) -> int:
    """Total frame count, rounded to nearest int."""
    if duration_sec <= 0.0 or fps <= 0.0:
        return 0
    return int(round(duration_sec * fps))


def frame_time(frame_idx: int, fps: float) -> float:
    """t_now for frame index `frame_idx` at `fps` frames/sec.

    Uses the exact 1001/30000 ratio when fps is the NTSC rate, otherwise
    the simple 1/fps division.
    """
    if abs(fps - NTSC_FPS) < 1e-6:
        return frame_idx * 1001.0 / 30000.0
    return frame_idx / fps


def events_in_window(
    events: Sequence[dict],
    t_now: float,
    window_sec: float,
    *,
    key: str = "t",
) -> list[dict]:
    """Slice point-in-time events to those inside [t_now - w/2, t_now + w/2].

    Assumes `events` is sorted by `key` in ascending order. Uses bisect
    for O(log n) lookup + O(k) copy where k is the slice size.
    """
    if not events:
        return []
    half = window_sec / 2.0
    lo = t_now - half
    hi = t_now + half
    ts = [float(e.get(key, 0.0)) for e in events]
    left = bisect.bisect_left(ts, lo)
    right = bisect.bisect_right(ts, hi)
    return list(events[left:right])


def spanning_events_in_window(
    events: Sequence[dict],
    t_now: float,
    window_sec: float,
    *,
    start_key: str = "t_start",
    end_key: str = "t_end",
) -> list[dict]:
    """Slice spanning events (envelopes, ramps) overlapping the window."""
    half = window_sec / 2.0
    lo = t_now - half
    hi = t_now + half
    out: list[dict] = []
    for e in events:
        s = float(e.get(start_key, 0.0))
        en = float(e.get(end_key, s))
        if en >= lo and s <= hi:
            out.append(e)
    return out
