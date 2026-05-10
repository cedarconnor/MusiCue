"""SMPTE timecode helpers for MusiCue.

Frames are integers; timecode is a string `HH:MM:SS:FF` (non-drop) or
`HH:MM:SS;FF` (drop-frame, semicolon before the final field per SMPTE
convention).

Drop-frame math reference: https://andrewduncan.net/timecodes/

The drop-frame algorithm is the canonical "10-frame round" approach used in
broadcast: at 29.97 fps, two frame numbers are dropped at the start of every
minute *except* every tenth minute. This is purely a labelling convention —
no actual frames are dropped — and exists because 29.97 ≈ 30000/1001 so a 30-
fps timecode drifts ~3.6 seconds per hour from real time. Dropping frame
numbers compensates.
"""
from __future__ import annotations


def t_to_frame(t: float, fps: float) -> int:
    """Convert seconds to a frame index (0-based)."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return int(round(t * fps))


def _format_non_drop(total_frames: int, fps_int: int) -> str:
    """HH:MM:SS:FF for integer / non-drop fps."""
    seconds, frames = divmod(total_frames, fps_int)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _format_drop_frame(total_frames: int, fps: float) -> str:
    """HH:MM:SS;FF using the SMPTE 10-frame-drop algorithm.

    Supports 29.97 (drop 2) and 59.94 (drop 4). 23.976 is technically
    non-drop in modern practice (no compensation needed for short content),
    but we treat it as non-drop and reject it here.
    """
    if abs(fps - 29.97) < 0.01:
        drop_frames = 2
        fps_int = 30
    elif abs(fps - 59.94) < 0.01:
        drop_frames = 4
        fps_int = 60
    else:
        raise ValueError(f"drop-frame only supported for 29.97 and 59.94, got {fps}")

    frames_per_10min = fps_int * 60 * 10 - drop_frames * 9
    frames_per_minute = fps_int * 60 - drop_frames
    d, m = divmod(total_frames, frames_per_10min)
    if m > drop_frames:
        adjusted = total_frames + drop_frames * 9 * d + drop_frames * (
            (m - drop_frames) // frames_per_minute
        )
    else:
        adjusted = total_frames + drop_frames * 9 * d

    seconds, frames = divmod(adjusted, fps_int)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"


def t_to_timecode(t: float, fps: float, drop_frame: bool = False) -> str:
    """SMPTE timecode string for the given seconds and FPS.

    Non-drop output uses colon-separated fields (`HH:MM:SS:FF`). Drop-frame
    uses a semicolon before the final field (`HH:MM:SS;FF`), matching the
    SMPTE convention used by Premiere, Resolve, Avid, FCPX.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    total_frames = t_to_frame(t, fps)
    if drop_frame:
        return _format_drop_frame(total_frames, fps)
    fps_int = int(round(fps))
    return _format_non_drop(total_frames, fps_int)
