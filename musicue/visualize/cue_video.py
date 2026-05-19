"""Render compiled cuesheets as MP4 video previews.

Public API:
    render_cue_video(cuesheet, audio_path, out_path, ...)
"""
from __future__ import annotations

import bisect
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from musicue.schemas import CueSheet  # noqa: E402

log = logging.getLogger(__name__)

NTSC_FPS = 30000.0 / 1001.0

HEADER_PX = 60
LANE_PX = 56
FIRE_PANEL_FRAC = 200 / 1280  # 200 px out of 1280 default width


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


def _lane_drawers():
    """Lazy import to avoid lanes <-> cue_video cycle at import time."""
    from musicue.visualize.lanes import (
        draw_continuous_lane,
        draw_envelope_lane,
        draw_header,
        draw_impulse_lane,
        draw_ramp_lane,
        draw_step_lane,
        fire_brightness,
    )

    return {
        "draw_header": draw_header,
        "fire_brightness": fire_brightness,
        "impulse": draw_impulse_lane,
        "envelope": draw_envelope_lane,
        "step": draw_step_lane,
        "ramp": draw_ramp_lane,
        "continuous": draw_continuous_lane,
    }


def compose_frame(
    cuesheet: CueSheet,
    t_now: float,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    window_sec: float = 5.0,
) -> None:
    """Render one PNG frame at t_now."""
    from musicue.visualize.colors import type_tint

    drawers = _lane_drawers()
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi), dpi=dpi,
        facecolor=(0.04, 0.04, 0.04),
    )
    n_lanes = len(cuesheet.tracks)
    if n_lanes == 0:
        fig.text(
            0.5, 0.5, "(empty cuesheet)",
            ha="center", va="center", color="white", fontsize=18,
        )
        fig.savefig(str(out_path), facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    header_h = HEADER_PX / height
    lane_h = (1.0 - header_h) / n_lanes

    ax_header = fig.add_axes((0.0, 1.0 - header_h, 1.0, header_h))
    drawers["draw_header"](ax_header, cuesheet, t_now)

    for i, track in enumerate(cuesheet.tracks):
        y_top = 1.0 - header_h - i * lane_h
        y_bottom = y_top - lane_h

        # Fire panel (left): lane name + flashing fire cell.
        ax_fire = fig.add_axes((0.0, y_bottom, FIRE_PANEL_FRAC, lane_h))
        ax_fire.set_xticks([])
        ax_fire.set_yticks([])
        for spine in ax_fire.spines.values():
            spine.set_visible(False)
        ax_fire.set_facecolor(type_tint(track.type))
        ax_fire.set_xlim(0, 1)
        ax_fire.set_ylim(0, 1)
        ax_fire.text(
            0.04, 0.5, track.name,
            color="white", fontsize=10, va="center", ha="left",
        )
        brightness = drawers["fire_brightness"](track, t_now)
        ax_fire.add_patch(Rectangle(
            (0.55, 0.15), 0.40, 0.70,
            facecolor=(1.0, 1.0, 1.0),
            alpha=max(0.1, brightness),
            edgecolor=(1.0, 1.0, 1.0, 0.4),
            linewidth=1.0,
        ))

        # Timeline strip (right): scrolling window.
        ax = fig.add_axes((FIRE_PANEL_FRAC, y_bottom, 1.0 - FIRE_PANEL_FRAC, lane_h))
        ax.set_facecolor(type_tint(track.type))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(t_now - window_sec / 2.0, t_now + window_sec / 2.0)
        ax.set_ylim(0.0, 1.0)

        draw = drawers.get(track.type)
        if draw is not None:
            try:
                draw(ax, track, t_now=t_now, window_sec=window_sec)
            except Exception:
                log.exception("lane renderer failed for %s", track.name)

        ax.axvline(
            t_now, color=(1.0, 0.92, 0.23),
            linewidth=1.5, alpha=0.85, zorder=5,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)


# ---- Worker process state ----
_WORKER_CUESHEET: CueSheet | None = None
_WORKER_OPTS: dict | None = None


def _worker_init(cuesheet_json: str, opts: dict) -> None:
    global _WORKER_CUESHEET, _WORKER_OPTS
    _WORKER_CUESHEET = CueSheet.model_validate_json(cuesheet_json)
    _WORKER_OPTS = opts


def _worker_render(args: tuple[int, str]) -> int:
    frame_idx, out_str = args
    assert _WORKER_CUESHEET is not None and _WORKER_OPTS is not None
    t_now = frame_time(frame_idx, _WORKER_OPTS["fps"])
    compose_frame(
        _WORKER_CUESHEET, t_now, Path(out_str),
        width=_WORKER_OPTS["width"],
        height=_WORKER_OPTS["height"],
        window_sec=_WORKER_OPTS["window_sec"],
    )
    return frame_idx


def _probe_audio_duration(audio_path: Path) -> float:
    """Probe audio duration in seconds.

    soundfile (libsndfile) handles WAV/FLAC/OGG cleanly but doesn't support
    AAC-in-m4a or some MP3 variants. Fall back to ffprobe for anything
    soundfile rejects — ffprobe is already a hard dependency of the render
    step (we shell out to ffmpeg for encoding), so this never widens the
    install surface.
    """
    try:
        return float(sf.info(str(audio_path)).duration)
    except Exception:
        pass
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            f"Cannot probe audio duration for {audio_path}: "
            "libsndfile cannot read this format and ffprobe is not on PATH."
        )
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe failed for {audio_path}: {result.stderr.strip()}"
        )
    return float(result.stdout.strip())


def _encode_video(
    frames_dir: Path,
    audio_path: Path,
    out_path: Path,
    fps: float,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install via scripts/install_ffmpeg.ps1 "
            "or place ffmpeg in vendor/ffmpeg/."
        )
    fps_arg = "30000/1001" if abs(fps - NTSC_FPS) < 1e-6 else f"{fps:.6f}"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", fps_arg,
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-r", fps_arg,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def render_cue_video(
    cuesheet: CueSheet,
    audio_path: Path,
    out_path: Path,
    *,
    fps: float = NTSC_FPS,
    width: int = 1280,
    height: int = 720,
    window_sec: float = 5.0,
    workers: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Render a cuesheet to an MP4 video, syncing to audio_path."""
    audio_path = Path(audio_path)
    out_path = Path(out_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    audio_duration = _probe_audio_duration(audio_path)
    if abs(audio_duration - cuesheet.duration_sec) > 0.5:
        log.warning(
            "audio duration %.2fs disagrees with cuesheet duration %.2fs; "
            "using audio duration",
            audio_duration, cuesheet.duration_sec,
        )

    n_frames = plan_frames(audio_duration, fps)
    if n_frames <= 0:
        raise ValueError(f"zero frames planned (duration={audio_duration}, fps={fps})")

    if workers is None:
        workers = max(1, (cpu_count() or 4) // 2)

    cuesheet_json = cuesheet.model_dump_json()
    opts = {
        "fps": fps, "width": width, "height": height,
        "window_sec": window_sec,
    }

    with tempfile.TemporaryDirectory(prefix="musicue-cue-video-") as td:
        frames_dir = Path(td)
        work = [(i, str(frames_dir / f"frame_{i:06d}.png")) for i in range(n_frames)]

        if workers <= 1:
            _worker_init(cuesheet_json, opts)
            for arg in work:
                _worker_render(arg)
                if progress and (arg[0] % 10 == 0 or arg[0] == n_frames - 1):
                    progress(arg[0] / n_frames)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(cuesheet_json, opts),
            ) as pool:
                done = 0
                for _ in pool.map(_worker_render, work, chunksize=8):
                    done += 1
                    if progress and (done % 10 == 0 or done == n_frames):
                        progress(done / n_frames)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _encode_video(frames_dir, audio_path, out_path, fps)

    if progress:
        progress(1.0)
