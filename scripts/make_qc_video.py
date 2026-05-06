"""
MusiCue QC video: waveform + onset/section overlay rendered with ffmpeg.

Usage:
  python scripts/make_qc_video.py --song song.wav --analysis runs/abc/analysis.json --out qc.mp4

Requires:
  ffmpeg on PATH (winget install Gyan.FFmpeg)
  matplotlib (pip install matplotlib)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # noqa: F401  -- keeps soundfile reachable for librosa.load

# Module-level state populated lazily in workers so we don't reload audio per frame.
_WORKER_AUDIO: tuple[np.ndarray, int, float] | None = None
_WORKER_ONSETS: dict[str, list[tuple[float, float]]] | None = None
_WORKER_SECTIONS: list[tuple[float, float, str]] | None = None


def _worker_init(audio_path: str, analysis_path: str) -> None:
    """Pre-load audio and analysis once per worker process."""
    global _WORKER_AUDIO, _WORKER_ONSETS, _WORKER_SECTIONS
    from musicue.schemas import AnalysisResult

    result = AnalysisResult.model_validate_json(Path(analysis_path).read_text())
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    _WORKER_AUDIO = (y, sr, len(y) / sr)
    _WORKER_ONSETS = {
        stem: [(o.t, getattr(o, "strength", 0.5)) for o in onsets]
        for stem, onsets in result.onsets.items()
    }
    _WORKER_SECTIONS = [(s.start, s.end, s.label) for s in result.sections]


def _render_frame(args: tuple[int, int, int, int, str]) -> int:
    """Worker: render one frame to disk."""
    frame_idx, fps, width, height, frames_dir = args
    assert _WORKER_AUDIO is not None
    y, sr, duration = _WORKER_AUDIO
    onsets = _WORKER_ONSETS or {}
    sections = _WORKER_SECTIONS or []

    stem_colors = {"drums": "#FF5722", "bass": "#2196F3",
                   "vocals": "#4CAF50", "other": "#9C27B0"}
    t_center = frame_idx / fps
    window_sec = 4.0
    t_start = max(0.0, t_center - window_sec / 2)
    t_end = min(duration, t_start + window_sec)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    s_start = int(t_start * sr)
    s_end = int(t_end * sr)
    chunk = y[s_start:s_end]
    chunk_t = np.linspace(t_start, t_end, len(chunk))
    ax.plot(chunk_t, chunk, color="#888888", lw=0.5, alpha=0.6)

    for stem_name, color in stem_colors.items():
        for t, _strength in onsets.get(stem_name, []):
            if t_start <= t <= t_end:
                ax.axvline(t, color=color, lw=1.5, alpha=0.8)

    for start, end, label in sections:
        if end >= t_start and start <= t_end:
            x0 = max(start, t_start)
            x1 = min(end, t_end)
            ax.axvspan(x0, x1, alpha=0.05, color="#ffffff")
            ax.text(x0 + 0.05, 0.85, label,
                    transform=ax.get_xaxis_transform(),
                    color="#ffffff", fontsize=7, alpha=0.7)

    ax.axvline(t_center, color="#FFEB3B", lw=2.0, alpha=0.9)
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    frame_path = Path(frames_dir) / f"frame_{frame_idx:06d}.png"
    plt.savefig(str(frame_path), dpi=100, facecolor=fig.get_facecolor())
    plt.close()
    return frame_idx


def render_frame_strip(
    audio_path: Path,
    analysis_json_path: Path,
    frames_dir: Path,
    fps: int = 24,
    width: int = 1920,
    height: int = 240,
    workers: int = 1,
) -> int:
    from musicue.schemas import AnalysisResult
    result = AnalysisResult.model_validate_json(analysis_json_path.read_text())
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr
    n_frames = int(duration * fps)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        # Single-process path: keep original simple code, no IPC overhead.
        stem_colors = {"drums": "#FF5722", "bass": "#2196F3",
                       "vocals": "#4CAF50", "other": "#9C27B0"}
        for frame_idx in range(n_frames):
            t_center = frame_idx / fps
            window_sec = 4.0
            t_start = max(0.0, t_center - window_sec / 2)
            t_end = min(duration, t_start + window_sec)

            fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
            fig.patch.set_facecolor("#1a1a1a")
            ax.set_facecolor("#1a1a1a")
            s_start = int(t_start * sr)
            s_end = int(t_end * sr)
            chunk = y[s_start:s_end]
            chunk_t = np.linspace(t_start, t_end, len(chunk))
            ax.plot(chunk_t, chunk, color="#888888", lw=0.5, alpha=0.6)
            for stem_name, color in stem_colors.items():
                for onset in result.onsets.get(stem_name, []):
                    if t_start <= onset.t <= t_end:
                        ax.axvline(onset.t, color=color, lw=1.5, alpha=0.8)
            for section in result.sections:
                if section.end >= t_start and section.start <= t_end:
                    x0 = max(section.start, t_start)
                    x1 = min(section.end, t_end)
                    ax.axvspan(x0, x1, alpha=0.05, color="#ffffff")
                    ax.text(x0 + 0.05, 0.85, section.label,
                            transform=ax.get_xaxis_transform(),
                            color="#ffffff", fontsize=7, alpha=0.7)
            ax.axvline(t_center, color="#FFEB3B", lw=2.0, alpha=0.9)
            ax.set_xlim(t_start, t_end)
            ax.set_ylim(-1.1, 1.1)
            ax.axis("off")
            plt.tight_layout(pad=0)
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            plt.savefig(str(frame_path), dpi=100, facecolor=fig.get_facecolor())
            plt.close()
        return n_frames

    # Parallel path: spawn worker pool, each worker pre-loads audio once.
    audio_str = str(audio_path)
    analysis_str = str(analysis_json_path)
    frames_str = str(frames_dir)
    work = [(i, fps, width, height, frames_str) for i in range(n_frames)]
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(audio_str, analysis_str),
    ) as pool:
        for _ in pool.map(_render_frame, work, chunksize=8):
            done += 1
            if done % 200 == 0 or done == n_frames:
                print(f"  rendered {done}/{n_frames} frames", flush=True)
    return n_frames


def encode_video(frames_dir: Path, audio_path: Path, out_path: Path, fps: int = 24) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("qc_video.mp4"))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of worker processes for frame rendering",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        print(f"Rendering frames with {args.workers} workers...")
        n = render_frame_strip(
            args.song, args.analysis, frames_dir,
            fps=args.fps, workers=args.workers,
        )
        print(f"Rendered {n} frames. Encoding video...")
        encode_video(frames_dir, args.song, args.out, fps=args.fps)
    print(f"QC video saved to {args.out}")
