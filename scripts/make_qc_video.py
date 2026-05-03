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
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf  # noqa: F401  -- keeps soundfile reachable for librosa.load


def render_frame_strip(
    audio_path: Path,
    analysis_json_path: Path,
    frames_dir: Path,
    fps: int = 24,
    width: int = 1920,
    height: int = 240,
) -> int:
    from musicue.schemas import AnalysisResult
    result = AnalysisResult.model_validate_json(analysis_json_path.read_text())

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr
    n_frames = int(duration * fps)
    frames_dir.mkdir(parents=True, exist_ok=True)

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
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        print("Rendering frames...")
        n = render_frame_strip(args.song, args.analysis, frames_dir, fps=args.fps)
        print(f"Rendered {n} frames. Encoding video...")
        encode_video(frames_dir, args.song, args.out, fps=args.fps)
    print(f"QC video saved to {args.out}")
