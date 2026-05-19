"""Audio loading and probing that avoids librosa's deprecated audioread path.

librosa.load() falls back to audioread for compressed formats (m4a/aac/mp3)
that libsndfile can't decode. audioread is deprecated and slated for removal
in librosa 1.0, and emits FutureWarning floods in the meantime.

This module decodes via ffmpeg directly (PCM f32le on stdout) for those
formats and uses soundfile for everything libsndfile can read. Both ffmpeg
and ffprobe are already hard installs (used for video encoding and existing
duration probing) so no new install surface.

Public API:
    load_audio(path, *, sr=None, mono=False) -> (np.ndarray, int)
        - mono=True returns shape (n_samples,)
        - mono=False returns (n_samples, n_channels) for multichannel or
          (n_samples,) for natively-mono sources
        - sr=None preserves the native sample rate; otherwise resamples
          via soxr

    get_duration(path) -> float       # seconds
    get_samplerate(path) -> int       # native sample rate (no resample)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _ffprobe_or_die(audio_path: Path) -> str:
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            f"libsndfile cannot read {audio_path} and ffprobe is not on PATH. "
            "Install ffmpeg or convert the file to WAV/FLAC."
        )
    return "ffprobe"


def _ffprobe_field(audio_path: Path, entry: str) -> str:
    ffprobe = _ffprobe_or_die(audio_path)
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", entry,
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe failed for {audio_path} ({entry}): {result.stderr.strip()}"
        )
    return result.stdout.strip().splitlines()[0]


def get_duration(audio_path: Path) -> float:
    try:
        return float(sf.info(str(audio_path)).duration)
    except sf.LibsndfileError:
        pass
    return float(_ffprobe_field(audio_path, "format=duration"))


def get_samplerate(audio_path: Path) -> int:
    try:
        return int(sf.info(str(audio_path)).samplerate)
    except sf.LibsndfileError:
        pass
    return int(_ffprobe_field(audio_path, "stream=sample_rate"))


def _ffmpeg_or_die(audio_path: Path) -> str:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"libsndfile cannot read {audio_path} and ffmpeg is not on PATH. "
            "Install ffmpeg or convert the file to WAV/FLAC."
        )
    return "ffmpeg"


def _ffmpeg_decode(audio_path: Path) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-supported audio file to a (samples_or_2d, sr) tuple.

    Returns the natively-encoded sample rate; resampling is the caller's
    responsibility (so load_audio's sr parameter stays the single source
    of truth for resampling).
    """
    ffmpeg = _ffmpeg_or_die(audio_path)
    native_sr = get_samplerate(audio_path)
    channels = int(_ffprobe_field(audio_path, "stream=channels"))
    result = subprocess.run(
        [
            ffmpeg, "-nostdin", "-loglevel", "error",
            "-i", str(audio_path),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(native_sr), "-ac", str(channels),
            "-",
        ],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {audio_path}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    raw = np.frombuffer(result.stdout, dtype=np.float32)
    if channels > 1:
        raw = raw.reshape(-1, channels)
    return raw, native_sr


def _read_native(audio_path: Path) -> tuple[np.ndarray, int]:
    """soundfile first, ffmpeg fallback. Returns (samples,) or (samples, channels)."""
    try:
        data, native_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        return data, int(native_sr)
    except sf.LibsndfileError:
        return _ffmpeg_decode(audio_path)


def load_audio(
    audio_path: Path | str,
    *,
    sr: int | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    audio_path = Path(audio_path)
    data, native_sr = _read_native(audio_path)
    data = np.ascontiguousarray(data, dtype=np.float32)

    if mono and data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    out_sr = native_sr
    if sr is not None and sr != native_sr:
        import soxr

        # soxr.resample handles both 1-D and 2-D (axis 0 = samples).
        data = soxr.resample(data, native_sr, sr).astype(np.float32)
        out_sr = sr

    return data, out_sr
