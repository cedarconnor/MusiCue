"""Per-stem peaks JSON generation for waveform rendering in the Web UI.

Produces a JSON file with a small metadata wrapper around interleaved
(min, max) float pairs per pixel column. The Web UI converts this to the
shape WaveSurfer.js v7 expects on load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

PEAKS_VERSION = 2


def compute_peaks(audio_path: Path, samples_per_pixel: int = 1024) -> dict:
    """Read audio (any libsndfile-supported format), downmix to mono, return peaks dict."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None, mono=False)
        data = (y.T if y.ndim > 1 else y).astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    n = len(data)
    n_pixels = int(np.ceil(n / samples_per_pixel))
    pad_len = n_pixels * samples_per_pixel - n
    if pad_len:
        data = np.concatenate([data, np.zeros(pad_len, dtype=np.float32)])

    framed = data.reshape(n_pixels, samples_per_pixel)
    mins = framed.min(axis=1)
    maxs = framed.max(axis=1)
    interleaved = np.empty(n_pixels * 2, dtype=np.float32)
    interleaved[0::2] = mins
    interleaved[1::2] = maxs

    return {
        "version": PEAKS_VERSION,
        "channels": 1,
        "sample_rate": int(sr),
        "samples_per_pixel": samples_per_pixel,
        "length": n_pixels,
        "data": [round(float(v), 4) for v in interleaved],
    }


def write_peaks(audio_path: Path, out_path: Path, samples_per_pixel: int = 1024) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    peaks = compute_peaks(audio_path, samples_per_pixel)
    out_path.write_text(json.dumps(peaks))
    return out_path
