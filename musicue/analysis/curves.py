from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

_BS1770_WINDOW = 0.4  # pyloudnorm integrated_loudness requires ≥400ms


def _read_audio_2d(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 with shape (samples, channels). Tries soundfile
    first (fast WAV/FLAC path) and falls back to librosa.load for compressed
    formats like m4a/mp3/aac that libsndfile cannot decode."""
    try:
        data, rate = sf.read(str(audio_path), dtype="float32")
    except sf.LibsndfileError:
        # librosa.load returns (channels, samples) when mono=False; transpose
        # to match the soundfile (samples, channels) layout.
        y, rate = librosa.load(str(audio_path), sr=None, mono=False)
        data = y.T if y.ndim > 1 else y
        data = np.ascontiguousarray(data, dtype=np.float32)
    return data, rate


def compute_lufs_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = _read_audio_2d(audio_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    meter = pyln.Meter(rate)
    hop = max(int(hop_sec * rate), 1)
    window_samples = int(_BS1770_WINDOW * rate)
    n = len(data)
    values: list[float] = []
    for i in range(0, n, hop):
        end = min(i + window_samples, n)
        start = max(0, end - window_samples)
        chunk = data[start:end]
        try:
            loudness = meter.integrated_loudness(chunk)
            if math.isinf(loudness) or math.isnan(loudness):
                values.append(-70.0)
            else:
                values.append(float(np.clip(loudness, -70.0, 0.0)))
        except Exception:
            values.append(-70.0)
    return {"hop_sec": hop / rate, "values": values}


def compute_rms_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    try:
        data, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        # librosa returns (channels, samples) for stereo; transpose to the
        # (samples, channels) layout sf.read uses so the mean(axis=1) below
        # works the same way for both code paths.
        y, rate = librosa.load(str(audio_path), sr=None, mono=False)
        data = (y.T if y.ndim > 1 else y).astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = max(1, int(hop_sec * rate))
    rms = librosa.feature.rms(y=data, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in rms]}


def compute_spectral_centroid_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, int(hop_sec * rate))
    centroid = librosa.feature.spectral_centroid(y=y, sr=rate, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in centroid]}


def compute_spectral_flux_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, int(hop_sec * rate))
    flux = librosa.onset.onset_strength(y=y, sr=rate, hop_length=hop)
    return {"hop_sec": hop / rate, "values": [float(v) for v in flux]}


def compute_stereo_width_pan(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = sf.read(str(audio_path))
    hop = max(1, int(hop_sec * rate))
    if data.ndim == 1:
        n = max(1, len(data) // hop)
        zeros = [0.0] * n
        return {
            "width": {"hop_sec": hop / rate, "values": zeros},
            "pan": {"hop_sec": hop / rate, "values": zeros},
        }
    L, R = data[:, 0], data[:, 1]
    width_vals, pan_vals = [], []
    for i in range(0, len(data) - hop, hop):
        l_chunk = L[i : i + hop]
        r_chunk = R[i : i + hop]
        mid = l_chunk + r_chunk
        side = l_chunk - r_chunk
        mid_rms = float(np.sqrt(np.mean(mid ** 2)) + 1e-9)
        side_rms = float(np.sqrt(np.mean(side ** 2)) + 1e-9)
        width_vals.append(float(np.clip(side_rms / mid_rms, 0.0, 1.0)))
        l_rms = float(np.sqrt(np.mean(l_chunk ** 2)) + 1e-9)
        r_rms = float(np.sqrt(np.mean(r_chunk ** 2)) + 1e-9)
        pan_vals.append(float(np.clip((r_rms - l_rms) / (r_rms + l_rms), -1.0, 1.0)))
    actual_hop = hop / rate
    return {
        "width": {"hop_sec": actual_hop, "values": width_vals},
        "pan": {"hop_sec": actual_hop, "values": pan_vals},
    }
