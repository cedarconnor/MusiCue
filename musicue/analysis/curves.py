from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln

from musicue.analysis import audio_io

_BS1770_WINDOW = 0.4  # pyloudnorm integrated_loudness requires ≥400ms


def _read_audio_2d(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 with shape (samples, channels) or (samples,) for mono."""
    return audio_io.load_audio(audio_path)


def compute_integrated_lufs(audio_path: Path) -> float | None:
    """Single integrated LUFS for the whole file (BS.1770 with gating).

    Returns None on failure (e.g. the meter rejects too-short signals).
    """
    try:
        data, rate = _read_audio_2d(audio_path)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        if math.isinf(loudness) or math.isnan(loudness):
            return None
        return float(loudness)
    except Exception:
        return None


def compute_lufs_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = _read_audio_2d(audio_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    meter = pyln.Meter(rate)
    hop = max(int(hop_sec * rate), 1)
    window_samples = int(_BS1770_WINDOW * rate)
    n = len(data)
    values: list[float] = []
    half = window_samples // 2
    for i in range(0, n, hop):
        # Centered window [i - 0.2s, i + 0.2s] so the value stamped at i
        # describes the audio around i, not the 400 ms after it. At the
        # edges the window is shifted (not shrunk) to stay inside the file,
        # since BS.1770 needs a full 400 ms block.
        start = max(0, min(i - half, n - window_samples))
        end = min(n, start + window_samples)
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


def compute_rms_curve(
    audio_path: Path, hop_sec: float = 0.04, frame_sec: float | None = None
) -> dict:
    """Centered RMS (linear amplitude) on a ``hop_sec`` grid.

    ``frame_sec=None`` keeps librosa's default 2048-sample window (~46 ms at
    44.1 kHz); pass e.g. ``0.1`` for a fixed-duration window regardless of
    sample rate. Frames are centered (librosa ``center=True``), so frame
    ``i`` describes the audio around ``i * hop``.
    """
    data, rate = audio_io.load_audio(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = max(1, int(hop_sec * rate))
    if frame_sec is None:
        rms = librosa.feature.rms(y=data, hop_length=hop)[0]
    else:
        frame = max(hop, int(round(frame_sec * rate)))
        rms = librosa.feature.rms(y=data, frame_length=frame, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in rms]}


def compute_spectral_centroid_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = audio_io.load_audio(audio_path, mono=True)
    hop = max(1, int(hop_sec * rate))
    centroid = librosa.feature.spectral_centroid(y=y, sr=rate, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in centroid]}


def compute_spectral_flux_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = audio_io.load_audio(audio_path, mono=True)
    hop = max(1, int(hop_sec * rate))
    flux = librosa.onset.onset_strength(y=y, sr=rate, hop_length=hop)
    return {"hop_sec": hop / rate, "values": [float(v) for v in flux]}


def compute_stereo_width_pan(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = _read_audio_2d(audio_path)
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
