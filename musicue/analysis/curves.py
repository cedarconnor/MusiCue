from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

_BS1770_WINDOW = 0.4  # pyloudnorm integrated_loudness requires ≥400ms


def compute_lufs_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = sf.read(str(audio_path), dtype="float32")
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
    data, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    hop = max(1, int(hop_sec * rate))
    rms = librosa.feature.rms(y=data, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in rms]}
