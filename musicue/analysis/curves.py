from __future__ import annotations
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path


def compute_lufs_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = sf.read(str(audio_path))
    if data.ndim == 1:
        data = data[:, np.newaxis]
    meter = pyln.Meter(rate)
    hop = max(int(hop_sec * rate), 100)
    values: list[float] = []
    for i in range(0, len(data), hop):
        chunk = data[i : i + hop]
        if len(chunk) < 100:
            values.append(values[-1] if values else -70.0)
            continue
        try:
            loudness = meter.integrated_loudness(chunk)
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
