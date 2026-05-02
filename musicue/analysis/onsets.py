from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def detect_onsets(audio_path: Path, sr: int = 22050) -> list[dict]:
    data, native_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if native_sr != sr:
        import soxr
        data = soxr.resample(data, native_sr, sr, quality="HQ")
    y = data
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.max() == 0:
        return []
    frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.07,
        wait=int(0.03 * sr / 512),
    )
    times = librosa.frames_to_time(frames, sr=sr)
    peak = float(onset_env.max())
    return [
        {
            "t": float(t),
            "strength": float(np.clip(onset_env[f] / peak, 0.0, 1.0)),
            "timescale": "micro",
            "drum_class": None,
            "drum_class_conf": None,
            "labels": [],
        }
        for t, f in zip(times, frames)
    ]
