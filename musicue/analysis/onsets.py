from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _robust_peak_strengths(peak_values: np.ndarray) -> np.ndarray:
    """Scale onset-envelope peak heights to [0, 1].

    Divides by the 99th percentile of the *peak* values rather than the
    global envelope max, so one freak transient doesn't squash every other
    onset toward zero.
    """
    if peak_values.size == 0:
        return peak_values
    ref = float(np.percentile(peak_values, 99))
    if ref <= 0:
        ref = float(peak_values.max())
    if ref <= 0:
        return np.zeros_like(peak_values)
    return np.clip(peak_values / ref, 0.0, 1.0)


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
    # Detect at the envelope *peaks* (backtrack=False) so strength is read
    # where the onset is actually salient. Backtracked frames sit on the
    # preceding local minimum, where the envelope is ~0 by construction.
    peaks = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        backtrack=False,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.07,
        wait=int(0.03 * sr / 512),
    )
    if len(peaks) == 0:
        return []
    # Timestamps still use the backtracked (attack-start) frame.
    starts = librosa.onset.onset_backtrack(peaks, onset_env)
    times = librosa.frames_to_time(starts, sr=sr)
    strengths = _robust_peak_strengths(onset_env[peaks])
    return [
        {
            "t": float(t),
            "strength": float(s),
            "timescale": "micro",
            "drum_class": None,
            "drum_class_conf": None,
            "labels": [],
        }
        for t, s in zip(times, strengths)
    ]
