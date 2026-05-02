import os
import tempfile

# Set NUMBA_CACHE_DIR to a writable temp directory before importing librosa.
# librosa uses @jit(cache=True) functions that try to write compiled cache to
# the librosa install directory, which may be read-only (e.g. C:\Program Files).
# Without a writable cache dir numba spins trying to acquire a file lock.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory) -> Path:
    """10-second 44100 Hz mono WAV: 440 Hz sine + 4 transient bursts at 0.5, 2.5, 5.0, 7.5s."""
    sr = 44100
    duration = 10.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(42)
    for burst_t in (0.5, 2.5, 5.0, 7.5):
        onset = int(burst_t * sr)
        length = min(2205, n - onset)
        decay = np.exp(-np.arange(length) / 220.0)
        signal[onset : onset + length] += 0.6 * decay * rng.standard_normal(length)
    signal = np.clip(signal, -1.0, 1.0).astype(np.float32)
    p = tmp_path_factory.mktemp("fixtures") / "test_song.wav"
    sf.write(str(p), signal, sr)
    return p
