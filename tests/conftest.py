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

from musicue.schemas import CueSheet, CueTrack


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


@pytest.fixture()
def full_cuesheet() -> CueSheet:
    """CueSheet exercising all five track types - used by exporter tests."""
    kick_env = {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}
    return CueSheet(
        source_sha256="abc123",
        grammar="concert_visuals",
        duration_sec=10.0,
        tempo_map=[{"t": 0.0, "bpm": 120.0}],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.9, "envelope": kick_env, "tags": ["kick"]},
                    {"t": 1.0, "strength": 0.8, "envelope": kick_env, "tags": ["kick"]},
                    {"t": 2.0, "strength": 0.7, "envelope": kick_env, "tags": ["kick"]},
                ],
            ),
            CueTrack(
                name="vocal_phrase",
                type="envelope",
                timescale="meso",
                events=[
                    {
                        "t_start": 3.0,
                        "t_end": 5.5,
                        "strength": 0.85,
                        "envelope": {"a": 0.30, "d": 0.20, "s": 0.7, "r": 0.50},
                        "tags": ["vocal_entry"],
                    },
                ],
            ),
            CueTrack(
                name="section_change",
                type="step",
                timescale="macro",
                events=[
                    {"t": 0.0, "value": 1, "label": "intro"},
                    {"t": 5.0, "value": 2, "label": "chorus"},
                ],
            ),
            CueTrack(
                name="section_ramp",
                type="ramp",
                timescale="macro",
                events=[
                    {
                        "t_start": 4.0,
                        "t_end": 5.0,
                        "from": 0.0,
                        "to": 1.0,
                        "shape": "ease_in_out",
                        "label": "intro->chorus",
                    },
                ],
            ),
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.1,
                values=[-20.0 + i * 0.05 for i in range(100)],
            ),
        ],
    )
