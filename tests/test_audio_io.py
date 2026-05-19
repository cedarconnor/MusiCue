"""Tests for musicue.analysis.audio_io.

The soundfile path is exercised against real WAV fixtures. The ffmpeg-decode
path is verified separately (it would require an m4a fixture) by faking the
LibsndfileError and verifying the dispatch to _ffmpeg_decode.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from musicue.analysis import audio_io


def _write_wav(path: Path, data: np.ndarray, sr: int) -> Path:
    sf.write(str(path), data, sr)
    return path


def test_load_audio_mono_wav_returns_1d(tmp_path):
    sr = 22050
    samples = np.linspace(-0.5, 0.5, sr * 2, dtype=np.float32)  # 2s mono
    wav = _write_wav(tmp_path / "mono.wav", samples, sr)

    data, rate = audio_io.load_audio(wav)

    assert data.ndim == 1
    assert rate == sr
    assert data.dtype == np.float32


def test_load_audio_stereo_wav_returns_samples_channels(tmp_path):
    sr = 44100
    n = sr  # 1 second
    stereo = np.column_stack([
        np.sin(np.linspace(0, 2 * np.pi * 440, n)).astype(np.float32),
        np.sin(np.linspace(0, 2 * np.pi * 880, n)).astype(np.float32),
    ])
    wav = _write_wav(tmp_path / "stereo.wav", stereo, sr)

    data, rate = audio_io.load_audio(wav)

    assert data.ndim == 2
    assert data.shape == (n, 2), data.shape  # (samples, channels)
    assert rate == sr


def test_load_audio_mono_flag_downmixes_stereo(tmp_path):
    sr = 22050
    n = sr
    stereo = np.column_stack([
        np.ones(n, dtype=np.float32) * 0.5,
        np.ones(n, dtype=np.float32) * -0.5,
    ])
    wav = _write_wav(tmp_path / "stereo.wav", stereo, sr)

    data, _ = audio_io.load_audio(wav, mono=True)

    assert data.ndim == 1
    assert data.shape == (n,)
    # left+right average = 0
    assert np.allclose(data, 0.0, atol=1e-6)


def test_load_audio_resamples_via_soxr(tmp_path):
    src_sr = 22050
    samples = np.linspace(-0.5, 0.5, src_sr, dtype=np.float32)  # 1s
    wav = _write_wav(tmp_path / "mono.wav", samples, src_sr)

    data, rate = audio_io.load_audio(wav, sr=44100)

    assert rate == 44100
    # length should roughly double
    assert abs(len(data) - 44100) < 100


def test_get_duration_soundfile_path(tmp_path):
    sr = 22050
    samples = np.zeros(sr * 3, dtype=np.float32)  # 3s
    wav = _write_wav(tmp_path / "x.wav", samples, sr)

    assert abs(audio_io.get_duration(wav) - 3.0) < 0.01


def test_get_samplerate_soundfile_path(tmp_path):
    samples = np.zeros(44100, dtype=np.float32)
    wav = _write_wav(tmp_path / "x.wav", samples, 44100)

    assert audio_io.get_samplerate(wav) == 44100


def test_get_duration_falls_back_to_ffprobe(tmp_path):
    """When soundfile can't open the file, _ffprobe_field is invoked."""
    fake = tmp_path / "fake.m4a"
    fake.write_bytes(b"not a real m4a")

    with patch(
        "musicue.analysis.audio_io._ffprobe_field", return_value="42.5"
    ) as mock:
        result = audio_io.get_duration(fake)

    assert result == 42.5
    mock.assert_called_once_with(fake, "format=duration")


def test_get_samplerate_falls_back_to_ffprobe(tmp_path):
    fake = tmp_path / "fake.m4a"
    fake.write_bytes(b"not a real m4a")

    with patch(
        "musicue.analysis.audio_io._ffprobe_field", return_value="48000"
    ) as mock:
        result = audio_io.get_samplerate(fake)

    assert result == 48000
    mock.assert_called_once_with(fake, "stream=sample_rate")


def test_load_audio_falls_back_to_ffmpeg_decode(tmp_path):
    """When sf.read raises LibsndfileError, _ffmpeg_decode handles the file."""
    fake = tmp_path / "fake.m4a"
    fake.write_bytes(b"not a real m4a")

    fake_pcm = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def raise_libsnd(*a, **kw):
        raise sf.LibsndfileError(0, prefix="")

    with patch("musicue.analysis.audio_io.sf.read", side_effect=raise_libsnd), \
         patch(
             "musicue.analysis.audio_io._ffmpeg_decode",
             return_value=(fake_pcm, 44100),
         ) as decode:
        data, rate = audio_io.load_audio(fake)

    assert rate == 44100
    assert np.array_equal(data, fake_pcm)
    decode.assert_called_once_with(fake)


def test_get_duration_raises_when_ffprobe_missing(tmp_path):
    fake = tmp_path / "fake.m4a"
    fake.write_bytes(b"not a real m4a")

    with patch("musicue.analysis.audio_io.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffprobe is not on PATH"):
            audio_io.get_duration(fake)
