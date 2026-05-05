from pathlib import Path
import json
import numpy as np
import soundfile as sf

from musicue.analysis.peaks import compute_peaks, write_peaks


def test_compute_peaks_shape(tmp_path: Path):
    sr = 44100
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), audio, sr)

    peaks = compute_peaks(wav, samples_per_pixel=1024)

    assert peaks["version"] == 2
    assert peaks["channels"] == 1
    assert peaks["sample_rate"] == sr
    assert peaks["samples_per_pixel"] == 1024
    expected_pixels = int(np.ceil(len(audio) / 1024))
    assert peaks["length"] == expected_pixels
    assert len(peaks["data"]) == expected_pixels * 2
    arr = np.array(peaks["data"], dtype=np.float32)
    assert arr.max() > 0.9
    assert arr.min() < -0.9


def test_compute_peaks_stereo(tmp_path: Path):
    sr = 44100
    audio = np.zeros((sr * 2, 2), dtype=np.float32)
    audio[:, 0] = 0.5
    audio[:, 1] = -0.5
    wav = tmp_path / "stereo.wav"
    sf.write(str(wav), audio, sr)

    peaks = compute_peaks(wav, samples_per_pixel=1024)
    assert peaks["channels"] == 1
    arr = np.array(peaks["data"])
    assert abs(arr).max() < 0.05


def test_write_peaks_json_roundtrip(tmp_path: Path):
    sr = 44100
    audio = (np.random.default_rng(0).standard_normal(sr) * 0.1).astype(np.float32)
    wav = tmp_path / "noise.wav"
    sf.write(str(wav), audio, sr)
    peaks_path = tmp_path / "peaks.mix.json"

    write_peaks(wav, peaks_path, samples_per_pixel=1024)

    assert peaks_path.exists()
    loaded = json.loads(peaks_path.read_text())
    assert loaded["version"] == 2
    assert loaded["sample_rate"] == sr
