import shutil as _shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from musicue.analysis.curves import (
    compute_lufs_curve,
    compute_rms_curve,
    compute_spectral_centroid_curve,
    compute_spectral_flux_curve,
    compute_stereo_width_pan,
)
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.pipeline import run_analysis
from musicue.analysis.separation import demucs_version, separate
from musicue.config import MusiCueConfig
from musicue.schemas import AnalysisResult


def test_demucs_version_returns_string():
    v = demucs_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_separate_raises_on_demucs_failure(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        with pytest.raises(RuntimeError, match="Demucs failed"):
            separate(wav, tmp_path / "out")


def test_separate_returns_four_stem_paths(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals", "other"):
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        stems = separate(wav, out_dir, model=model)

    assert set(stems.keys()) == {"drums", "bass", "vocals", "other"}
    for p in stems.values():
        assert p.exists()


def test_separate_raises_when_stem_missing(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals"):  # "other" missing
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with pytest.raises(FileNotFoundError, match="other"):
            separate(wav, out_dir, model=model)


def test_detect_onsets_returns_list(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    assert isinstance(onsets, list)
    assert len(onsets) >= 2


def test_onset_event_fields(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    assert len(onsets) > 0
    o = onsets[0]
    assert "t" in o and "strength" in o and "timescale" in o and "labels" in o
    assert 0.0 <= o["strength"] <= 1.0
    assert o["timescale"] == "micro"
    assert o["labels"] == []
    assert o["drum_class"] is None


def test_onset_times_ascending(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    times = [o["t"] for o in onsets]
    assert times == sorted(times)


def test_onsets_detect_all_four_bursts(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    times = [o["t"] for o in onsets]
    for burst_t in (0.5, 2.5, 5.0, 7.5):
        assert any(abs(t - burst_t) < 0.25 for t in times), f"No onset near {burst_t}s"


def test_lufs_curve_structure(synthetic_wav):
    curve = compute_lufs_curve(synthetic_wav, hop_sec=0.04)
    assert "hop_sec" in curve and "values" in curve
    assert curve["hop_sec"] == pytest.approx(0.04, abs=0.01)
    assert len(curve["values"]) > 0
    assert all(isinstance(v, float) for v in curve["values"])


def test_lufs_curve_range(synthetic_wav):
    curve = compute_lufs_curve(synthetic_wav, hop_sec=0.04)
    assert all(-70.0 <= v <= 0.0 for v in curve["values"])


def test_rms_curve_non_negative(synthetic_wav):
    curve = compute_rms_curve(synthetic_wav, hop_sec=0.04)
    assert len(curve["values"]) > 0
    assert all(v >= 0.0 for v in curve["values"])


def test_spectral_centroid_curve(synthetic_wav):
    c = compute_spectral_centroid_curve(synthetic_wav, hop_sec=0.04)
    assert len(c["values"]) > 0
    assert all(v >= 0 for v in c["values"])


def test_spectral_flux_curve(synthetic_wav):
    c = compute_spectral_flux_curve(synthetic_wav, hop_sec=0.04)
    assert len(c["values"]) > 0
    assert all(v >= 0 for v in c["values"])


def test_stereo_width_pan_on_mono_returns_zero(synthetic_wav):
    result = compute_stereo_width_pan(synthetic_wav, hop_sec=0.04)
    # synthetic_wav is mono; width should be 0 or near 0
    assert "width" in result and "pan" in result
    assert all(abs(v) < 0.01 for v in result["width"]["values"])


def _make_cfg(tmp_path):
    cfg = MusiCueConfig()
    cfg.cache_dir = tmp_path / "cache"
    cfg.runs_dir = tmp_path / "runs"
    return cfg


def _fake_separate(audio_path, out_dir, model):
    stem_dir = out_dir / model / audio_path.stem
    stem_dir.mkdir(parents=True)
    stems = {}
    for s in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{s}.wav"
        _shutil.copy(audio_path, p)
        stems[s] = p
    return stems


def test_pipeline_returns_analysis_result(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    assert isinstance(result, AnalysisResult)
    assert result.source.duration_sec > 0
    assert set(result.stems.keys()) == {"drums", "bass", "vocals", "other"}
    assert "drums" in result.onsets
    assert "lufs" in result.curves
    assert "rms_drums" in result.curves


def test_pipeline_source_sha256_matches_file(tmp_path, synthetic_wav):
    import hashlib
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    expected = hashlib.sha256(synthetic_wav.read_bytes()).hexdigest()
    assert result.source.sha256 == expected


def test_pipeline_caches_and_skips_demucs_on_rerun(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate) as mock_sep:
        run_analysis(synthetic_wav, cfg)
        run_analysis(synthetic_wav, cfg)
    assert mock_sep.call_count == 1  # demucs ran only once


def test_m1_pipeline_includes_beats_and_sections(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)

    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        with patch("musicue.analysis.pipeline.detect_structure") as mock_struct:
            mock_struct.return_value = {
                "tempo": {
                    "bpm_global": 120.0,
                    "bpm_curve": [{"t": 0.0, "bpm": 120.0}],
                    "time_signature": [4, 4],
                },
                "beats": [
                    {
                        "t": 0.5,
                        "beat_in_bar": 1,
                        "bar": 1,
                        "is_downbeat": True,
                        "confidence": 0.9,
                        "timescale": "micro",
                    },
                ],
                "sections": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "label": "intro",
                        "confidence": 0.9,
                        "timescale": "macro",
                    },
                ],
            }
            result = run_analysis(synthetic_wav, cfg)

    assert result.tempo is not None
    assert result.tempo.bpm_global == pytest.approx(120.0)
    assert len(result.beats) == 1
    assert result.beats[0].is_downbeat is True
    assert len(result.sections) == 1
    assert result.sections[0].label == "intro"


def test_m1_pipeline_includes_spectral_curves(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    assert "spectral_centroid" in result.curves
    assert "spectral_flux" in result.curves


@pytest.mark.integration
def test_full_pipeline_wav_to_csv(tmp_path, synthetic_wav):
    """Full pipeline on real audio. Requires demucs installed. Mark: integration."""
    import csv as csv_mod

    from musicue.compile.compiler import compile_analysis
    from musicue.exporters.csv import export as csv_export

    cfg = MusiCueConfig()
    cfg.cache_dir = tmp_path / "cache"
    cfg.runs_dir = tmp_path / "runs"

    analysis = run_analysis(synthetic_wav, cfg)
    assert analysis.source.duration_sec > 0
    assert "drums" in analysis.onsets
    assert "lufs" in analysis.curves

    cuesheet = compile_analysis(analysis, grammar="concert_visuals")
    assert cuesheet.duration_sec > 0
    assert len(cuesheet.tracks) >= 1

    out_csv = tmp_path / "output.csv"
    csv_export(cuesheet, out_csv)
    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) > 0
    assert "time_sec" in rows[0]


def test_onset_strength_nonzero_and_tracks_loudness(tmp_path):
    """Strength is read at the envelope peak (not the backtracked minimum),
    so it is > 0 and louder hits score higher."""
    sr = 22050
    rng = np.random.default_rng(0)
    # Steady bed so the hits' flux is measured against real signal, not
    # digital silence (where any hit is an "infinite" dB rise).
    y = (0.02 * rng.standard_normal(int(sr * 6.0))).astype(np.float32)
    burst_len = int(0.05 * sr)
    decay = np.exp(-np.arange(burst_len) / (0.01 * sr)).astype(np.float32)
    amps = {1.0: 0.05, 2.5: 0.2, 4.0: 0.8}
    for t0, amp in amps.items():
        i = int(t0 * sr)
        y[i:i + burst_len] += amp * decay * rng.standard_normal(burst_len).astype(np.float32)
    p = tmp_path / "hits.wav"
    sf.write(str(p), y, sr)

    onsets = detect_onsets(p, sr=sr)
    by_hit = {}
    for t0 in amps:
        near = [o for o in onsets if abs(o["t"] - t0) < 0.1]
        assert near, f"no onset near {t0}s"
        by_hit[t0] = max(o["strength"] for o in near)
    assert all(s > 0.0 for s in by_hit.values()), by_hit
    assert by_hit[1.0] < by_hit[2.5] < by_hit[4.0], by_hit
    assert all(0.0 <= o["strength"] <= 1.0 for o in onsets)


def test_lufs_curve_is_centered_on_step(tmp_path):
    """A loudness step at 12 s must cross its (power) midpoint at ~12 s, not
    0.2 s early as with a look-ahead window stamped at its start."""
    sr = 44100
    t = np.arange(int(sr * 24.0)) / sr
    amp = np.where(t < 12.0, 0.05, 0.5)
    y = (amp * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    p = tmp_path / "step.wav"
    sf.write(str(p), y, sr)

    curve = compute_lufs_curve(p, hop_sec=0.01)
    hop = curve["hop_sec"]
    power = 10.0 ** (np.asarray(curve["values"]) / 10.0)
    lo = float(np.median(power[: int(10.0 / hop)]))
    hi = float(np.median(power[int(14.0 / hop):]))
    mid = (lo + hi) / 2.0
    cross_idx = int(np.argmax(power > mid))
    assert abs(cross_idx * hop - 12.0) <= 0.05, cross_idx * hop
