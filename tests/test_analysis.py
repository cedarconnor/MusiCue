import shutil
import shutil as _shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from musicue.analysis.separation import separate, demucs_version
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.curves import compute_lufs_curve, compute_rms_curve


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


from musicue.analysis.pipeline import run_analysis
from musicue.config import MusiCueConfig
from musicue.schemas import AnalysisResult


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
