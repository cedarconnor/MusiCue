"""Cache-hit branch must materialize run_dir artifacts.

Regression test for the case where `run_analysis` returned the cached
AnalysisResult without writing `analysis.json` / `peaks.*` into
`cfg.runs_dir`, causing the UI editor URL to 404 on the freshly-returned
analysis_id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicue.analysis.pipeline import _version_dict, _write_run_artifacts
from musicue.config import MusiCueConfig
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    SourceInfo,
)


def _wav(tmp_path: Path, name: str = "x.wav") -> Path:
    p = tmp_path / name
    sf.write(str(p), np.zeros(44100, dtype=np.float32), 44100)
    return p


def test_write_run_artifacts_writes_analysis_json(tmp_path):
    audio = _wav(tmp_path)
    run_dir = tmp_path / "run"
    result = AnalysisResult(
        source=SourceInfo(path=str(audio), sha256="abc", duration_sec=1.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4"),
        stems={},
    )
    _write_run_artifacts(result, audio, run_dir)
    assert (run_dir / "analysis.json").exists()
    assert (run_dir / "peaks.mix.json").exists()


def test_write_run_artifacts_writes_stem_peaks_when_files_exist(tmp_path):
    audio = _wav(tmp_path)
    stem_audio = _wav(tmp_path, "drums.wav")
    run_dir = tmp_path / "run"
    result = AnalysisResult(
        source=SourceInfo(path=str(audio), sha256="abc", duration_sec=1.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4"),
        stems={"drums": str(stem_audio)},
    )
    _write_run_artifacts(result, audio, run_dir)
    assert (run_dir / "peaks.drums.json").exists()


def test_write_run_artifacts_skips_missing_stems_gracefully(tmp_path):
    audio = _wav(tmp_path)
    run_dir = tmp_path / "run"
    result = AnalysisResult(
        source=SourceInfo(path=str(audio), sha256="abc", duration_sec=1.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4"),
        stems={"drums": str(tmp_path / "nope.wav")},  # path doesn't exist
    )
    # Should not raise even though the stem audio is missing.
    _write_run_artifacts(result, audio, run_dir)
    assert (run_dir / "analysis.json").exists()
    assert not (run_dir / "peaks.drums.json").exists()


def test_version_dict_includes_behavior_affecting_analysis_settings():
    cfg = MusiCueConfig()
    cfg.analysis.phrase_gap_sec = {"vocals": 1.2, "other": 0.2}
    cfg.analysis.clap_top_k = 7
    cfg.analysis.clap_threshold = 0.8

    versions = _version_dict(cfg)

    assert versions["phrase_gap_sec"] == {"vocals": 1.2, "other": 0.2}
    assert versions["clap_top_k"] == 7
    assert versions["clap_threshold"] == 0.8


def test_version_dict_includes_code_versions():
    import musicue
    from musicue.analysis import pipeline

    versions = _version_dict(MusiCueConfig())
    assert versions["musicue_version"] == musicue.__version__
    assert versions["analysis_algo_version"] == pipeline.ANALYSIS_ALGO_VERSION


def test_cache_key_changes_with_analysis_algo_version(tmp_path, monkeypatch):
    from musicue.analysis import pipeline
    from musicue.cache import build_audio_cache_key

    audio = _wav(tmp_path)
    cfg = MusiCueConfig()
    key_a = build_audio_cache_key(audio, _version_dict(cfg))
    monkeypatch.setattr(pipeline, "ANALYSIS_ALGO_VERSION", "test-bump")
    key_b = build_audio_cache_key(audio, _version_dict(cfg))
    assert key_a != key_b


class _StopPipeline(Exception):
    pass


def _stop(*_a, **_k):
    raise _StopPipeline


def test_run_analysis_force_bypasses_cache(tmp_path, monkeypatch):
    from musicue.analysis import pipeline
    from musicue.cache import Cache

    audio = _wav(tmp_path)
    cfg = MusiCueConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    lookups: list[str] = []

    def _fake_get(self, key, suffix):
        lookups.append(key)
        return None

    monkeypatch.setattr(Cache, "get", _fake_get)
    # Stop right after the cache check: _sha256 is the first fresh-path step.
    monkeypatch.setattr(pipeline, "_sha256", _stop)

    with pytest.raises(_StopPipeline):
        pipeline.run_analysis(audio, cfg)
    assert len(lookups) == 1

    with pytest.raises(_StopPipeline):
        pipeline.run_analysis(audio, cfg, force=True)
    assert len(lookups) == 1, "force=True must not consult the cache"
