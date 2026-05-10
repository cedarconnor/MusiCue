"""Cache-hit branch must materialize run_dir artifacts.

Regression test for the case where `run_analysis` returned the cached
AnalysisResult without writing `analysis.json` / `peaks.*` into
`cfg.runs_dir`, causing the UI editor URL to 404 on the freshly-returned
analysis_id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from musicue.analysis.pipeline import _write_run_artifacts
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
