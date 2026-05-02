from pathlib import Path

import pytest

from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    OnsetEvent,
    SourceInfo,
    TimedCurve,
)


def _make_analysis_json(tmp_path: Path) -> Path:
    result = AnalysisResult(
        source=SourceInfo(path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={"drums": [OnsetEvent(t=0.5, strength=0.9), OnsetEvent(t=1.0, strength=0.8)]},
        curves={"lufs": TimedCurve(hop_sec=0.04, values=[-20.0] * 250)},
    )
    p = tmp_path / "analysis.json"
    p.write_text(result.model_dump_json())
    return p


def test_summary_returns_dict(tmp_path):
    from musicue.inspect import summarize
    path = _make_analysis_json(tmp_path)
    summary = summarize(path)
    assert "duration_sec" in summary
    assert "onset_counts" in summary
    assert summary["duration_sec"] == pytest.approx(10.0)
    assert summary["onset_counts"]["drums"] == 2


def test_summary_lists_curves(tmp_path):
    from musicue.inspect import summarize
    path = _make_analysis_json(tmp_path)
    summary = summarize(path)
    assert "curves" in summary
    assert "lufs" in summary["curves"]
