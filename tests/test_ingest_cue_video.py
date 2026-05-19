"""Verify the analyze ingest path auto-generates cue_video.mp4."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest  # noqa: F401  -- used implicitly by pytest-asyncio auto mode

from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    CueSheet,
    CueTrack,
    SourceInfo,
)


def _stub_analysis(audio_path: Path) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(
            path=str(audio_path), sha256="abc",
            duration_sec=2.0, sample_rate=44100,
        ),
        analysis_config=AnalysisConfig(),
        stems={},
    )


def _stub_cuesheet() -> CueSheet:
    return CueSheet(
        source_sha256="abc", grammar="concert_visuals", duration_sec=2.0,
        tracks=[CueTrack(
            name="x", type="impulse", timescale="micro",
            events=[{
                "t": 1.0, "strength": 1.0,
                "envelope": {"a": 0.01, "d": 0.1, "s": 0.0, "r": 0.0},
            }],
        )],
    )


async def test_default_analyze_invokes_render_cue_video(
    tmp_path, synthetic_wav, monkeypatch
) -> None:
    from musicue.ui.routes import songs

    actual = tmp_path / "abc"
    actual.mkdir()
    # Write the stub analysis.json the hook expects to read.
    (actual / "analysis.json").write_text(_stub_analysis(synthetic_wav).model_dump_json())

    monkeypatch.setattr(
        "musicue.analysis.pipeline.run_analysis",
        lambda *a, **kw: _stub_analysis(synthetic_wav),
    )
    monkeypatch.setattr(
        "musicue.analysis.pipeline.compute_run_dir",
        lambda *a, **kw: actual,
    )
    monkeypatch.setattr(
        "musicue.compile.compiler.compile_analysis",
        lambda *a, **kw: _stub_cuesheet(),
    )

    captured = {}

    def fake_render(cuesheet, audio_path, out_path, **kw):
        captured["called"] = True
        captured["out_path"] = Path(out_path)
        Path(out_path).write_bytes(b"FAKE")

    monkeypatch.setattr(
        "musicue.visualize.cue_video.render_cue_video", fake_render
    )

    publish = AsyncMock()
    await songs._default_analyze(
        audio_path=synthetic_wav,
        run_dir=actual / "pending",
        publish=publish,
        job_id=None,
        pool=None,
    )
    assert captured.get("called") is True
    assert captured["out_path"] == actual / "cue_video.mp4"
    assert (actual / "cuesheet.json").exists()


async def test_default_analyze_survives_render_failure(
    tmp_path, synthetic_wav, monkeypatch
) -> None:
    from musicue.ui.routes import songs

    actual = tmp_path / "abc"
    actual.mkdir()
    (actual / "analysis.json").write_text(_stub_analysis(synthetic_wav).model_dump_json())

    monkeypatch.setattr(
        "musicue.analysis.pipeline.run_analysis",
        lambda *a, **kw: _stub_analysis(synthetic_wav),
    )
    monkeypatch.setattr(
        "musicue.analysis.pipeline.compute_run_dir",
        lambda *a, **kw: actual,
    )
    monkeypatch.setattr(
        "musicue.compile.compiler.compile_analysis",
        lambda *a, **kw: _stub_cuesheet(),
    )

    def boom(*a, **kw):
        raise RuntimeError("intentional render failure")

    monkeypatch.setattr(
        "musicue.visualize.cue_video.render_cue_video", boom
    )

    publish = AsyncMock()
    result = await songs._default_analyze(
        audio_path=synthetic_wav,
        run_dir=actual / "pending",
        publish=publish,
        job_id=None,
        pool=None,
    )
    assert result == "abc"
