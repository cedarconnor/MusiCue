import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from musicue.cli import app
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    CueSheet,
    MusiCueBundle,
    SourceInfo,
    TempoInfo,
)

runner = CliRunner()


def _write_minimal_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"fake-audio-bytes")
    sha = hashlib.sha256(audio.read_bytes()).hexdigest()

    analysis_dir = tmp_path / "song"
    analysis_dir.mkdir()
    analysis = AnalysisResult(
        source=SourceInfo(path=str(audio), sha256=sha, duration_sec=1.0, sample_rate=44100),
        analysis_config=AnalysisConfig(),
        stems={},
        tempo=TempoInfo(bpm_global=120.0),
    )
    analysis_path = analysis_dir / "analysis.json"
    analysis_path.write_text(analysis.model_dump_json())

    cuesheet = CueSheet(source_sha256=sha, grammar="concert_visuals", duration_sec=1.0)
    cuesheet_path = tmp_path / "song.cuesheet.json"
    cuesheet_path.write_text(cuesheet.model_dump_json())

    return audio, analysis_path, cuesheet_path


def test_export_bundle_writes_sibling_file(tmp_path):
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)

    result = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
    ])

    assert result.exit_code == 0, result.output
    expected = tmp_path / "song.musicue.json"
    assert expected.exists()

    bundle = MusiCueBundle.model_validate_json(expected.read_text())
    assert bundle.schema_version == "1.0"
    assert bundle.duration_sec == 1.0


def test_export_bundle_explicit_output(tmp_path):
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)
    output = tmp_path / "custom.musicue.json"

    result = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert output.exists()


def test_export_bundle_refuses_existing_without_force(tmp_path):
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)
    target = tmp_path / "song.musicue.json"
    target.write_text("{}")

    result = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
    ])
    assert result.exit_code != 0

    result2 = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--force",
    ])
    assert result2.exit_code == 0, result2.output
