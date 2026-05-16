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
    assert bundle.schema_version == "1.1"
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


def test_committed_fixture_roundtrips():
    fixture = Path("tests/fixtures/sample.musicue.json")
    if not fixture.exists():
        return
    bundle = MusiCueBundle.model_validate_json(fixture.read_text())
    assert bundle.schema_version.startswith("1.")


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


def test_export_bundle_folder_writes_full_layout(tmp_path):
    """--folder switches to the four-file project layout."""
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)
    out = tmp_path / "exports" / "song"

    res = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--folder", str(out),
    ])
    assert res.exit_code == 0, res.output

    assert (out / "song.wav").exists()
    assert (out / "song.musicue.json").exists()
    assert (out / "manifest.json").exists()
    assert not (out / "stems").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == "cedartoy-project/1"
    assert manifest["grammar"] == "concert_visuals"


def test_export_bundle_folder_include_stems(tmp_path):
    """--include-stems copies stems from --stems-dir when present."""
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)

    stems = tmp_path / "stems"
    stems.mkdir()
    for name in ("drums", "bass", "vocals", "other"):
        (stems / f"{name}.wav").write_bytes(b"fake-stem-bytes")

    out = tmp_path / "exports" / "song"
    res = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--folder", str(out),
        "--include-stems",
        "--stems-dir", str(stems),
    ])
    assert res.exit_code == 0, res.output
    for name in ("drums", "bass", "vocals", "other"):
        assert (out / "stems" / f"{name}.wav").exists()


def test_export_bundle_include_stems_requires_folder(tmp_path):
    """--include-stems without --folder is a usage error."""
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)
    res = runner.invoke(app, [
        "export-bundle", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--include-stems",
    ])
    assert res.exit_code == 2


def test_send_to_cedartoy_alias_runs(tmp_path):
    """`send-to-cedartoy` alias produces the same folder layout."""
    audio, analysis_path, cuesheet_path = _write_minimal_artifacts(tmp_path)
    out = tmp_path / "exports" / "song"

    res = runner.invoke(app, [
        "send-to-cedartoy", str(audio),
        "--analysis", str(analysis_path),
        "--cuesheet", str(cuesheet_path),
        "--output", str(out),
        "--no-stems",
    ])
    assert res.exit_code == 0, res.output
    assert (out / "song.musicue.json").exists()
    assert (out / "manifest.json").exists()
