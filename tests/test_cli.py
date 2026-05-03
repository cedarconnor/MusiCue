import subprocess
import sys


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "musicue"] + list(args),
        capture_output=True, text=True,
    )


def test_top_level_help():
    r = cli("--help")
    assert r.returncode == 0
    assert "analyze" in r.stdout


def test_analyze_help():
    r = cli("analyze", "--help")
    assert r.returncode == 0
    assert "SONG" in r.stdout or "song" in r.stdout


def test_compile_help():
    r = cli("compile", "--help")
    assert r.returncode == 0


def test_export_help():
    r = cli("export", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout


def test_render_help():
    r = cli("render", "--help")
    assert r.returncode == 0


def test_inspect_help():
    r = cli("inspect", "--help")
    assert r.returncode == 0
    assert "analysis" in r.stdout.lower()


def test_plot_help():
    r = cli("plot", "--help")
    assert r.returncode == 0


def test_listen_help():
    r = cli("listen", "--help")
    assert r.returncode == 0
    assert "click" in r.stdout.lower() or "audio" in r.stdout.lower()


def test_diff_help():
    r = cli("diff", "--help")
    assert r.returncode == 0


def test_export_unknown_target_exits_nonzero(tmp_path):
    from musicue.schemas import CueSheet
    cs = CueSheet(source_sha256="x", grammar="g", duration_sec=1.0, tempo_map=[], tracks=[])
    cuesheet_path = tmp_path / "cs.json"
    cuesheet_path.write_text(cs.model_dump_json())
    r = cli("export", str(cuesheet_path), "--target", "nonexistent_format",
            "--out", str(tmp_path / "out"))
    assert r.returncode != 0


def test_render_batch_help():
    r = cli("render", "--help")
    assert r.returncode == 0
    assert "--batch" in r.stdout or "batch" in r.stdout.lower()


def test_render_batch_requires_directory(tmp_path):
    """--batch with a file path (not directory) should exit nonzero."""
    # Create a real file so the test exercises the directory check, not file existence
    fake_audio = tmp_path / "nofile.wav"
    fake_audio.write_bytes(b"")
    r = cli("render", str(fake_audio), "--batch")
    assert r.returncode != 0


def test_render_batch_empty_directory_errors(tmp_path):
    """--batch on empty directory should error."""
    r = cli("render", str(tmp_path), "--batch")
    assert r.returncode != 0
