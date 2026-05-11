import subprocess
import sys


def test_cli_prints_table_and_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "musicue.health.readiness", "--print-table"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for name in (
        "python_venv",
        "torch",
        "cuda",
        "ffmpeg",
        "demucs",
        "basic_pitch",
        "allin1",
        "clap",
    ):
        assert name in result.stdout
    assert "overall:" in result.stdout.lower()


def test_cli_json_mode_emits_parseable_json():
    import json

    result = subprocess.run(
        [sys.executable, "-m", "musicue.health.readiness", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "components" in data
    assert "overall" in data
