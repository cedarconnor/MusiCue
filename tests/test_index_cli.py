"""`musicue index` CLI rebuilds and prints status."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _seed_song(root: Path, sid: str = "s1") -> None:
    (root / "songs" / sid / "analyses" / "a1").mkdir(parents=True)
    (root / "songs" / sid / "title.txt").write_text("t")
    (root / "songs" / sid / "source.wav").write_bytes(b"\0")
    (root / "songs" / sid / "analyses" / "a1" / "analysis.json").write_text(
        json.dumps({"schema_version": "v3", "source": {}, "tempo": {}})
    )


def test_index_rebuild_creates_db(tmp_path: Path) -> None:
    _seed_song(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "musicue.cli", "index", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "rebuilt" in proc.stdout.lower() or "indexed" in proc.stdout.lower()
    assert (tmp_path / "index.db").exists()


def test_index_status_prints_counts(tmp_path: Path) -> None:
    _seed_song(tmp_path)
    subprocess.run(
        [sys.executable, "-m", "musicue.cli", "index", "--root", str(tmp_path)],
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "musicue.cli",
            "index",
            "--status",
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "songs: 1" in proc.stdout
    assert "analyses: 1" in proc.stdout
