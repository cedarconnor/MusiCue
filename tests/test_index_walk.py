"""Indexer: filesystem walk -> SQLite rows."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from musicue.index import index as indexer
from musicue.index import schema


def _make_song(
    root: Path,
    sid: str,
    title: str,
    *,
    source_url: str | None = None,
    trashed_at: str | None = None,
    thumbnail: bool = False,
    analysis: dict | None = None,
    loops: dict | None = None,
) -> None:
    d = root / "songs" / sid
    (d / "analyses").mkdir(parents=True, exist_ok=True)
    (d / "title.txt").write_text(title, encoding="utf-8")
    (d / "source.wav").write_bytes(b"\x00")
    if source_url:
        (d / "source_url.txt").write_text(source_url, encoding="utf-8")
    if trashed_at:
        (d / ".trashed_at").write_text(trashed_at, encoding="utf-8")
    if thumbnail:
        (d / "thumbnail.jpg").write_bytes(b"jpgbytes")
    if analysis is not None:
        aid = "a1"
        ad = d / "analyses" / aid
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "analysis.json").write_text(
            json.dumps(analysis), encoding="utf-8"
        )
        if loops is not None:
            (ad / "loops.json").write_text(
                json.dumps(loops), encoding="utf-8"
            )


def test_rebuild_populates_songs(tmp_path: Path) -> None:
    _make_song(tmp_path, "abc", "Hello", source_url="https://x/y", thumbnail=True)
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)
    rows = db.execute(
        "SELECT id, title, source_url, has_thumbnail, trashed_at FROM songs"
    ).fetchall()
    assert rows == [("abc", "Hello", "https://x/y", 1, None)]


def test_rebuild_picks_up_trashed_at(tmp_path: Path) -> None:
    _make_song(tmp_path, "abc", "Hello", trashed_at="2026-05-09T12:00:00Z")
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)
    (ts,) = db.execute("SELECT trashed_at FROM songs").fetchone()
    assert ts == "2026-05-09T12:00:00Z"


def test_rebuild_records_analysis_flags(tmp_path: Path) -> None:
    analysis = {
        "schema_version": "v3",
        "source": {"duration_sec": 222.5},
        "tempo": {"bpm_global": 128.0},
        "lufs_integrated": -8.3,
        "onsets": {
            "drums": [{"t": 1.0, "drum_class": "kick", "labels": []}],
            "bass": [{"t": 1.0, "drum_class": None, "labels": ["bass"]}],
        },
    }
    _make_song(tmp_path, "abc", "Hello", analysis=analysis)
    (tmp_path / "songs" / "abc" / "analyses" / "a1" / "stems").mkdir()

    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)

    (bpm, dur, lufs) = db.execute(
        "SELECT bpm_global, duration_sec, lufs_integrated FROM songs"
    ).fetchone()
    assert bpm == 128.0 and dur == 222.5 and lufs == -8.3

    (has_stems, has_clap, has_drum_cls, ver) = db.execute(
        "SELECT has_stems, has_clap, has_drum_cls, schema_ver FROM analyses"
    ).fetchone()
    assert has_stems == 1 and has_clap == 1 and has_drum_cls == 1 and ver == "v3"


def test_rebuild_imports_loops_json(tmp_path: Path) -> None:
    analysis = {"schema_version": "v3", "source": {}, "tempo": {}}
    loops = {
        "loop_in": 1.0,
        "loop_out": 2.0,
        "enabled": True,
        "updated_at": "2026-05-09T12:00:00Z",
    }
    _make_song(tmp_path, "abc", "Hello", analysis=analysis, loops=loops)
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)
    row = db.execute(
        "SELECT loop_in, loop_out, enabled FROM loop_regions"
    ).fetchone()
    assert row == (1.0, 2.0, 1)


def test_ensure_current_rebuilds_on_version_mismatch(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    db.execute("CREATE TABLE songs (id TEXT)")
    db.execute("PRAGMA user_version = 0")
    db.commit()
    _make_song(tmp_path, "abc", "Hello")
    indexer.ensure_current(db, tmp_path)
    (n,) = db.execute("SELECT COUNT(*) FROM songs").fetchone()
    assert n == 1
    (uv,) = db.execute("PRAGMA user_version").fetchone()
    assert uv == schema.SCHEMA_VERSION


def test_ensure_current_skips_when_already_current(tmp_path: Path) -> None:
    """When the schema version matches AND the filesystem agrees with the
    DB row count, ensure_current is a no-op."""
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    _make_song(tmp_path, "abc", "Hello")
    assert indexer.ensure_current(db, tmp_path) is True  # first call rebuilds
    # Second call with matching state should not rebuild. We can't rely on
    # a forged sentinel any more (drift detection would catch it), so we
    # assert the boolean return value.
    assert indexer.ensure_current(db, tmp_path) is False


def test_ensure_current_resyncs_on_filesystem_drift(tmp_path: Path) -> None:
    """If the filesystem grows new song dirs without going through the
    write-through API, ensure_current rebuilds on next startup so the
    Library actually shows them."""
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.ensure_current(db, tmp_path)  # empty
    _make_song(tmp_path, "abc", "Hello")
    assert indexer.ensure_current(db, tmp_path) is True
    (n,) = db.execute("SELECT COUNT(*) FROM songs").fetchone()
    assert n == 1
