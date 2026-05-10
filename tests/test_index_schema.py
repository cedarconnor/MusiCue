"""Schema bootstrap + FTS5 detection."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from musicue.index import schema


def test_create_all_yields_expected_tables(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    names = {
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        )
    }
    assert {"songs", "analyses", "loop_regions"} <= names
    assert "songs_title_idx" in names
    assert "songs_trashed_idx" in names
    # FTS5 either present (default) or absent (graceful skip).
    assert ("songs_fts" in names) == schema.has_fts5(db)


def test_user_version_set_to_schema_version(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    (got,) = db.execute("PRAGMA user_version").fetchone()
    assert got == schema.SCHEMA_VERSION


def test_drop_all_removes_everything(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    schema.drop_all(db)
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
    ).fetchall()
    assert rows == []
