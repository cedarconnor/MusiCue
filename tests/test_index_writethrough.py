"""User-data writes go through both DB and filesystem sidecar."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from musicue.index import index as indexer
from musicue.index import query, schema


def _bootstrap(tmp_path: Path) -> sqlite3.Connection:
    (tmp_path / "songs" / "s1" / "analyses" / "a1").mkdir(parents=True)
    (tmp_path / "songs" / "s1" / "title.txt").write_text("t")
    (tmp_path / "songs" / "s1" / "source.wav").write_bytes(b"\0")
    (tmp_path / "songs" / "s1" / "analyses" / "a1" / "analysis.json").write_text(
        json.dumps({"schema_version": "v3", "source": {}, "tempo": {}})
    )
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)
    return db


def test_set_trashed_writes_sidecar_and_db(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    indexer.set_trashed(db, tmp_path, "s1", trashed=True)
    sidecar = tmp_path / "songs" / "s1" / ".trashed_at"
    assert sidecar.exists()
    (got,) = db.execute("SELECT trashed_at FROM songs WHERE id='s1'").fetchone()
    assert got and got == sidecar.read_text(encoding="utf-8").strip()


def test_set_trashed_clears_both(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    indexer.set_trashed(db, tmp_path, "s1", trashed=True)
    indexer.set_trashed(db, tmp_path, "s1", trashed=False)
    sidecar = tmp_path / "songs" / "s1" / ".trashed_at"
    assert not sidecar.exists()
    (got,) = db.execute("SELECT trashed_at FROM songs WHERE id='s1'").fetchone()
    assert got is None


def test_set_loop_writes_sidecar_and_upserts(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    indexer.set_loop(
        db, tmp_path, "s1", "a1",
        {"loop_in": 1.0, "loop_out": 2.0, "enabled": True},
    )
    sidecar = tmp_path / "songs" / "s1" / "analyses" / "a1" / "loops.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["loop_in"] == 1.0 and payload["enabled"] is True
    assert query.get_loop(db, "s1", "a1") == {
        "loop_in": 1.0,
        "loop_out": 2.0,
        "enabled": True,
        "updated_at": payload["updated_at"],
    }
    indexer.set_loop(
        db, tmp_path, "s1", "a1",
        {"loop_in": 5.0, "loop_out": 9.0, "enabled": False},
    )
    assert query.get_loop(db, "s1", "a1")["loop_in"] == 5.0


def test_rebuild_after_db_wipe_recovers_user_data(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    indexer.set_trashed(db, tmp_path, "s1", trashed=True)
    indexer.set_loop(
        db, tmp_path, "s1", "a1",
        {"loop_in": 1.0, "loop_out": 2.0, "enabled": True},
    )
    db.close()

    (tmp_path / "i.db").unlink()
    db2 = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db2)
    indexer.rebuild(db2, tmp_path)

    (ts,) = db2.execute(
        "SELECT trashed_at FROM songs WHERE id='s1'"
    ).fetchone()
    assert ts is not None
    assert query.get_loop(db2, "s1", "a1") is not None


def test_delete_song_removes_dir_and_row(tmp_path: Path) -> None:
    db = _bootstrap(tmp_path)
    indexer.set_trashed(db, tmp_path, "s1", trashed=True)
    indexer.delete_song(db, tmp_path, "s1")
    assert not (tmp_path / "songs" / "s1").exists()
    (n,) = db.execute(
        "SELECT COUNT(*) FROM songs WHERE id='s1'"
    ).fetchone()
    assert n == 0


def test_delete_song_refuses_when_not_trashed(tmp_path: Path) -> None:
    import pytest

    db = _bootstrap(tmp_path)
    with pytest.raises(ValueError):
        indexer.delete_song(db, tmp_path, "s1")
