"""Server-side loop persistence."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

SONG_ID = "1" * 64
ANALYSIS_ID = "a" * 12
LOOP_URL = f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/loop"


def _seed(tmp_path: Path) -> None:
    d = tmp_path / "songs" / SONG_ID / "analyses" / ANALYSIS_ID
    d.mkdir(parents=True)
    (tmp_path / "songs" / SONG_ID / "title.txt").write_text("t")
    (tmp_path / "songs" / SONG_ID / "source.wav").write_bytes(b"\0")
    (d / "analysis.json").write_text(
        json.dumps({"schema_version": "v3", "source": {}, "tempo": {}})
    )


def test_loop_round_trip(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    from musicue.index import index as indexer

    indexer.rebuild(app.state.index_db, tmp_path)
    client = TestClient(app)
    assert client.get(LOOP_URL).status_code == 204
    r = client.put(
        LOOP_URL,
        json={"loop_in": 1.0, "loop_out": 2.0, "enabled": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["loop_in"] == 1.0 and body["enabled"] is True

    sidecar = tmp_path / "songs" / SONG_ID / "analyses" / ANALYSIS_ID / "loops.json"
    assert sidecar.exists()
    persisted = json.loads(sidecar.read_text())
    assert persisted["loop_out"] == 2.0

    r2 = client.get(LOOP_URL)
    assert r2.status_code == 200
    assert r2.json()["loop_out"] == 2.0


def test_loop_survives_db_wipe(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    from musicue.index import index as indexer
    from musicue.index import schema

    indexer.rebuild(app.state.index_db, tmp_path)
    client = TestClient(app)
    client.put(
        LOOP_URL,
        json={"loop_in": 5.0, "loop_out": 9.0, "enabled": True},
    )
    app.state.index_db.close()
    db_path = tmp_path / "index.db"
    db_path.unlink()
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    schema.create_all(db)
    indexer.rebuild(db, tmp_path)
    app.state.index_db = db

    r = client.get(LOOP_URL)
    assert r.status_code == 200 and r.json()["loop_in"] == 5.0


def test_loop_rejects_invalid_bounds(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    from musicue.index import index as indexer

    indexer.rebuild(app.state.index_db, tmp_path)
    client = TestClient(app)
    r = client.put(
        LOOP_URL,
        json={"loop_in": 5.0, "loop_out": 5.0, "enabled": True},
    )
    assert r.status_code == 400
