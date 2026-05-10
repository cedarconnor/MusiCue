"""Trash + untrash + hard-delete + job-in-flight 409."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _seed(tmp_path: Path, sid: str = "s1", aid: str = "a1") -> None:
    d = tmp_path / "songs" / sid / "analyses" / aid
    d.mkdir(parents=True)
    (tmp_path / "songs" / sid / "title.txt").write_text("t")
    (tmp_path / "songs" / sid / "source.wav").write_bytes(b"\0")
    (d / "analysis.json").write_text(
        json.dumps({"schema_version": "v3", "source": {}, "tempo": {}})
    )


def _refresh_index(app, tmp_path: Path) -> None:
    from musicue.index import index as indexer

    indexer.rebuild(app.state.index_db, tmp_path)


def test_trash_then_untrash(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh_index(app, tmp_path)
    client = TestClient(app)
    assert client.post("/api/songs/s1/trash").status_code == 200
    assert (tmp_path / "songs" / "s1" / ".trashed_at").exists()
    assert {s["id"] for s in client.get("/api/songs").json()["songs"]} == set()
    assert client.post("/api/songs/s1/untrash").status_code == 200
    assert not (tmp_path / "songs" / "s1" / ".trashed_at").exists()


def test_hard_delete_requires_trash(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh_index(app, tmp_path)
    client = TestClient(app)
    assert client.delete("/api/songs/s1").status_code == 409
    client.post("/api/songs/s1/trash")
    assert client.delete("/api/songs/s1").status_code == 200
    assert not (tmp_path / "songs" / "s1").exists()


def test_trash_blocks_when_job_in_flight(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh_index(app, tmp_path)
    job = app.state.jobs.submit(kind="analyze", payload={"song_id": "s1"})
    client = TestClient(app)
    r = client.post("/api/songs/s1/trash")
    assert r.status_code == 409
    assert "job" in r.json()["detail"].lower()
    app.state.jobs.request_cancel(job.id)
    assert client.post("/api/songs/s1/trash").status_code == 200
