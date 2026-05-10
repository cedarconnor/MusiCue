"""Trash + untrash + hard-delete + job-in-flight 409."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

SID = "1" * 64
AID = "a" * 12


def _seed(tmp_path: Path, sid: str = SID, aid: str = AID) -> None:
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
    assert client.post(f"/api/songs/{SID}/trash").status_code == 200
    assert (tmp_path / "songs" / SID / ".trashed_at").exists()
    assert {s["id"] for s in client.get("/api/songs").json()["songs"]} == set()
    assert client.post(f"/api/songs/{SID}/untrash").status_code == 200
    assert not (tmp_path / "songs" / SID / ".trashed_at").exists()


def test_hard_delete_requires_trash(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh_index(app, tmp_path)
    client = TestClient(app)
    assert client.delete(f"/api/songs/{SID}").status_code == 409
    client.post(f"/api/songs/{SID}/trash")
    assert client.delete(f"/api/songs/{SID}").status_code == 200
    assert not (tmp_path / "songs" / SID).exists()


def test_trash_blocks_when_job_in_flight(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh_index(app, tmp_path)
    job = app.state.jobs.submit(kind="analyze", payload={"song_id": SID})
    client = TestClient(app)
    r = client.post(f"/api/songs/{SID}/trash")
    assert r.status_code == 409
    assert "job" in r.json()["detail"].lower()
    app.state.jobs.request_cancel(job.id)
    assert client.post(f"/api/songs/{SID}/trash").status_code == 200
