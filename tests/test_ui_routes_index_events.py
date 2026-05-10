"""Index banner SSE + empty-trash route."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_empty_trash_returns_aggregate(make_app, tmp_path: Path) -> None:
    app = make_app(tmp_path)
    client = TestClient(app)
    from musicue.index import index as indexer

    db = app.state.index_db
    for sid, aid in (("s1", "a1"), ("s2", "a2")):
        d = tmp_path / "songs" / sid / "analyses" / aid
        d.mkdir(parents=True)
        (tmp_path / "songs" / sid / "title.txt").write_text("t")
        (tmp_path / "songs" / sid / "source.wav").write_bytes(b"\0")
        (d / "analysis.json").write_text(
            '{"schema_version":"v3","source":{},"tempo":{}}'
        )
    indexer.rebuild(db, tmp_path)
    indexer.set_trashed(db, tmp_path, "s1", trashed=True)
    indexer.set_trashed(db, tmp_path, "s2", trashed=True)

    r = client.post("/api/library/empty-trash")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == 2 and body["skipped"] == []
    assert not (tmp_path / "songs" / "s1").exists()


def test_index_events_emits_idle_when_current(make_app, tmp_path: Path) -> None:
    app = make_app(tmp_path)
    client = TestClient(app)
    with client.stream("GET", "/api/library/index_events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in r.iter_lines():
            if line and line.startswith("data:"):
                events.append(line)
            if len(events) >= 1:
                break
        assert any("idle" in e for e in events)
