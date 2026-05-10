"""GET /api/songs query parameters: search, filter, sort, trashed, paging."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def _seed(tmp_path: Path) -> None:
    songs = [
        ("s1", "a1", "Alpha", "https://x/a", {"bpm_global": 100.0}),
        (
            "s2",
            "a2",
            "Bravo Tango",
            None,
            {
                "bpm_global": 130.0,
                "onsets": {
                    "drums": [
                        {"t": 0.0, "drum_class": "kick", "labels": ["kick"]}
                    ]
                },
            },
        ),
        ("s3", "a3", "Charlie", "https://x/c", {"bpm_global": 160.0}),
    ]
    for sid, aid, title, url, extra in songs:
        d = tmp_path / "songs" / sid / "analyses" / aid
        d.mkdir(parents=True)
        (tmp_path / "songs" / sid / "title.txt").write_text(title)
        (tmp_path / "songs" / sid / "source.wav").write_bytes(b"\0")
        if url:
            (tmp_path / "songs" / sid / "source_url.txt").write_text(url)
        analysis = {
            "schema_version": "v3",
            "source": {"duration_sec": 60.0},
            "tempo": {"bpm_global": extra.get("bpm_global", 0.0)},
        }
        if "onsets" in extra:
            analysis["onsets"] = extra["onsets"]
        (d / "analysis.json").write_text(json.dumps(analysis))
        (d / "stems").mkdir()


def _refresh(app, tmp_path: Path) -> None:
    from musicue.index import index as indexer

    indexer.rebuild(app.state.index_db, tmp_path)


def test_default_returns_all_non_trashed(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs")
    assert r.status_code == 200
    body = r.json()
    ids = sorted(s["id"] for s in body["songs"])
    assert ids == ["s1", "s2", "s3"]


def test_q_filters_by_title(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs?q=brav")
    assert [s["id"] for s in r.json()["songs"]] == ["s2"]


def test_filter_chip_has_clap(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs?filter=has_clap")
    assert [s["id"] for s in r.json()["songs"]] == ["s2"]


def test_sort_title(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs?sort=title")
    assert [s["title"] for s in r.json()["songs"]] == [
        "Alpha",
        "Bravo Tango",
        "Charlie",
    ]


def test_trashed_flag(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    from musicue.index import index as indexer

    indexer.set_trashed(app.state.index_db, tmp_path, "s2", trashed=True)
    client = TestClient(app)
    assert {s["id"] for s in client.get("/api/songs").json()["songs"]} == {
        "s1",
        "s3",
    }
    assert {
        s["id"] for s in client.get("/api/songs?trashed=1").json()["songs"]
    } == {"s2"}


def test_response_includes_analysis_ids(make_app, tmp_path: Path) -> None:
    _seed(tmp_path)
    app = make_app(tmp_path)
    _refresh(app, tmp_path)
    client = TestClient(app)
    body = client.get("/api/songs").json()
    by_id = {s["id"]: s for s in body["songs"]}
    assert by_id["s1"]["analysis_ids"] == ["a1"]
    assert by_id["s1"]["has_analysis"] is True
