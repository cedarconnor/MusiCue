"""Thumbnail route: 200 when present, 404 when absent."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

SID = "1" * 64


def _seed(tmp_path: Path, sid: str, with_thumb: bool) -> None:
    d = tmp_path / "songs" / sid
    d.mkdir(parents=True)
    (d / "title.txt").write_text("t")
    (d / "source.wav").write_bytes(b"\0")
    if with_thumb:
        (d / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff")


def test_thumbnail_present(make_app, tmp_path: Path) -> None:
    _seed(tmp_path, SID, with_thumb=True)
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{SID}/thumbnail")
    assert r.status_code == 200
    assert r.headers["content-type"] in ("image/jpeg", "image/jpg")
    assert r.content == b"\xff\xd8\xff"


def test_thumbnail_missing(make_app, tmp_path: Path) -> None:
    _seed(tmp_path, SID, with_thumb=False)
    app = make_app(tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{SID}/thumbnail")
    assert r.status_code == 404
