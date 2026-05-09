"""Per-stem audio route for v0.1a Multitrack lanes."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(storage_root=tmp_path)
    return TestClient(app)


def _seed_song_with_stem(
    storage_root: Path, song_id: str, analysis_id: str, stem: str
) -> Path:
    sd = storage_root / "songs" / song_id
    (sd / "analyses" / analysis_id / "stems").mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")
    (sd / "analyses" / analysis_id / "analysis.json").write_text("{}")
    p = sd / "analyses" / analysis_id / "stems" / f"{stem}.wav"
    p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEdrums")
    return p


def test_get_stem_serves_wav_with_cache_control(
    tmp_path: Path, client: TestClient
) -> None:
    _seed_song_with_stem(tmp_path, "song1", "an1", "drums")

    r = client.get("/api/songs/song1/analyses/an1/stems/drums")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert "max-age" in r.headers.get("cache-control", "")
    assert r.content.startswith(b"RIFF")


def test_get_stem_404_when_missing(client: TestClient) -> None:
    r = client.get("/api/songs/nosong/analyses/noan/stems/drums")
    assert r.status_code == 404
    assert r.json()["detail"] == "stem not generated"
