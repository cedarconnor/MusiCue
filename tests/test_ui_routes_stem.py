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


def _seed_demucs_layout(
    storage_root: Path,
    song_id: str,
    analysis_id: str,
    stem: str,
    model: str = "htdemucs_ft",
    audio_stem: str = "source",
) -> Path:
    """Seed the actual demucs output layout: stems/<model>/<audio_stem>/<stem>.wav."""
    sd = storage_root / "songs" / song_id
    target_dir = sd / "analyses" / analysis_id / "stems" / model / audio_stem
    target_dir.mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")
    (sd / "analyses" / analysis_id / "analysis.json").write_text("{}")
    p = target_dir / f"{stem}.wav"
    p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEdrums")
    return p


def _seed_flat_layout(
    storage_root: Path, song_id: str, analysis_id: str, stem: str
) -> Path:
    """Seed the flat fallback layout: stems/<stem>.wav."""
    sd = storage_root / "songs" / song_id
    (sd / "analyses" / analysis_id / "stems").mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")
    (sd / "analyses" / analysis_id / "analysis.json").write_text("{}")
    p = sd / "analyses" / analysis_id / "stems" / f"{stem}.wav"
    p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEflat")
    return p


def test_get_stem_finds_demucs_nested_layout(
    tmp_path: Path, client: TestClient
) -> None:
    sid = "1" * 64
    aid = "a" * 12
    _seed_demucs_layout(tmp_path, sid, aid, "drums")

    r = client.get(f"/api/songs/{sid}/analyses/{aid}/stems/drums")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert "max-age" in r.headers.get("cache-control", "")
    assert r.content.startswith(b"RIFF")


def test_get_stem_falls_back_to_flat_layout(
    tmp_path: Path, client: TestClient
) -> None:
    sid = "2" * 64
    aid = "b" * 12
    _seed_flat_layout(tmp_path, sid, aid, "bass")

    r = client.get(f"/api/songs/{sid}/analyses/{aid}/stems/bass")
    assert r.status_code == 200
    assert r.content.endswith(b"flat")


def test_get_stem_404_when_missing(client: TestClient) -> None:
    r = client.get(
        f"/api/songs/{'9' * 64}/analyses/{'9' * 12}/stems/drums"
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "stem not generated"


def test_get_stem_400_for_unknown_stem(client: TestClient) -> None:
    r = client.get(
        f"/api/songs/{'1' * 64}/analyses/{'1' * 12}/stems/banjo"
    )
    assert r.status_code == 400
    assert "unknown stem" in r.json()["detail"]


def test_get_stem_400_for_bad_song_id(client: TestClient) -> None:
    r = client.get("/api/songs/bad-id/analyses/" + "1" * 12 + "/stems/drums")
    assert r.status_code == 400


def test_get_stem_400_for_bad_analysis_id(client: TestClient) -> None:
    r = client.get(f"/api/songs/{'1' * 64}/analyses/short/stems/drums")
    assert r.status_code == 400
