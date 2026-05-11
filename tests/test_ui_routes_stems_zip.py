"""Tests for GET /api/songs/<sid>/analyses/<aid>/stems.zip."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


SONG_ID = "1" * 64
ANALYSIS_ID = "a" * 12


def _plant_stems(storage_root: Path, *, layout: str = "nested") -> None:
    """Drop fake stem WAVs into the demucs-nested or flat layout."""
    sd = storage_root / "songs" / SONG_ID
    adir = sd / "analyses" / ANALYSIS_ID
    if layout == "nested":
        target = adir / "stems" / "htdemucs_ft" / "source"
    else:
        target = adir / "stems"
    target.mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"RIFFsource")
    (adir / "analysis.json").write_text("{}")
    for stem, marker in (
        ("drums", b"RIFFdrums"),
        ("bass", b"RIFFbass"),
        ("vocals", b"RIFFvocals"),
        ("other", b"RIFFother"),
    ):
        (target / f"{stem}.wav").write_bytes(marker)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(storage_root=tmp_path))


def test_zip_bundles_all_four_stems(tmp_path, client):
    _plant_stems(tmp_path)
    r = client.get(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/stems.zip"
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    cd = r.headers.get("content-disposition", "")
    assert "stems-" in cd and ".zip" in cd

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = sorted(zf.namelist())
    assert names == ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    # Spot-check one stem made the round trip.
    assert zf.read("drums.wav") == b"RIFFdrums"


def test_zip_works_with_flat_layout(tmp_path, client):
    _plant_stems(tmp_path, layout="flat")
    r = client.get(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/stems.zip"
    )
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "drums.wav" in zf.namelist()


def test_zip_404_when_no_stems(tmp_path, client):
    # Only seed the analysis dir, no stems/.
    sd = tmp_path / "songs" / SONG_ID
    (sd / "analyses" / ANALYSIS_ID).mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"")
    (sd / "analyses" / ANALYSIS_ID / "analysis.json").write_text("{}")
    r = client.get(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/stems.zip"
    )
    assert r.status_code == 404


def test_zip_partial_stems_still_returns(tmp_path, client):
    """If only drums and vocals exist on disk, zip those two and return 200."""
    sd = tmp_path / "songs" / SONG_ID
    adir = sd / "analyses" / ANALYSIS_ID
    target = adir / "stems" / "htdemucs_ft" / "source"
    target.mkdir(parents=True)
    (sd / "title.txt").write_text("seed", encoding="utf-8")
    (sd / "source.wav").write_bytes(b"")
    (adir / "analysis.json").write_text("{}")
    (target / "drums.wav").write_bytes(b"RIFFd")
    (target / "vocals.wav").write_bytes(b"RIFFv")

    r = client.get(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/stems.zip"
    )
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == ["drums.wav", "vocals.wav"]


def test_zip_bad_song_id_400(tmp_path, client):
    r = client.get(f"/api/songs/bad-id/analyses/{ANALYSIS_ID}/stems.zip")
    assert r.status_code == 400
