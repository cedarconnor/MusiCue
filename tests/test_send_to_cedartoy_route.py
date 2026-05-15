"""HTTP tests for the send-to-cedartoy endpoint."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from musicue.ui.server import create_app
from tests.test_bundle_builder import (
    make_analysis_fixture,
    make_cuesheet_fixture,
)

# Must be 12 lowercase hex chars to pass musicue.ui.routes._validators.
ANALYSIS_ID = "abcdef012345"


def _seed_song(storage_root: Path, source_sha: str, audio_bytes_path: Path) -> str:
    """Drop a song + analysis into the on-disk storage layout."""
    song_dir = storage_root / "songs" / source_sha
    song_dir.mkdir(parents=True, exist_ok=True)
    target_audio = song_dir / "source.wav"
    shutil.copy2(audio_bytes_path, target_audio)
    (song_dir / "title.txt").write_text("Test Song", encoding="utf-8")

    analyses_dir = song_dir / "analyses" / ANALYSIS_ID
    analyses_dir.mkdir(parents=True, exist_ok=True)
    analysis = make_analysis_fixture(audio_path=target_audio)
    (analyses_dir / "analysis.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )
    return analysis.source.sha256


@pytest.fixture
def client(tmp_path):
    app = create_app(storage_root=tmp_path)
    return TestClient(app), tmp_path


def _seed_with_audio(tmp_path: Path, root: Path) -> str:
    audio = tmp_path / "seed.wav"
    sf.write(str(audio), np.zeros(11025, dtype="float32"), 44100, subtype="PCM_16")
    from musicue.ui.storage import sha256_of_file
    sha = sha256_of_file(audio)
    _seed_song(root, sha, audio)
    return sha


def test_send_to_cedartoy_writes_folder(client, tmp_path):
    c, root = client
    sha = _seed_with_audio(tmp_path, root)

    out = tmp_path / "exports" / "song"
    resp = c.post(
        f"/api/songs/{sha}/analyses/{ANALYSIS_ID}/send-to-cedartoy",
        json={
            "output_folder": str(out),
            "grammar": "concert_visuals",
            "include_stems": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output_folder"] == str(out)
    assert body["ok"] is True

    assert (out / "song.wav").exists()
    assert (out / "song.musicue.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["grammar"] == "concert_visuals"


def test_send_to_cedartoy_404_when_song_missing(client, tmp_path):
    c, _root = client
    fake_sha = "0" * 64
    resp = c.post(
        f"/api/songs/{fake_sha}/analyses/{ANALYSIS_ID}/send-to-cedartoy",
        json={
            "output_folder": str(tmp_path / "out"),
            "grammar": "concert_visuals",
        },
    )
    assert resp.status_code == 404


def test_send_to_cedartoy_400_when_grammar_invalid(client, tmp_path):
    c, root = client
    sha = _seed_with_audio(tmp_path, root)
    resp = c.post(
        f"/api/songs/{sha}/analyses/{ANALYSIS_ID}/send-to-cedartoy",
        json={
            "output_folder": str(tmp_path / "out"),
            "grammar": "bogus",
        },
    )
    assert resp.status_code == 400


def test_send_to_cedartoy_409_when_target_exists(client, tmp_path):
    c, root = client
    sha = _seed_with_audio(tmp_path, root)

    out = tmp_path / "out"
    out.mkdir()

    resp = c.post(
        f"/api/songs/{sha}/analyses/{ANALYSIS_ID}/send-to-cedartoy",
        json={
            "output_folder": str(out),
            "grammar": "concert_visuals",
        },
    )
    assert resp.status_code == 409


def test_send_to_cedartoy_with_stems(client, tmp_path):
    """End-to-end: cached stems on disk are copied into the output folder."""
    c, root = client
    sha = _seed_with_audio(tmp_path, root)

    # Drop stems into the analysis dir to mimic a cached Demucs run.
    stems_dir = root / "songs" / sha / "analyses" / ANALYSIS_ID / "stems"
    stems_dir.mkdir(parents=True)
    for name in ("drums", "bass", "vocals", "other"):
        sf.write(
            str(stems_dir / f"{name}.wav"),
            np.zeros(11025, dtype="float32"),
            44100,
            subtype="PCM_16",
        )

    out = tmp_path / "exports" / "song"
    resp = c.post(
        f"/api/songs/{sha}/analyses/{ANALYSIS_ID}/send-to-cedartoy",
        json={
            "output_folder": str(out),
            "grammar": "concert_visuals",
            "include_stems": True,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["stems_included"] is True

    for name in ("drums", "bass", "vocals", "other"):
        assert (out / "stems" / f"{name}.wav").exists()
