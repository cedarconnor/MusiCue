import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def _make_wav_bytes(sr: int = 44100, sec: float = 1.0) -> bytes:
    t = np.linspace(0, sec, int(sr * sec), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_list_songs_empty(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs")
    assert r.status_code == 200
    assert r.json() == {"songs": []}


def test_upload_then_list(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    wav = _make_wav_bytes()
    r = client.post(
        "/api/songs",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"title": "Test Song"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["title"] == "Test Song"
    assert "id" in body

    r = client.get("/api/songs")
    assert r.status_code == 200
    assert len(r.json()["songs"]) == 1

    r = client.get(f"/api/songs/{body['id']}")
    assert r.status_code == 200
    assert r.json()["title"] == "Test Song"


def test_get_unknown_song_404(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs/deadbeef")
    assert r.status_code == 404


def test_upload_default_title_is_filename_stem(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    wav = _make_wav_bytes()
    r = client.post(
        "/api/songs",
        files={"file": ("My Track.wav", wav, "audio/wav")},
    )
    assert r.status_code == 201
    assert r.json()["title"] == "My Track"
