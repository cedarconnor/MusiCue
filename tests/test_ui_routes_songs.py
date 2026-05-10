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


def test_list_and_get_surface_source_url(tmp_path):
    """Frontend MetadataCard renders a hostname chip iff source_url is on
    the API payload. Was missing in the initial v0.1a routes; ext-review
    2026-05-09 caught that the URL chip never appeared."""
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    wav = _make_wav_bytes()
    r = client.post(
        "/api/songs",
        files={"file": ("u.wav", wav, "audio/wav")},
    )
    song_id = r.json()["id"]

    # Simulate the URL-ingest path's sidecar write. v0.1b reads the list
    # through the SQLite index, so we also have to ask the index to
    # re-ingest this song dir (the real /from_url runner calls
    # ``refresh_song`` for us after sidecars are written).
    sd = tmp_path / "songs" / song_id
    (sd / "source_url.txt").write_text(
        "https://www.youtube.com/watch?v=demo", encoding="utf-8"
    )
    from musicue.index import index as indexer

    indexer.refresh_song(app.state.index_db, app.state.storage_root, song_id)

    r = client.get(f"/api/songs/{song_id}")
    assert r.status_code == 200
    assert r.json()["source_url"] == "https://www.youtube.com/watch?v=demo"

    r = client.get("/api/songs")
    songs = r.json()["songs"]
    assert songs[0]["source_url"] == "https://www.youtube.com/watch?v=demo"


def test_list_source_url_is_null_for_file_uploads(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    wav = _make_wav_bytes()
    client.post("/api/songs", files={"file": ("u.wav", wav, "audio/wav")})

    r = client.get("/api/songs")
    assert r.json()["songs"][0]["source_url"] is None
