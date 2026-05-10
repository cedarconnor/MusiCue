from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def _seed(tmp_path: Path) -> tuple[str, str]:
    song_id = "b" * 64
    analysis_id = "f" * 12
    base = tmp_path / "songs" / song_id
    (base / "analyses" / analysis_id).mkdir(parents=True)
    (base / "title.txt").write_text("T", encoding="utf-8")
    sf.write(str(base / "source.wav"),
             np.zeros(44100, dtype=np.float32), 44100)
    (base / "analyses" / analysis_id / "analysis.json").write_text("{}")
    return song_id, analysis_id


def test_click_track_endpoint(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)

    def fake_render(source: Path, analysis_path: Path, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), np.zeros(44100, dtype=np.float32), 44100)

    app.state.click_render = fake_render
    client = TestClient(app)

    r = client.post(f"/api/songs/{song_id}/analyses/{analysis_id}/click")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    assert r.json()["size_bytes"] > 0

    r = client.get(f"/api/songs/{song_id}/analyses/{analysis_id}/click.wav")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")


def test_click_uses_cache_on_second_call(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    calls = {"n": 0}

    def fake_render(source: Path, analysis_path: Path, out_path: Path) -> None:
        calls["n"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), np.zeros(44100, dtype=np.float32), 44100)

    app.state.click_render = fake_render
    client = TestClient(app)
    client.post(f"/api/songs/{song_id}/analyses/{analysis_id}/click")
    client.post(f"/api/songs/{song_id}/analyses/{analysis_id}/click")
    assert calls["n"] == 1


def test_get_click_404_before_render(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{song_id}/analyses/{analysis_id}/click.wav")
    assert r.status_code == 404
