import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def _seed(tmp_path: Path) -> tuple[str, str]:
    song_id = "a" * 64
    analysis_id = "f" * 12
    base = tmp_path / "songs" / song_id
    (base / "analyses" / analysis_id).mkdir(parents=True)
    (base / "title.txt").write_text("Test", encoding="utf-8")
    sr = 44100
    sf.write(str(base / "source.wav"),
             np.zeros(sr // 2, dtype=np.float32), sr)
    (base / "analyses" / analysis_id / "analysis.json").write_text(
        json.dumps({"version": 1, "tempo": {"bpm": 128}})
    )
    (base / "analyses" / analysis_id / "peaks.mix.json").write_text(
        json.dumps({"version": 2, "data": [-0.5, 0.5, -0.6, 0.6]})
    )
    return song_id, analysis_id


def test_get_analysis(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{song_id}/analyses/{analysis_id}")
    assert r.status_code == 200
    assert r.json()["tempo"]["bpm"] == 128


def test_get_peaks(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{song_id}/analyses/{analysis_id}/peaks/mix")
    assert r.status_code == 200
    assert r.json()["data"] == [-0.5, 0.5, -0.6, 0.6]


def test_get_peaks_unknown_stem_404(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{song_id}/analyses/{analysis_id}/peaks/vocals")
    assert r.status_code == 404


def test_get_source_audio(tmp_path):
    song_id, analysis_id = _seed(tmp_path)
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get(f"/api/songs/{song_id}/source")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
