import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def _wav_bytes(sec: float = 0.5) -> bytes:
    sr = 44100
    audio = np.zeros(int(sr * sec), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def fake_run_analysis(audio_path: Path, run_dir: Path, publish):
    await publish({"type": "progress", "fraction": 0.5, "stage": "halfway"})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis.json").write_text('{"version": 1}')
    (run_dir / "peaks.mix.json").write_text('{"version": 2, "data": []}')
    return "fake-analysis-id"


def test_analyze_endpoint_returns_job_id(tmp_path):
    app = create_app(storage_root=tmp_path)
    app.state.analyze_func = fake_run_analysis
    client = TestClient(app)
    r = client.post(
        "/api/songs",
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"title": "T"},
    )
    song_id = r.json()["id"]

    r = client.post(f"/api/songs/{song_id}/analyze")
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["queue"] == "analyze"


def test_analyze_unknown_song_404(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.post("/api/songs/deadbeef/analyze")
    assert r.status_code == 404


def test_jobs_websocket_streams_progress(tmp_path):
    app = create_app(storage_root=tmp_path)
    app.state.analyze_func = fake_run_analysis
    client = TestClient(app)
    r = client.post(
        "/api/songs",
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"title": "T"},
    )
    song_id = r.json()["id"]
    r = client.post(f"/api/songs/{song_id}/analyze")
    job_id = r.json()["job_id"]

    received: list[dict] = []
    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
        for _ in range(10):
            try:
                evt = ws.receive_json()
            except Exception:
                break
            received.append(evt)
            if evt.get("type") in ("complete", "error"):
                break

    types = [e["type"] for e in received]
    assert "complete" in types
    complete = [e for e in received if e["type"] == "complete"][0]
    assert complete["result"]["analysis_id"] == "fake-analysis-id"


def test_jobs_snapshot_endpoint(tmp_path):
    app = create_app(storage_root=tmp_path)
    app.state.analyze_func = fake_run_analysis
    client = TestClient(app)
    r = client.post(
        "/api/songs",
        files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
        data={"title": "T"},
    )
    song_id = r.json()["id"]
    r = client.post(f"/api/songs/{song_id}/analyze")
    job_id = r.json()["job_id"]

    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == job_id
    assert body["kind"] == "analyze"
