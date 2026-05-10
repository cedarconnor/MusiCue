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


def test_jobs_sse_streams_progress(tmp_path):
    """v0.1b: WS endpoint dropped in favour of SSE-only.

    Same payload contract; the previous test exercised the WS path that
    no longer exists. Migrated to /events.
    """
    import json

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
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            evt = json.loads(line[len("data:"):].strip())
            received.append(evt)
            if evt.get("type") in ("complete", "error"):
                break

    types = [e["type"] for e in received]
    assert "complete" in types
    complete = [e for e in received if e["type"] == "complete"][0]
    assert complete["result"]["analysis_id"] == "fake-analysis-id"


def test_cancel_running_analyze_emits_cancelled_event(tmp_path):
    """POST /cancel on a running job: the SSE stream sees a terminal event
    and closes, rather than hanging on the slow analyze."""
    import asyncio
    import json

    import httpx

    app = create_app(storage_root=tmp_path)

    started = asyncio.Event()

    async def slow_analyze(audio_path: Path, run_dir: Path, publish):
        await publish({"type": "progress", "fraction": 0.1, "stage": "starting"})
        started.set()
        for _ in range(50):
            await asyncio.sleep(0.05)
        return "never-reached"

    app.state.analyze_func = slow_analyze

    async def driver() -> str:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", timeout=10.0
        ) as client:
            r = await client.post(
                "/api/songs",
                files={"file": ("t.wav", _wav_bytes(), "audio/wav")},
                data={"title": "T"},
            )
            song_id = r.json()["id"]

            r = await client.post(f"/api/songs/{song_id}/analyze")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            await asyncio.wait_for(started.wait(), timeout=2.0)

            cancel_resp = await client.post(f"/api/jobs/{job_id}/cancel")
            assert cancel_resp.status_code == 200

            async with client.stream(
                "GET", f"/api/jobs/{job_id}/events"
            ) as resp:
                chunks: list[str] = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)
                return "".join(chunks)

    import time

    t0 = time.monotonic()
    body = asyncio.run(driver())
    elapsed = time.monotonic() - t0

    types = [
        json.loads(line[6:])["type"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    # Cancel hook should publish a synthetic terminal event so the SSE
    # stream closes promptly -- we shouldn't have to wait for slow_analyze's
    # natural completion (~2.5s).
    assert types[-1] == "cancelled"
    assert elapsed < 2.0, f"SSE stream took {elapsed:.2f}s to close after cancel"


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
