"""SSE endpoint emits the same event shape as the WS stream.

Uses httpx.AsyncClient against ASGITransport so the SSE consumer and the
publisher both run on the same asyncio event loop -- ``asyncio.Queue`` is
loop-local, so cross-thread driver patterns hang.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from musicue.ui.server import create_app


def _parse_sse(body: str) -> list[dict]:
    out = []
    for chunk in body.strip().split("\n\n"):
        for line in chunk.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


@pytest.mark.asyncio
async def test_sse_unknown_job_returns_404(tmp_path: Path) -> None:
    app = create_app(storage_root=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get("/api/jobs/does-not-exist/events")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_sse_stream_replays_status_then_emits_progress_and_complete(
    tmp_path: Path,
) -> None:
    app = create_app(storage_root=tmp_path)
    jobs = app.state.jobs
    job = jobs.submit(kind="analyze", payload={"song_id": "s1"})

    transport = httpx.ASGITransport(app=app)

    async def driver() -> None:
        # Let the SSE consumer subscribe first.
        await asyncio.sleep(0.1)
        await jobs.publish(job.id, {"type": "progress", "fraction": 0.5})
        await jobs.complete(job.id, result={"analysis_id": "an1"})

    async def consumer() -> str:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", timeout=10.0
        ) as client:
            async with client.stream("GET", f"/api/jobs/{job.id}/events") as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                chunks: list[str] = []
                async for chunk in r.aiter_text():
                    chunks.append(chunk)
                return "".join(chunks)

    body, _ = await asyncio.gather(consumer(), driver())

    events = _parse_sse(body)
    types = [e.get("type") for e in events]
    assert types[0] == "status"  # replayed first
    assert "progress" in types
    assert "complete" in types
    assert types[-1] == "complete"
