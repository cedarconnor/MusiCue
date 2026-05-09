"""URL-ingest route: validates URL, runs ingest, kicks analyze."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from musicue.ui import ingest as ingest_mod
from musicue.ui.ingest import DownloadedTrack
from musicue.ui.server import create_app


def _seed_fake_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(url: str, dest_dir: Path, progress_cb=None) -> DownloadedTrack:
        if progress_cb is not None:
            progress_cb(0.5)
            progress_cb(1.0)
        wav = dest_dir / "abc123.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")
        return DownloadedTrack(
            audio_path=wav,
            title="Test Title",
            thumbnail_url=None,
            duration_sec=180.0,
            source_url=url,
        )

    monkeypatch.setattr(ingest_mod, "download_url", fake_download)


@pytest.mark.asyncio
async def test_from_url_rejects_unsupported_scheme(tmp_path: Path) -> None:
    app = create_app(storage_root=tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.post(
            "/api/songs/from_url", json={"url": "file:///etc/passwd"}
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_from_url_happy_path_returns_job_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_fake_ingest(monkeypatch)

    async def fake_analyze(audio_path: Path, run_dir: Path, publish) -> str:
        await publish({"type": "progress", "fraction": 0.5, "stage": "analyzing"})
        analysis_id = "an1"
        out = run_dir.parent / analysis_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text("{}", encoding="utf-8")
        return analysis_id

    app = create_app(storage_root=tmp_path)
    app.state.analyze_func = fake_analyze
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", timeout=10.0
    ) as client:
        r = await client.post(
            "/api/songs/from_url",
            json={"url": "https://www.youtube.com/watch?v=abc123"},
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        # Drain SSE stream.
        async with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
            chunks: list[str] = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
            body = "".join(chunks)

    events = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    types = [e["type"] for e in events]
    assert "progress" in types
    assert types[-1] == "complete"

    completion = next(e for e in events if e["type"] == "complete")
    result = completion["result"]
    assert "song_id" in result
    assert "analysis_id" in result

    # Source URL persisted on disk.
    sd = tmp_path / "songs" / result["song_id"]
    assert (sd / "source_url.txt").read_text(encoding="utf-8").startswith("https://")


@pytest.mark.asyncio
async def test_from_url_yt_dlp_failure_emits_error_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(url: str, dest_dir: Path, progress_cb=None) -> DownloadedTrack:
        raise RuntimeError("yt-dlp: video unavailable")

    monkeypatch.setattr(ingest_mod, "download_url", boom)

    app = create_app(storage_root=tmp_path)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", timeout=10.0
    ) as client:
        r = await client.post(
            "/api/songs/from_url", json={"url": "https://www.youtube.com/watch?v=x"}
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        async with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
            chunks: list[str] = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
            body = "".join(chunks)

    types = [
        json.loads(line[6:])["type"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert types[-1] == "error"
