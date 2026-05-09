"""Songs router: list, detail, upload, analyze."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from musicue.ui import ingest

router = APIRouter(prefix="/api", tags=["songs"])


async def _default_analyze(audio_path: Path, run_dir: Path, publish) -> str:
    """Real analyze path. Runs the synchronous run_analysis in a thread,
    publishes coarse progress at stage boundaries, returns the analysis id
    (the deterministic run_dir name from compute_run_dir)."""
    from musicue.analysis.pipeline import compute_run_dir, run_analysis
    from musicue.config import MusiCueConfig

    cfg = MusiCueConfig()
    cfg.runs_dir = run_dir.parent

    await publish({"type": "progress", "fraction": 0.05, "stage": "starting"})
    await publish({"type": "progress", "fraction": 0.1, "stage": "analyzing"})
    await asyncio.get_running_loop().run_in_executor(None, run_analysis, audio_path, cfg)
    await publish({"type": "progress", "fraction": 0.95, "stage": "writing"})
    actual_run_dir = compute_run_dir(audio_path, cfg)
    return actual_run_dir.name


@router.get("/songs")
def list_songs(request: Request) -> dict:
    storage = request.app.state.storage
    songs = storage.list_songs()
    return {
        "songs": [
            {
                "id": s.id,
                "title": s.title,
                "has_analysis": s.has_analysis,
                "analysis_ids": s.analysis_ids,
            }
            for s in songs
        ]
    }


@router.get("/songs/{song_id}")
def get_song(song_id: str, request: Request) -> dict:
    storage = request.app.state.storage
    rec = storage.get_song(song_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="song not found")
    summaries = {s.id: s for s in storage.list_songs()}
    summary = summaries.get(song_id)
    return {
        "id": rec.id,
        "title": rec.title,
        "source_ext": rec.source_ext,
        "has_analysis": summary.has_analysis if summary else False,
        "analysis_ids": summary.analysis_ids if summary else [],
    }


@router.post("/songs", status_code=201)
async def upload_song(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")
    storage = request.app.state.storage
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        rec = storage.register_source(
            tmp_path, title=title or Path(file.filename).stem
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"id": rec.id, "title": rec.title, "source_ext": rec.source_ext}


@router.post("/songs/{song_id}/analyze", status_code=202)
async def analyze_song(song_id: str, request: Request) -> dict:
    storage = request.app.state.storage
    rec = storage.get_song(song_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="song not found")
    jobs = request.app.state.jobs
    analyze_func = getattr(request.app.state, "analyze_func", _default_analyze)
    job = jobs.submit(kind="analyze", payload={"song_id": song_id})
    run_dir = storage.analyses_dir(song_id) / "pending"

    async def publish(event: dict) -> None:
        await jobs.publish(job.id, event)

    async def runner() -> None:
        try:
            analysis_id = await analyze_func(rec.source_path, run_dir, publish)
            await jobs.complete(job.id, result={"analysis_id": analysis_id})
        except Exception as exc:  # noqa: BLE001
            await jobs.fail(job.id, error=f"{type(exc).__name__}: {exc}")

    asyncio.create_task(runner())
    return {"job_id": job.id, "queue": "analyze", "status": "queued"}


class FromUrlRequest(BaseModel):
    url: str


@router.post("/songs/from_url", status_code=202)
async def songs_from_url(request: Request, body: FromUrlRequest) -> dict:
    storage = request.app.state.storage
    jobs = request.app.state.jobs
    analyze_func = getattr(request.app.state, "analyze_func", _default_analyze)

    # Pre-validate so callers get a 400 instead of an async error event.
    try:
        ingest._validate_url(body.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = jobs.submit(kind="ingest", payload={"url": body.url})

    async def publish(event: dict) -> None:
        await jobs.publish(job.id, event)

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        try:
            with tempfile.TemporaryDirectory(prefix="musicue-ingest-") as td:
                # Stage 1: download (0..0.3).
                def dl_progress(f: float) -> None:
                    asyncio.run_coroutine_threadsafe(
                        publish(
                            {
                                "type": "progress",
                                "stage": "downloading",
                                "fraction": f * 0.3,
                            }
                        ),
                        loop,
                    )

                track = await asyncio.to_thread(
                    ingest.download_url,
                    body.url,
                    Path(td),
                    dl_progress,
                )

                rec = storage.register_source(track.audio_path, title=track.title)
                if track.source_url:
                    (storage.song_dir(rec.id) / "source_url.txt").write_text(
                        track.source_url, encoding="utf-8"
                    )
                if track.thumbnail_url:
                    await asyncio.to_thread(
                        ingest._download_thumbnail,
                        track.thumbnail_url,
                        storage.song_dir(rec.id) / "thumbnail.jpg",
                    )

            # Stage 2: analyze (0.3..1.0). Reuse the existing analyze_func
            # contract; rescale fractions in a wrapping publish.
            async def rescaled_publish(event: dict) -> None:
                if event.get("type") == "progress" and "fraction" in event:
                    f = float(event["fraction"])
                    event = {**event, "fraction": 0.3 + 0.7 * f}
                await publish(event)

            run_dir = storage.analyses_dir(rec.id) / "pending"
            analysis_id = await analyze_func(
                rec.source_path, run_dir, rescaled_publish
            )
            await jobs.complete(
                job.id,
                result={"song_id": rec.id, "analysis_id": analysis_id},
            )
        except Exception as exc:  # noqa: BLE001
            await jobs.fail(job.id, error=f"{type(exc).__name__}: {exc}")

    asyncio.create_task(runner())
    return {"job_id": job.id, "queue": "ingest", "status": "queued"}
