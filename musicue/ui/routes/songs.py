"""Songs router: list, detail, upload, analyze, trash, thumbnail."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from musicue.index import index as indexer
from musicue.index import query as index_query
from musicue.ui import ingest
from musicue.ui.routes._validators import validate_song_id

router = APIRouter(prefix="/api", tags=["songs"])


async def _default_analyze(
    audio_path: Path,
    run_dir: Path,
    publish,
    *,
    job_id: str | None = None,
    pool=None,
) -> str:
    """Real analyze path. Submits run_analysis to the cancellable AnalyzePool
    when one is available; falls back to the default executor (thread) when
    no pool/job_id was provided — this keeps the test-mode analyze_func
    overrides simple."""
    from musicue.analysis.pipeline import compute_run_dir, run_analysis
    from musicue.config import MusiCueConfig

    cfg = MusiCueConfig()
    cfg.runs_dir = run_dir.parent

    await publish({"type": "progress", "fraction": 0.05, "stage": "starting"})
    await publish({"type": "progress", "fraction": 0.1, "stage": "analyzing"})
    if pool is not None and job_id is not None:
        # Route through AnalyzePool.submit so JobManager.cancel can actually
        # SIGTERM the worker process. asyncio.wrap_future converts the
        # concurrent.futures.Future the pool returned into an awaitable.
        fut = pool.submit(job_id, run_analysis, audio_path, cfg)
        await asyncio.wrap_future(fut)
    else:
        # Test / SSE-only path: stay in-process so coroutine fakes work.
        await asyncio.get_running_loop().run_in_executor(
            None, run_analysis, audio_path, cfg
        )
    await publish({"type": "progress", "fraction": 0.95, "stage": "writing"})
    actual_run_dir = compute_run_dir(audio_path, cfg)
    return actual_run_dir.name


@router.get("/songs")
def list_songs(
    request: Request,
    q: Annotated[str | None, Query()] = None,
    filter: Annotated[List[str] | None, Query()] = None,
    sort: Annotated[str, Query()] = "added_at",
    trashed: Annotated[int, Query()] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    db = request.app.state.index_db
    rows = index_query.list_songs(
        db,
        q=q,
        filters=filter or (),
        sort=sort,
        trashed=bool(trashed),
        limit=limit,
        offset=offset,
    )
    if rows:
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        analyses: dict[str, list[str]] = {sid: [] for sid in ids}
        for sid, aid in db.execute(
            f"SELECT song_id, id FROM analyses WHERE song_id IN ({placeholders}) "
            f"ORDER BY created_at DESC",
            ids,
        ):
            analyses[sid].append(aid)
        for r in rows:
            r["analysis_ids"] = analyses.get(r["id"], [])
            r["has_analysis"] = bool(r["analysis_ids"])
    return {"songs": rows}


@router.get("/songs/{song_id}")
def get_song(song_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
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
        "source_url": rec.source_url,
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
    indexer.refresh_song(
        request.app.state.index_db, request.app.state.storage_root, rec.id
    )
    return {"id": rec.id, "title": rec.title, "source_ext": rec.source_ext}


@router.post("/songs/{song_id}/analyze", status_code=202)
async def analyze_song(song_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
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

    pool = getattr(request.app.state, "pool", None)

    async def runner() -> None:
        try:
            if analyze_func is _default_analyze:
                analysis_id = await analyze_func(
                    rec.source_path, run_dir, publish, job_id=job.id, pool=pool
                )
            else:
                analysis_id = await analyze_func(rec.source_path, run_dir, publish)
            indexer.refresh_song(
                request.app.state.index_db,
                request.app.state.storage_root,
                song_id,
            )
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
    # Both checks (structural + SSRF guard) run synchronously here; the
    # SSRF guard does DNS, but it's bounded by the resolver's timeout.
    try:
        ingest._validate_url(body.url)
        ingest._validate_destination_safe(body.url)
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
            pool = getattr(request.app.state, "pool", None)
            if analyze_func is _default_analyze:
                analysis_id = await analyze_func(
                    rec.source_path,
                    run_dir,
                    rescaled_publish,
                    job_id=job.id,
                    pool=pool,
                )
            else:
                analysis_id = await analyze_func(
                    rec.source_path, run_dir, rescaled_publish
                )
            indexer.refresh_song(
                request.app.state.index_db,
                request.app.state.storage_root,
                rec.id,
            )
            await jobs.complete(
                job.id,
                result={"song_id": rec.id, "analysis_id": analysis_id},
            )
        except Exception as exc:  # noqa: BLE001
            await jobs.fail(job.id, error=f"{type(exc).__name__}: {exc}")

    asyncio.create_task(runner())
    return {"job_id": job.id, "queue": "ingest", "status": "queued"}


@router.get("/songs/{song_id}/thumbnail")
def get_thumbnail(song_id: str, request: Request) -> FileResponse:
    song_id = validate_song_id(song_id)
    storage = request.app.state.storage
    p = storage.song_dir(song_id) / "thumbnail.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(
        p,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@router.post("/songs/{song_id}/trash")
def trash_song(song_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
    db = request.app.state.index_db
    root = request.app.state.storage_root
    jobs = request.app.state.jobs
    if jobs.list_for_song(song_id):
        raise HTTPException(
            status_code=409, detail="job in progress for this song"
        )
    try:
        ts = indexer.set_trashed(db, root, song_id, trashed=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="song not found")
    return {"id": song_id, "trashed_at": ts}


@router.post("/songs/{song_id}/untrash")
def untrash_song(song_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
    db = request.app.state.index_db
    root = request.app.state.storage_root
    try:
        indexer.set_trashed(db, root, song_id, trashed=False)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="song not found")
    return {"id": song_id, "trashed_at": None}


@router.delete("/songs/{song_id}")
def delete_song(song_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
    db = request.app.state.index_db
    root = request.app.state.storage_root
    try:
        indexer.delete_song(db, root, song_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="song not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"id": song_id, "deleted": True}
