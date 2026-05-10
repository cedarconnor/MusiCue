"""Analyses router: GET analysis JSON, GET peaks per stem, GET source audio, loop persistence."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from musicue.index import index as indexer
from musicue.index import query as index_query

router = APIRouter(prefix="/api/songs/{song_id}", tags=["analyses"])


class LoopBody(BaseModel):
    loop_in: float = Field(ge=0)
    loop_out: float = Field(ge=0)
    enabled: bool = True


@router.get("/analyses/{analysis_id}")
def get_analysis(song_id: str, analysis_id: str, request: Request) -> dict:
    storage = request.app.state.storage
    p = storage.analysis_dir(song_id, analysis_id) / "analysis.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="analysis not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/analyses/{analysis_id}/peaks/{stem}")
def get_peaks(song_id: str, analysis_id: str, stem: str, request: Request) -> dict:
    storage = request.app.state.storage
    p = storage.analysis_dir(song_id, analysis_id) / f"peaks.{stem}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="peaks not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/analyses/{analysis_id}/stems/{stem}")
def get_stem(
    song_id: str, analysis_id: str, stem: str, request: Request
) -> FileResponse:
    storage = request.app.state.storage
    stems_root = storage.analysis_dir(song_id, analysis_id) / "stems"
    # Demucs writes to <stems_root>/<model_name>/<audio_stem>/<stem>.wav.
    # Glob handles different models (htdemucs_ft, htdemucs_6s, ...) and
    # different source filenames without hard-coding either.
    matches = list(stems_root.glob(f"**/{stem}.wav"))
    if not matches:
        # Also accept the flat layout (analysis_dir/stems/<stem>.wav) so a
        # future run that flattens the demucs output keeps working.
        flat = stems_root / f"{stem}.wav"
        if flat.exists():
            matches = [flat]
    if not matches:
        raise HTTPException(status_code=404, detail="stem not generated")
    return FileResponse(
        matches[0],
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000"},
    )


@router.get("/analyses/{analysis_id}/loop", response_model=None)
def get_loop(song_id: str, analysis_id: str, request: Request):
    db = request.app.state.index_db
    loop = index_query.get_loop(db, song_id, analysis_id)
    if loop is None:
        return Response(status_code=204)
    return loop


@router.put("/analyses/{analysis_id}/loop")
def put_loop(
    song_id: str, analysis_id: str, body: LoopBody, request: Request
) -> dict:
    if body.loop_out <= body.loop_in:
        raise HTTPException(
            status_code=400, detail="loop_out must exceed loop_in"
        )
    db = request.app.state.index_db
    root = request.app.state.storage_root
    try:
        return indexer.set_loop(
            db, root, song_id, analysis_id, body.model_dump()
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="analysis not found")


@router.get("/source")
def get_source(song_id: str, request: Request) -> FileResponse:
    storage = request.app.state.storage
    rec = storage.get_song(song_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="song not found")
    media_type = {
        "wav": "audio/wav",
        "flac": "audio/flac",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
    }.get(rec.source_ext, f"audio/{rec.source_ext}")
    return FileResponse(
        rec.source_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000"},
    )
