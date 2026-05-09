"""Analyses router: GET analysis JSON, GET peaks per stem, GET source audio."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/songs/{song_id}", tags=["analyses"])


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
