"""Songs router: list, detail, upload."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter(prefix="/api", tags=["songs"])


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
