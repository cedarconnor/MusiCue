"""Click-track endpoints: master click WAV + per-stem click marks."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from musicue.ui.routes._validators import (
    validate_analysis_id,
    validate_song_id,
    validate_stem,
)

router = APIRouter(prefix="/api/songs/{song_id}/analyses/{analysis_id}",
                   tags=["click"])


def _real_render(source: Path, analysis_path: Path, out_path: Path) -> None:
    """Compile a default cuesheet from analysis, render the click track,
    and also emit per-stem clicks-only WAVs alongside it."""
    from musicue.compile.compiler import compile_analysis
    from musicue.listen import render_click_track, render_stem_click_marks
    from musicue.schemas import AnalysisResult

    analysis = AnalysisResult.model_validate_json(
        analysis_path.read_text(encoding="utf-8")
    )
    cuesheet = compile_analysis(analysis, grammar="concert_visuals")
    render_click_track(cuesheet, source, out_path)
    render_stem_click_marks(cuesheet, out_path.parent)


@router.post("/click")
def render_click(song_id: str, analysis_id: str, request: Request) -> dict:
    song_id = validate_song_id(song_id)
    analysis_id = validate_analysis_id(analysis_id)
    storage = request.app.state.storage
    rec = storage.get_song(song_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="song not found")
    analysis_dir = storage.analysis_dir(song_id, analysis_id)
    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail="analysis not found")
    out_path = analysis_dir / "click.wav"
    if not out_path.exists():
        render_func = getattr(request.app.state, "click_render", _real_render)
        render_func(rec.source_path, analysis_dir / "analysis.json", out_path)
    return {"ready": True, "size_bytes": out_path.stat().st_size}


@router.get("/click.wav")
def get_click_wav(song_id: str, analysis_id: str, request: Request) -> FileResponse:
    song_id = validate_song_id(song_id)
    analysis_id = validate_analysis_id(analysis_id)
    storage = request.app.state.storage
    out_path = storage.analysis_dir(song_id, analysis_id) / "click.wav"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="click not generated")
    # No-cache because the click WAV gets regenerated when the server
    # rendering code changes; we don't want browsers serving stale audio.
    return FileResponse(
        out_path,
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/click_marks/{stem}")
def get_click_marks(
    song_id: str, analysis_id: str, stem: str, request: Request,
) -> FileResponse:
    """Per-stem clicks-only WAV, layered with the stem WAV in the UI
    when a stem is soloed with the click track on."""
    song_id = validate_song_id(song_id)
    analysis_id = validate_analysis_id(analysis_id)
    stem = validate_stem(stem)
    storage = request.app.state.storage
    out_path = (
        storage.analysis_dir(song_id, analysis_id) / f"click_marks.{stem}.wav"
    )
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="click marks not generated")
    return FileResponse(
        out_path,
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
