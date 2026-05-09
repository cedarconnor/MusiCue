"""Click-track endpoint: generate (cached) and serve the QC click WAV."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/songs/{song_id}/analyses/{analysis_id}",
                   tags=["click"])


def _real_render(source: Path, analysis_path: Path, out_path: Path) -> None:
    """Compile a default cuesheet from analysis and render the click track."""
    from musicue.compile.compiler import compile_analysis
    from musicue.listen import render_click_track
    from musicue.schemas import AnalysisResult

    analysis = AnalysisResult.model_validate_json(
        analysis_path.read_text(encoding="utf-8")
    )
    cuesheet = compile_analysis(analysis, grammar="concert_visuals")
    render_click_track(cuesheet, source, out_path)


@router.post("/click")
def render_click(song_id: str, analysis_id: str, request: Request) -> dict:
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
