"""Route: POST /api/songs/{song_id}/analyses/{analysis_id}/send-to-cedartoy.

Composes the existing analysis + a freshly-compiled cuesheet into a
portable CedarToy project folder on the server's filesystem. Same code
path the CLI uses; see musicue/compile/cedartoy_folder.py.
"""
from __future__ import annotations

from importlib.metadata import version as _pkg_version
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from musicue.compile.cedartoy_folder import build_cedartoy_folder
from musicue.compile.compiler import compile_analysis
from musicue.schemas import AnalysisResult
from musicue.ui.routes._validators import (
    validate_analysis_id,
    validate_song_id,
)

router = APIRouter(
    prefix="/api/songs/{song_id}/analyses/{analysis_id}",
    tags=["cedartoy"],
)

_GRAMMARS = (
    "concert_visuals",
    "character_animation",
    "lighting",
    "camera_edit",
)


class SendToCedarToyRequest(BaseModel):
    output_folder: str = Field(..., description="Server-local folder path to create.")
    grammar: str = Field("concert_visuals")
    include_stems: bool = False
    force_analyze: bool = Field(
        False,
        description="If true, re-run the analysis pipeline ignoring cache. "
                    "Blocks the request for the duration of analysis (~2 min). "
                    "Pre-existing analysis on disk is overwritten.",
    )


@router.post("/send-to-cedartoy")
def send_to_cedartoy(
    song_id: str,
    analysis_id: str,
    body: SendToCedarToyRequest,
    request: Request,
) -> dict:
    song_id = validate_song_id(song_id)
    analysis_id = validate_analysis_id(analysis_id)

    if body.grammar not in _GRAMMARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown grammar '{body.grammar}'. "
                f"Available: {', '.join(_GRAMMARS)}"
            ),
        )

    storage = request.app.state.storage
    song = storage.get_song(song_id)
    if song is None:
        raise HTTPException(status_code=404, detail="song not found")
    analysis_path = storage.analysis_dir(song_id, analysis_id) / "analysis.json"

    if body.force_analyze:
        # Re-run analysis synchronously. Blocks the request. Same pattern as
        # the /click endpoint, which also runs a multi-second pipeline inside
        # the request handler. Async-job delivery is a deferred improvement.
        from musicue.analysis.pipeline import run_analysis
        from musicue.config import MusiCueConfig
        cfg = MusiCueConfig()
        result = run_analysis(song.source_path, cfg)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
    elif not analysis_path.exists():
        raise HTTPException(status_code=404, detail="analysis not found")

    out_dir = Path(body.output_folder)
    if out_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"target folder already exists: {out_dir}",
        )

    try:
        analysis = AnalysisResult.model_validate_json(
            analysis_path.read_text(encoding="utf-8")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"analysis parse error: {e}"
        ) from e

    try:
        cuesheet = compile_analysis(analysis, grammar=body.grammar)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"compile failed: {e}"
        ) from e

    stems_src = storage.analysis_dir(song_id, analysis_id) / "stems"
    try:
        mc_ver = _pkg_version("musicue")
    except Exception:
        mc_ver = "0.0.0+dev"

    try:
        manifest = build_cedartoy_folder(
            audio_path=song.source_path,
            analysis=analysis,
            cuesheet=cuesheet,
            out_dir=out_dir,
            grammar=body.grammar,
            musicue_version=mc_ver,
            include_stems=body.include_stems,
            stems_src_dir=stems_src if body.include_stems else None,
            original_audio_name=song.source_path.name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"folder build failed: {e}"
        ) from e

    return {
        "ok": True,
        "output_folder": str(out_dir),
        "stems_included": (
            manifest.stems_omitted_reason is None and body.include_stems
        ),
        "stems_omitted_reason": manifest.stems_omitted_reason,
    }
