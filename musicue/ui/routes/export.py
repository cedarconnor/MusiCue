"""Export router: POST compiles + exports a cuesheet and streams the file back.

Mirrors the CLI `compile` + `export` pipeline (see musicue/cli.py) so a
single POST does both layers and returns the artifact as a download.
"""
from __future__ import annotations

import importlib
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from musicue.compile.compiler import compile_analysis
from musicue.schemas import AnalysisResult

router = APIRouter(prefix="/api/songs/{song_id}", tags=["export"])

# Mirrors musicue.cli._EXPORTERS. Kept duplicated rather than imported because
# musicue.cli pulls typer at import-time and we don't want the UI server to
# inherit that dependency just to read a dict.
_EXPORTERS: dict[str, tuple[str, str]] = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
    "midi": ("musicue.exporters.midi", ".mid"),
    "after_effects": ("musicue.exporters.aftereffects", ".jsx"),
    "touchdesigner": ("musicue.exporters.touchdesigner", ".csv"),
    "osc": ("musicue.exporters.osc", "_osc.json"),
    "houdini": ("musicue.exporters.houdini", "_houdini.csv"),
    "disguise": ("musicue.exporters.disguise", "_disguise.csv"),
    "unreal": ("musicue.exporters.unreal", "_unreal.json"),
}

_GRAMMARS: tuple[str, ...] = (
    "concert_visuals", "character_animation", "lighting", "camera_edit",
)

# Filename sanitization: strip path separators and anything path-traversal-ish.
# Cap to 200 chars to stay under most filesystems' limits with room for the ext.
_FNAME_BAD = re.compile(r"[\\/]|[\x00-\x1f]")


def _sanitize_filename(stem: str | None, default: str = "cuesheet") -> str:
    if not stem:
        return default
    cleaned = _FNAME_BAD.sub("_", stem).strip(" .")
    return cleaned[:200] if cleaned else default


class ExportRequest(BaseModel):
    format: str = Field(..., description="One of the registered exporters.")
    grammar: str = Field(..., description="One of the four built-in grammars.")
    filename: str | None = None
    fps: float | None = Field(default=None, gt=0, le=240)
    drop_frame: bool = False
    ticks_per_beat: int | None = Field(default=None, gt=0, le=10000)
    osc_host: str | None = None
    osc_port: int | None = Field(default=None, gt=0, lt=65536)


@router.post("/analyses/{analysis_id}/export")
def export_cuesheet(
    song_id: str,
    analysis_id: str,
    body: ExportRequest,
    request: Request,
) -> FileResponse:
    if body.format not in _EXPORTERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown format '{body.format}'. Available: {', '.join(_EXPORTERS)}",
        )
    if body.grammar not in _GRAMMARS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown grammar '{body.grammar}'. Available: {', '.join(_GRAMMARS)}",
        )

    storage = request.app.state.storage
    apath = storage.analysis_dir(song_id, analysis_id) / "analysis.json"
    if not apath.exists():
        raise HTTPException(status_code=404, detail="analysis not found")

    try:
        analysis = AnalysisResult.model_validate_json(apath.read_text(encoding="utf-8"))
    except Exception as e:  # malformed analysis.json on disk
        raise HTTPException(status_code=500, detail=f"analysis parse error: {e}") from e

    try:
        cuesheet = compile_analysis(
            analysis,
            grammar=body.grammar,
            fps=body.fps,
            drop_frame=body.drop_frame if body.fps is not None else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compile failed: {e}") from e

    module_name, suffix = _EXPORTERS[body.format]
    mod = importlib.import_module(module_name)

    # Per-format opts. Exporters all accept **opts so unknown kwargs are ignored,
    # but we only forward the ones the format actually uses to keep request
    # validation tight.
    opts: dict = {}
    if body.format in ("after_effects", "disguise") and body.fps is not None:
        opts["fps"] = body.fps
    if body.format == "midi" and body.ticks_per_beat is not None:
        opts["ticks_per_beat"] = body.ticks_per_beat
    if body.format == "osc":
        if body.osc_host is not None:
            opts["host"] = body.osc_host
        if body.osc_port is not None:
            opts["port"] = body.osc_port

    fname_stem = _sanitize_filename(body.filename)
    download_name = f"{fname_stem}{suffix}"

    # NamedTemporaryFile with delete=False so we can return its path; FastAPI's
    # FileResponse streams it. Cleanup happens via OS temp dir reaping; keeping
    # the file around for a few minutes is fine and avoids a tricky background
    # task race where the response hasn't finished sending yet.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    try:
        mod.export(cuesheet, Path(tmp.name), **opts)
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"export failed: {e}") from e

    return FileResponse(
        path=tmp.name,
        media_type="application/octet-stream",
        filename=download_name,
    )
