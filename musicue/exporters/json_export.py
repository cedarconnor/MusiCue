from __future__ import annotations

from pathlib import Path

from musicue.schemas import CueSheet


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cuesheet.model_dump_json(indent=2))
