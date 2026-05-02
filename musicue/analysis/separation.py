from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from importlib.metadata import version as _pkg_version

    def demucs_version() -> str:
        try:
            return _pkg_version("demucs")
        except Exception:
            return "unknown"
except ImportError:
    def demucs_version() -> str:
        return "unknown"


def separate(
    audio_path: Path,
    out_dir: Path,
    model: str = "htdemucs_ft",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(out_dir),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    stem_dir = out_dir / model / audio_path.stem
    stems: dict[str, Path] = {}
    for stem_name in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{stem_name}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Demucs did not produce expected stem: {p}")
        stems[stem_name] = p
    return stems
