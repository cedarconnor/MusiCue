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
    stem_dir = out_dir / model / audio_path.stem
    expected: dict[str, Path] = {
        n: stem_dir / f"{n}.wav" for n in ("drums", "bass", "vocals", "other")
    }
    # Idempotent fast path: if the four expected stems are already on disk for this
    # (out_dir, model, song) tuple, skip the (very expensive) demucs subprocess.
    if all(p.exists() for p in expected.values()):
        return expected

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(out_dir),
        str(audio_path),
    ]
    # ``encoding='utf-8'`` + ``errors='replace'`` so demucs' progress bars
    # (which contain unicode block chars) don't crash the cp1252 charmap
    # decoder on Windows. ``errors='replace'`` keeps stderr readable.
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    for stem_name, p in expected.items():
        if not p.exists():
            raise FileNotFoundError(f"Demucs did not produce expected stem: {p}")
    return expected
