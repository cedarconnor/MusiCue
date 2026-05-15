"""Build a portable CedarToy project folder.

Layout written by build_cedartoy_folder()::

    <out_dir>/
      song.wav
      song.musicue.json
      stems/                       (optional)
        drums.wav  bass.wav  vocals.wav  other.wav
      manifest.json

Atomicity: everything is written to a sibling temp folder and renamed
into place on success. A failure mid-build leaves no folder at the
target path.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SCHEMA = "cedartoy-project/1"
STEM_NAMES = ("drums", "bass", "vocals", "other")


@dataclass
class CedarToyProjectManifest:
    audio_filename: str
    original_audio: str
    grammar: str
    musicue_version: str
    exported_at: str
    stems_omitted_reason: str | None = None
    schema: str = MANIFEST_SCHEMA

    def to_dict(self) -> dict:
        d: dict = {
            "schema": self.schema,
            "audio_filename": self.audio_filename,
            "original_audio": self.original_audio,
            "grammar": self.grammar,
            "musicue_version": self.musicue_version,
            "exported_at": self.exported_at,
        }
        if self.stems_omitted_reason is not None:
            d["stems_omitted_reason"] = self.stems_omitted_reason
        return d


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _copy_audio_as_wav(src: Path, dest: Path) -> None:
    """Copy src to dest. If src is already a WAV, copy bytes; else decode.

    Native sample rate is preserved — no resampling.

    Decoder strategy mirrors musicue.analysis.curves: try soundfile first
    (fast for WAV/FLAC/OGG) and fall back to librosa.load (which goes
    through audioread + ffmpeg) for compressed formats like M4A/MP3
    that soundfile can't open.
    """
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, dest)
        return
    import soundfile as sf
    try:
        data, sr = sf.read(str(src), always_2d=False)
    except (sf.LibsndfileError, RuntimeError):
        import librosa
        # librosa returns (channels, samples) when mono=False; transpose
        # so soundfile.write sees (samples, channels) or 1-D.
        y, sr = librosa.load(str(src), sr=None, mono=False)
        data = y.T if y.ndim == 2 else y
    sf.write(str(dest), data, sr, subtype="PCM_16")


def build_cedartoy_folder(
    *,
    audio_path: Path,
    analysis,
    cuesheet,
    out_dir: Path,
    grammar: str,
    musicue_version: str,
    exported_at: str | None = None,
    include_stems: bool = False,
    stems_src_dir: Path | None = None,
    original_audio_name: str | None = None,
) -> CedarToyProjectManifest:
    """Atomically build a CedarToy project folder at out_dir.

    Returns the manifest dataclass (also written as manifest.json).
    Raises FileExistsError if out_dir already exists; caller is
    responsible for clearing the path.
    """
    # Imports are local so importing this module doesn't pull pydantic
    # schemas in environments that only need the manifest dataclass.
    from musicue.compile.bundle import build_bundle

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"Target folder already exists: {out_dir} — caller must "
            f"remove it or pick a different path"
        )

    timestamp = exported_at or _iso_utc_now()
    original_audio_name = original_audio_name or audio_path.name

    stems_omitted_reason: str | None = None
    stems_to_copy: list[Path] = []
    if include_stems:
        src = Path(stems_src_dir) if stems_src_dir else None
        if src is None or not src.exists():
            stems_omitted_reason = (
                "cache missing and force_analyze=false; "
                f"stems_src_dir={src}"
            )
        else:
            for name in STEM_NAMES:
                p = src / f"{name}.wav"
                if p.exists():
                    stems_to_copy.append(p)
            if not stems_to_copy:
                stems_omitted_reason = (
                    f"cache missing (no stem WAVs in {src})"
                )

    manifest = CedarToyProjectManifest(
        audio_filename="song.wav",
        original_audio=original_audio_name,
        grammar=grammar,
        musicue_version=musicue_version,
        exported_at=timestamp,
        stems_omitted_reason=stems_omitted_reason,
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # Sibling temp dir so the atomic rename is on the same filesystem.
    tmp = Path(tempfile.mkdtemp(prefix=".cedartoy-tmp-", dir=out_dir.parent))
    try:
        _copy_audio_as_wav(audio_path, tmp / "song.wav")

        bundle = build_bundle(analysis, cuesheet)
        (tmp / "song.musicue.json").write_text(
            bundle.model_dump_json(indent=2), encoding="utf-8"
        )

        if stems_to_copy:
            (tmp / "stems").mkdir()
            for p in stems_to_copy:
                shutil.copy2(p, tmp / "stems" / p.name)

        (tmp / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )

        tmp.rename(out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest
