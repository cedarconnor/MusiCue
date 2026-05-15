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

from dataclasses import dataclass

MANIFEST_SCHEMA = "cedartoy-project/1"


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
