"""Defensive validators for UI route parameters.

Song IDs are sha256(audio) hex digests (64 chars). Analysis IDs are the
first 12 chars of a deterministic cache key (also hex). Stems come from a
closed set. We reject anything else *before* path or glob construction so
metacharacters, traversal sequences, and unexpected lengths can't escape
the storage root or expand the glob pattern.
"""
from __future__ import annotations

import re

from fastapi import HTTPException

# sha256 hex = exactly 64 lowercase hex characters.
_SONG_ID_RE = re.compile(r"^[a-f0-9]{64}$")
# Analysis IDs are cache_key[:12], same charset, exactly 12 characters.
_ANALYSIS_ID_RE = re.compile(r"^[a-f0-9]{12}$")

ALLOWED_STEMS: frozenset[str] = frozenset({"mix", "drums", "bass", "vocals", "other"})


def validate_song_id(song_id: str) -> str:
    if not _SONG_ID_RE.fullmatch(song_id):
        raise HTTPException(status_code=400, detail="invalid song id")
    return song_id


def validate_analysis_id(analysis_id: str) -> str:
    if not _ANALYSIS_ID_RE.fullmatch(analysis_id):
        raise HTTPException(status_code=400, detail="invalid analysis id")
    return analysis_id


def validate_stem(stem: str) -> str:
    if stem not in ALLOWED_STEMS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown stem '{stem}'. Allowed: {', '.join(sorted(ALLOWED_STEMS))}",
        )
    return stem
