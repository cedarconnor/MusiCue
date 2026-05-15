"""Unit tests for the CedarToy folder export builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from musicue.compile.cedartoy_folder import (
    CedarToyProjectManifest,
    MANIFEST_SCHEMA,
)


def test_manifest_to_dict_includes_required_fields():
    m = CedarToyProjectManifest(
        audio_filename="song.wav",
        original_audio="My Song Title.mp3",
        grammar="concert_visuals",
        musicue_version="0.4.1",
        exported_at="2026-05-14T19:32:11Z",
    )
    d = m.to_dict()
    assert d["schema"] == MANIFEST_SCHEMA
    assert d["audio_filename"] == "song.wav"
    assert d["original_audio"] == "My Song Title.mp3"
    assert d["grammar"] == "concert_visuals"
    assert d["musicue_version"] == "0.4.1"
    assert d["exported_at"] == "2026-05-14T19:32:11Z"
    assert "stems_omitted_reason" not in d  # only emitted when set


def test_manifest_emits_stems_omitted_reason_when_set():
    m = CedarToyProjectManifest(
        audio_filename="song.wav",
        original_audio="song.wav",
        grammar="concert_visuals",
        musicue_version="0.4.1",
        exported_at="2026-05-14T19:32:11Z",
        stems_omitted_reason="cache missing and force_analyze=false",
    )
    d = m.to_dict()
    assert d["stems_omitted_reason"] == "cache missing and force_analyze=false"
