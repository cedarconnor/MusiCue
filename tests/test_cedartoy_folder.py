"""Unit tests for the CedarToy folder export builder."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from musicue.compile.cedartoy_folder import (
    CedarToyProjectManifest,
    MANIFEST_SCHEMA,
    build_cedartoy_folder,
)
from musicue.schemas import MusiCueBundle

from tests.test_bundle_builder import (
    make_analysis_fixture,
    make_cuesheet_fixture,
)


def _write_silent_wav(path: Path, duration_sec: float = 0.25) -> None:
    import numpy as np
    import soundfile as sf

    sr = 44100
    n = int(sr * duration_sec)
    sf.write(str(path), np.zeros(n, dtype="float32"), sr, subtype="PCM_16")


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


def test_build_folder_writes_audio_bundle_and_manifest(tmp_path):
    audio_src = tmp_path / "src" / "song.wav"
    audio_src.parent.mkdir(parents=True)
    _write_silent_wav(audio_src)

    out_dir = tmp_path / "out" / "song"

    analysis = make_analysis_fixture(audio_path=audio_src)
    cuesheet = make_cuesheet_fixture(source_sha256=analysis.source.sha256)

    build_cedartoy_folder(
        audio_path=audio_src,
        analysis=analysis,
        cuesheet=cuesheet,
        out_dir=out_dir,
        grammar="concert_visuals",
        musicue_version="0.4.1-test",
        exported_at="2026-05-14T00:00:00Z",
    )

    assert (out_dir / "song.wav").exists()
    assert (out_dir / "song.musicue.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert not (out_dir / "stems").exists()

    # Bundle round-trips through the schema.
    bundle = MusiCueBundle.model_validate_json(
        (out_dir / "song.musicue.json").read_text()
    )
    assert bundle.source_sha256 == analysis.source.sha256

    # Manifest carries the supplied metadata.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema"] == "cedartoy-project/1"
    assert manifest["audio_filename"] == "song.wav"
    assert manifest["grammar"] == "concert_visuals"
    assert manifest["musicue_version"] == "0.4.1-test"
