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


def test_build_folder_copies_stems_when_requested(tmp_path):
    audio_src = tmp_path / "src" / "song.wav"
    audio_src.parent.mkdir(parents=True)
    _write_silent_wav(audio_src)

    stems_src = tmp_path / "stems"
    stems_src.mkdir()
    for name in ("drums", "bass", "vocals", "other"):
        _write_silent_wav(stems_src / f"{name}.wav")

    out_dir = tmp_path / "out" / "song"
    analysis = make_analysis_fixture(audio_path=audio_src)
    cuesheet = make_cuesheet_fixture(source_sha256=analysis.source.sha256)

    manifest = build_cedartoy_folder(
        audio_path=audio_src,
        analysis=analysis,
        cuesheet=cuesheet,
        out_dir=out_dir,
        grammar="concert_visuals",
        musicue_version="0.4.1-test",
        include_stems=True,
        stems_src_dir=stems_src,
    )

    for name in ("drums", "bass", "vocals", "other"):
        assert (out_dir / "stems" / f"{name}.wav").exists()
    assert manifest.stems_omitted_reason is None


def test_build_folder_records_reason_when_stems_src_missing(tmp_path):
    audio_src = tmp_path / "src" / "song.wav"
    audio_src.parent.mkdir(parents=True)
    _write_silent_wav(audio_src)

    out_dir = tmp_path / "out" / "song"
    analysis = make_analysis_fixture(audio_path=audio_src)
    cuesheet = make_cuesheet_fixture(source_sha256=analysis.source.sha256)

    manifest = build_cedartoy_folder(
        audio_path=audio_src,
        analysis=analysis,
        cuesheet=cuesheet,
        out_dir=out_dir,
        grammar="concert_visuals",
        musicue_version="0.4.1-test",
        include_stems=True,
        stems_src_dir=tmp_path / "does" / "not" / "exist",
    )

    assert not (out_dir / "stems").exists()
    assert "cache missing" in (manifest.stems_omitted_reason or "")
    saved = json.loads((out_dir / "manifest.json").read_text())
    assert "cache missing" in saved["stems_omitted_reason"]


def test_build_folder_decodes_non_wav_audio(tmp_path):
    import numpy as np
    import soundfile as sf

    flac_src = tmp_path / "src" / "song.flac"
    flac_src.parent.mkdir(parents=True)
    sr = 44100
    sf.write(str(flac_src), np.zeros(int(sr * 0.25), dtype="float32"), sr)

    out_dir = tmp_path / "out" / "song"
    analysis = make_analysis_fixture(audio_path=flac_src)
    cuesheet = make_cuesheet_fixture(source_sha256=analysis.source.sha256)

    manifest = build_cedartoy_folder(
        audio_path=flac_src,
        analysis=analysis,
        cuesheet=cuesheet,
        out_dir=out_dir,
        grammar="concert_visuals",
        musicue_version="0.4.1-test",
    )

    out_wav = out_dir / "song.wav"
    assert out_wav.exists()
    info = sf.info(str(out_wav))
    assert info.samplerate == sr
    assert manifest.original_audio == "song.flac"


def test_build_folder_atomic_on_failure(tmp_path, monkeypatch):
    audio_src = tmp_path / "src" / "song.wav"
    audio_src.parent.mkdir(parents=True)
    _write_silent_wav(audio_src)

    out_dir = tmp_path / "out" / "song"
    analysis = make_analysis_fixture(audio_path=audio_src)
    cuesheet = make_cuesheet_fixture(source_sha256=analysis.source.sha256)

    # Force build_bundle to raise to simulate mid-build failure.
    import musicue.compile.cedartoy_folder as mod
    def boom(*a, **kw):
        raise RuntimeError("synthetic build_bundle failure")
    monkeypatch.setattr(mod, "build_bundle", boom, raising=False)
    # build_bundle is imported inside build_cedartoy_folder; patch the
    # source module too so the local import picks up the boom.
    import musicue.compile.bundle as bundle_mod
    monkeypatch.setattr(bundle_mod, "build_bundle", boom)

    with pytest.raises(RuntimeError, match="synthetic"):
        build_cedartoy_folder(
            audio_path=audio_src,
            analysis=analysis,
            cuesheet=cuesheet,
            out_dir=out_dir,
            grammar="concert_visuals",
            musicue_version="0.4.1-test",
        )

    # No folder at the target.
    assert not out_dir.exists()
    # And no leftover .cedartoy-tmp-* siblings.
    assert not any(
        p.name.startswith(".cedartoy-tmp-")
        for p in out_dir.parent.iterdir()
    )
