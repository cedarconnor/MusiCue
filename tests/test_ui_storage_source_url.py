"""Source URL field round-trip for v0.1a URL ingest."""
from __future__ import annotations

from pathlib import Path

import pytest

from musicue.ui.storage import UIStorage


@pytest.fixture
def storage(tmp_path: Path) -> UIStorage:
    return UIStorage(tmp_path)


def _write_audio(p: Path) -> None:
    p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")


def test_source_url_absent_returns_none(storage: UIStorage, tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _write_audio(src)
    rec = storage.register_source(src, title="local")

    got = storage.get_song(rec.id)
    assert got is not None
    assert got.source_url is None

    summaries = storage.list_songs()
    assert summaries[0].source_url is None


def test_source_url_persisted_via_sidecar_file(
    storage: UIStorage, tmp_path: Path
) -> None:
    src = tmp_path / "in.wav"
    _write_audio(src)
    rec = storage.register_source(src, title="from-url")

    # Caller (the ingest route) writes source_url.txt after register_source.
    (storage.song_dir(rec.id) / "source_url.txt").write_text(
        "https://www.youtube.com/watch?v=abc123", encoding="utf-8"
    )

    got = storage.get_song(rec.id)
    assert got is not None
    assert got.source_url == "https://www.youtube.com/watch?v=abc123"

    summaries = storage.list_songs()
    assert summaries[0].source_url == "https://www.youtube.com/watch?v=abc123"
