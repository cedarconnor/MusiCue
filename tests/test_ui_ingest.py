"""yt-dlp wrapper. Real network calls are out of scope for unit tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from musicue.ui import ingest


class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL. Captures opts; returns scripted info."""

    instances: list["_FakeYDL"] = []

    def __init__(self, opts: dict[str, Any]) -> None:
        self.opts = opts
        self.extracted: list[str] = []
        type(self).instances.append(self)

    def __enter__(self) -> "_FakeYDL":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self.extracted.append(url)
        # Simulate yt-dlp writing the post-processed wav into outtmpl's dir.
        outtmpl = self.opts["outtmpl"]
        if isinstance(outtmpl, dict):
            outtmpl = outtmpl["default"]
        # outtmpl format: ".../<id>.%(ext)s" — substitute manually.
        target = Path(outtmpl.replace("%(id)s", "abc123").replace("%(ext)s", "wav"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfake")
        return {
            "id": "abc123",
            "title": "Test Song",
            "thumbnail": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
            "duration": 234,
            "webpage_url": url,
            "ext": "wav",  # post-processor rewrote
        }


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    _FakeYDL.instances.clear()


def test_download_url_returns_wav_path_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "_YoutubeDL", _FakeYDL)

    track = ingest.download_url(
        "https://www.youtube.com/watch?v=abc123", tmp_path
    )

    assert track.audio_path.exists()
    assert track.audio_path.suffix == ".wav"
    assert track.title == "Test Song"
    assert track.thumbnail_url == "https://i.ytimg.com/vi/abc123/hqdefault.jpg"
    assert track.duration_sec == 234
    assert track.source_url == "https://www.youtube.com/watch?v=abc123"


def test_download_url_rejects_file_scheme(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported url scheme"):
        ingest.download_url("file:///etc/passwd", tmp_path)


def test_download_url_rejects_hostless_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing host"):
        ingest.download_url("http:///just-a-path", tmp_path)


def test_download_url_propagates_yt_dlp_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BoomYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            raise RuntimeError("Private video")

    monkeypatch.setattr(ingest, "_YoutubeDL", _BoomYDL)

    with pytest.raises(RuntimeError, match="Private video"):
        ingest.download_url("https://www.youtube.com/watch?v=x", tmp_path)


def test_download_url_invokes_progress_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    class _ProgressYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            hooks = self.opts["progress_hooks"]
            for h in hooks:
                h({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                h({"status": "downloading", "downloaded_bytes": 100, "total_bytes": 100})
            return super().extract_info(url, download)

    monkeypatch.setattr(ingest, "_YoutubeDL", _ProgressYDL)

    ingest.download_url(
        "https://example.com/x", tmp_path, progress_cb=lambda f: seen.append(f)
    )

    assert seen == [0.5, 1.0]
