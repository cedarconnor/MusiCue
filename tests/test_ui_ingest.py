"""yt-dlp wrapper. Real network calls are out of scope for unit tests."""
from __future__ import annotations

import ipaddress
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


def test_download_thumbnail_writes_jpg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bytes = b"\xff\xd8\xff\xe0fakejpeg"

    def fake_urlopen(url: str, timeout: float = 10.0):
        class _Resp:
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): return None
            def read(self_inner): return fake_bytes
        return _Resp()

    monkeypatch.setattr(ingest, "_urlopen", fake_urlopen)

    out = tmp_path / "thumbnail.jpg"
    ingest._download_thumbnail("https://example.com/thumb.jpg", out)

    assert out.read_bytes() == fake_bytes


def test_download_thumbnail_swallows_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(url: str, timeout: float = 10.0):
        raise OSError("network down")

    monkeypatch.setattr(ingest, "_urlopen", boom)

    out = tmp_path / "thumbnail.jpg"
    # Must not raise; thumbnail is best-effort.
    ingest._download_thumbnail("https://example.com/thumb.jpg", out)
    assert not out.exists()


# ----- SSRF guard ----------------------------------------------------------


def _fake_resolve(ips: list[str]):
    return lambda host: [ipaddress.ip_address(ip) for ip in ips]


@pytest.mark.parametrize(
    "literal_url",
    [
        "http://127.0.0.1/",
        "http://localhost/",  # via resolution to 127.0.0.1
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata
        "http://[::1]/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
    ],
)
def test_validate_destination_safe_rejects_internal_targets(
    literal_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolve hostnames like "localhost" to 127.0.0.1.
    monkeypatch.setattr(ingest, "_resolve_hostname", _fake_resolve(["127.0.0.1"]))
    with pytest.raises(ValueError, match="refusing|loopback|private|link-local"):
        ingest._validate_destination_safe(literal_url)


def test_validate_destination_safe_accepts_public_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest, "_resolve_hostname", _fake_resolve(["8.8.8.8"]))
    # Should not raise.
    ingest._validate_destination_safe("https://example.com/x")


def test_validate_destination_safe_rejects_if_any_resolved_ip_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Hostname that resolves to BOTH a public and a private IP -- conservative
    # behaviour is to reject (an attacker can race the connect to the private).
    monkeypatch.setattr(
        ingest, "_resolve_hostname", _fake_resolve(["8.8.8.8", "10.0.0.1"])
    )
    with pytest.raises(ValueError, match="private"):
        ingest._validate_destination_safe("https://example.com/x")


def test_download_url_rejects_unresolvable_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(host: str):
        # Mirror what the real _resolve_hostname does on a gaierror.
        raise ValueError(f"could not resolve host {host!r}: no such host")

    monkeypatch.setattr(ingest, "_resolve_hostname", boom)
    with pytest.raises(ValueError, match="could not resolve"):
        ingest.download_url("https://nonexistent.invalid/x", tmp_path)


def test_duration_filter_rejects_long_videos() -> None:
    f = ingest._make_duration_filter(30 * 60)
    assert f({"duration": 60}) is None
    assert "duration cap" in (f({"duration": 60 * 60}) or "")
    assert f({}) is None  # no declared duration -- accept; size cap covers
