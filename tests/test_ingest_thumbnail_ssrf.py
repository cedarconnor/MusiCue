"""Thumbnail fetch SSRF + size cap.

The yt-dlp thumbnail URL is not the user-supplied URL — it's whatever the
remote info-dict gives us — so we re-validate it through the same guard,
and cap the body size to keep a hostile redirect from filling the disk.
"""
from __future__ import annotations

import ipaddress
from pathlib import Path

from musicue.ui import ingest


def test_thumbnail_rejects_loopback(monkeypatch, tmp_path):
    # If _validate_destination_safe is allowed to call DNS, it'd resolve
    # localhost → 127.0.0.1 and reject. Patch _resolve_hostname for
    # deterministic behaviour.
    def fake_resolve(host: str):
        return [ipaddress.ip_address("127.0.0.1")]

    monkeypatch.setattr(ingest, "_resolve_hostname", fake_resolve)
    # _urlopen should never be reached.
    called = {"n": 0}

    def fake_urlopen(*_a, **_kw):
        called["n"] += 1
        raise AssertionError("should have been blocked before fetch")

    monkeypatch.setattr(ingest, "_urlopen", fake_urlopen)

    dest = tmp_path / "thumb.jpg"
    ingest._download_thumbnail("http://internal-host/thumb.jpg", dest)
    assert called["n"] == 0
    assert not dest.exists()


def test_thumbnail_rejects_oversize_body(monkeypatch, tmp_path):
    """A response bigger than the cap is dropped, no partial write."""

    def fake_resolve(host: str):
        return [ipaddress.ip_address("8.8.8.8")]  # public, passes guard

    class FakeResponse:
        def read(self, n: int) -> bytes:
            # Return n+1 bytes so the caller's "is this over the cap" check
            # fires.
            return b"x" * n

        def __enter__(self): return self
        def __exit__(self, *_): pass

    monkeypatch.setattr(ingest, "_resolve_hostname", fake_resolve)
    monkeypatch.setattr(ingest, "_urlopen", lambda *a, **k: FakeResponse())

    dest = tmp_path / "thumb.jpg"
    ingest._download_thumbnail("https://example.com/giant.jpg", dest)
    assert not dest.exists()


def test_thumbnail_writes_small_body(monkeypatch, tmp_path):
    def fake_resolve(host: str):
        return [ipaddress.ip_address("8.8.8.8")]

    class FakeResponse:
        def read(self, n: int) -> bytes:
            return b"\xff\xd8\xffJPEG"  # well-under-cap JPEG-ish payload

        def __enter__(self): return self
        def __exit__(self, *_): pass

    monkeypatch.setattr(ingest, "_resolve_hostname", fake_resolve)
    monkeypatch.setattr(ingest, "_urlopen", lambda *a, **k: FakeResponse())

    dest = tmp_path / "thumb.jpg"
    ingest._download_thumbnail("https://example.com/small.jpg", dest)
    assert dest.exists()
    assert dest.read_bytes() == b"\xff\xd8\xffJPEG"


def test_thumbnail_rejects_invalid_scheme(monkeypatch, tmp_path):
    """`file://` and other non-http(s) schemes are dropped by _validate_url."""
    def fake_urlopen(*_a, **_kw):
        raise AssertionError("should have been blocked before fetch")

    monkeypatch.setattr(ingest, "_urlopen", fake_urlopen)
    dest = tmp_path / "thumb.jpg"
    ingest._download_thumbnail("file:///etc/passwd", dest)
    assert not dest.exists()
