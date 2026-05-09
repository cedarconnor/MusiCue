"""yt-dlp wrapper for v0.1a URL ingest.

Single responsibility: turn a user-supplied URL into a WAV on disk plus
metadata. No FastAPI imports; tests substitute a fake ``YoutubeDL``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


# Late-bind so tests can monkeypatch ``ingest._YoutubeDL`` without importing
# yt_dlp at module-load time (saves ~150ms of test collection).
def _YoutubeDL(opts: dict[str, Any]):  # pragma: no cover - replaced in prod path
    from yt_dlp import YoutubeDL

    return YoutubeDL(opts)


@dataclass
class DownloadedTrack:
    audio_path: Path
    title: str
    thumbnail_url: str | None
    duration_sec: float | None
    source_url: str


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError("missing host in url")


def _make_progress_hook(
    cb: Callable[[float], None] | None,
) -> Callable[[dict[str, Any]], None]:
    def hook(d: dict[str, Any]) -> None:
        if cb is None:
            return
        if d.get("status") != "downloading":
            return
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        done = d.get("downloaded_bytes")
        if not total or done is None:
            return
        cb(min(1.0, max(0.0, done / total)))

    return hook


def download_url(
    url: str,
    dest_dir: Path,
    progress_cb: Callable[[float], None] | None = None,
) -> DownloadedTrack:
    """Download ``url`` into ``dest_dir`` as a WAV; return metadata.

    ``dest_dir`` should be a freshly-created (writable) directory; the caller
    typically uses ``tempfile.TemporaryDirectory()``. The returned
    ``audio_path`` lives inside ``dest_dir`` and is valid until the caller
    cleans the directory.
    """
    _validate_url(url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_make_progress_hook(progress_cb)],
    }

    with _YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    track_id = info["id"]
    # Post-processor rewrote the file to .wav regardless of the source ext.
    audio_path = dest_dir / f"{track_id}.wav"
    if not audio_path.exists():
        # Fall back to whatever ext yt-dlp reported.
        audio_path = dest_dir / f"{track_id}.{info.get('ext', 'wav')}"
    if not audio_path.exists():
        raise RuntimeError(f"yt-dlp reported success but audio missing at {audio_path}")

    return DownloadedTrack(
        audio_path=audio_path,
        title=info.get("title") or track_id,
        thumbnail_url=info.get("thumbnail"),
        duration_sec=float(info["duration"]) if info.get("duration") else None,
        source_url=info.get("webpage_url") or url,
    )


def _urlopen(url: str, timeout: float = 10.0):  # pragma: no cover - net path
    from urllib.request import urlopen

    return urlopen(url, timeout=timeout)


def _download_thumbnail(url: str, dest: Path) -> None:
    """Best-effort thumbnail fetch. Swallows errors — the v0.1b Library
    reads-or-skips, so a missing thumbnail.jpg is not fatal."""
    try:
        with _urlopen(url, timeout=10.0) as r:
            dest.write_bytes(r.read())
    except Exception:
        return
