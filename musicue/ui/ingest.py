"""yt-dlp wrapper for v0.1a URL ingest.

Single responsibility: turn a user-supplied URL into a WAV on disk plus
metadata. No FastAPI imports; tests substitute a fake ``YoutubeDL``.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

# Hard caps on what we'll let yt-dlp pull. v0.1a's job model is one analyze
# at a time on a developer machine; these are about cutting off pathological
# inputs (giant podcast archives, infinite livestreams) rather than
# adversarial untrusted input.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_DOWNLOAD_DURATION_SEC = 30 * 60  # 30 min


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
    """Cheap structural checks. Does NOT touch DNS -- callers wanting full
    SSRF protection must additionally call ``_validate_destination_safe``.
    Kept separate so the route layer can return 400 fast without the DNS
    round-trip on obviously-bad URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported url scheme: {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError("missing host in url")


def _resolve_hostname(host: str) -> list[ipaddress._BaseAddress]:
    """Resolve all A/AAAA records for ``host``. Used by the SSRF guard.
    Tests can monkeypatch this."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host {host!r}: {e}") from e
    out: list[ipaddress._BaseAddress] = []
    for info in infos:
        ip_str = info[4][0]
        # Strip IPv6 zone identifier if present (fe80::1%eth0).
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        try:
            out.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    return out


def _validate_destination_safe(url: str) -> None:
    """Reject URLs whose hostname resolves to a private/loopback/link-local
    address. Defends against trivial SSRF probes (e.g. cloud metadata at
    169.254.169.254, internal services at 10.x, localhost services).

    Caveats acknowledged for v0.1a:
    - DNS round-trip happens here but yt-dlp does its own DNS later -- a
      short-TTL record could resolve differently between the two calls
      (DNS rebinding). Mitigated by the localhost-bind default; production
      ship would need a custom HTTP transport that re-validates per-connect.
    - yt-dlp follows redirects to other hosts; we only validate the user's
      input URL. A YouTube video that redirects to 169.254.169.254 would
      bypass this check. Acceptable for v0.1a's threat model (single-user
      dev machine on localhost); not for multi-tenant deployments.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("missing host in url")

    # Block literal IP addresses pointed at internal ranges before DNS.
    try:
        literal = ipaddress.ip_address(host)
        ips: list[ipaddress._BaseAddress] = [literal]
    except ValueError:
        ips = _resolve_hostname(host)

    for ip in ips:
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(
                f"refusing to fetch from {ip} "
                f"(loopback/private/link-local/reserved)"
            )


def _make_duration_filter(max_duration_sec: float):
    """Return a yt-dlp ``match_filter`` callable that rejects entries whose
    declared duration exceeds ``max_duration_sec``. Returning a string from
    the filter signals yt-dlp to skip; ``None`` means accept."""

    def filter_(info: dict[str, Any], *_a, **_kw):
        duration = info.get("duration")
        if duration is not None and float(duration) > max_duration_sec:
            return f"track exceeds {max_duration_sec:.0f}s duration cap"
        return None

    return filter_


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
    _validate_destination_safe(url)
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
        # Caps to abort pathological inputs early.
        "max_filesize": MAX_DOWNLOAD_BYTES,
        "match_filter": _make_duration_filter(MAX_DOWNLOAD_DURATION_SEC),
        "socket_timeout": 30,
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


_MAX_THUMBNAIL_BYTES = 8 * 1024 * 1024  # 8 MiB hard cap; thumbnails are tiny.


def _download_thumbnail(url: str, dest: Path) -> None:
    """Best-effort thumbnail fetch. Swallows errors — the v0.1b Library
    reads-or-skips, so a missing thumbnail.jpg is not fatal.

    Goes through the same SSRF guard as the user-supplied URL. yt-dlp may
    hand us a thumbnail URL pointing at an internal/private address; that
    would silently exfiltrate metadata from the host network if we fetched
    it raw. We re-validate, cap the body size, and abandon the fetch on
    failure rather than partially writing the destination file.
    """
    try:
        _validate_url(url)
        _validate_destination_safe(url)
        with _urlopen(url, timeout=10.0) as r:
            # Read up to the cap + 1; any extra byte means the response was
            # bigger than we're willing to commit to disk, so we abandon.
            buf = r.read(_MAX_THUMBNAIL_BYTES + 1)
        if len(buf) > _MAX_THUMBNAIL_BYTES:
            return
        dest.write_bytes(buf)
    except Exception:
        # Cleanest behavior under any error (validation, network, write):
        # leave no partial thumbnail on disk. The Library reads-or-skips.
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return
