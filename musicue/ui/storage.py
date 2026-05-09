"""Filesystem-backed artifact storage for the Web UI.

Layout::

    <root>/
      songs/
        <source_sha256>/
          source.<ext>
          title.txt
          analyses/<analysis_id>/
            analysis.json
            peaks.mix.json
            peaks.<stem>.json
            stems/<stem>.wav

No SQLite index in the MVP; ``list_songs`` scans the filesystem.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _read_source_url(song_dir: Path) -> str | None:
    p = song_dir / "source_url.txt"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


@dataclass
class SongRecord:
    id: str
    title: str
    source_path: Path
    source_ext: str
    source_url: str | None = None


@dataclass
class SongSummary:
    id: str
    title: str
    has_analysis: bool
    analysis_ids: list[str] = field(default_factory=list)
    source_url: str | None = None


class UIStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def songs_dir(self) -> Path:
        return self.root / "songs"

    def song_dir(self, song_id: str) -> Path:
        return self.songs_dir / song_id

    def analyses_dir(self, song_id: str) -> Path:
        return self.song_dir(song_id) / "analyses"

    def analysis_dir(self, song_id: str, analysis_id: str) -> Path:
        return self.analyses_dir(song_id) / analysis_id

    def register_source(self, src: Path, title: str | None = None) -> SongRecord:
        song_id = sha256_of_file(src)
        target_dir = self.song_dir(song_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lower().lstrip(".") or "bin"
        dest = target_dir / f"source.{ext}"
        if not dest.exists():
            shutil.copy2(src, dest)
        (target_dir / "title.txt").write_text(title or src.stem, encoding="utf-8")
        return SongRecord(
            id=song_id,
            title=title or src.stem,
            source_path=dest,
            source_ext=ext,
        )

    def list_songs(self) -> list[SongSummary]:
        if not self.songs_dir.exists():
            return []
        out: list[SongSummary] = []
        for d in sorted(self.songs_dir.iterdir()):
            if not d.is_dir():
                continue
            title_file = d / "title.txt"
            title = (
                title_file.read_text(encoding="utf-8").strip()
                if title_file.exists()
                else d.name
            )
            analyses_root = d / "analyses"
            analysis_ids: list[str] = []
            if analyses_root.exists():
                analysis_ids = sorted(
                    a.name for a in analyses_root.iterdir() if a.is_dir()
                )
            out.append(
                SongSummary(
                    id=d.name,
                    title=title,
                    has_analysis=bool(analysis_ids),
                    analysis_ids=analysis_ids,
                    source_url=_read_source_url(d),
                )
            )
        return out

    def get_song(self, song_id: str) -> SongRecord | None:
        d = self.song_dir(song_id)
        if not d.is_dir():
            return None
        sources = list(d.glob("source.*"))
        if not sources:
            return None
        src = sources[0]
        title = (
            (d / "title.txt").read_text(encoding="utf-8").strip()
            if (d / "title.txt").exists()
            else src.stem
        )
        return SongRecord(
            id=song_id,
            title=title,
            source_path=src,
            source_ext=src.suffix.lstrip("."),
            source_url=_read_source_url(d),
        )
