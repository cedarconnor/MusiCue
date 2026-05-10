"""Filesystem walker that populates the SQLite index."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from musicue.index import schema


def _utc_iso_from_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _read_optional_text(p: Path) -> str | None:
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


def _source_ext(song_dir: Path) -> str | None:
    for child in song_dir.iterdir():
        if child.is_file() and child.name.startswith("source."):
            return child.suffix.lstrip(".")
    return None


def _summarise_analysis(analysis_dir: Path) -> dict | None:
    ajson = analysis_dir / "analysis.json"
    if not ajson.exists():
        return None
    try:
        data = json.loads(ajson.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    has_stems = (analysis_dir / "stems").is_dir()
    has_clap = False
    has_drum_cls = False
    onsets = data.get("onsets") or {}
    for stem, items in onsets.items():
        for o in items or []:
            if o.get("labels"):
                has_clap = True
            if stem == "drums" and o.get("drum_class"):
                has_drum_cls = True
            if has_clap and has_drum_cls:
                break
        if has_clap and has_drum_cls:
            break
    phrases = data.get("phrases") or {}
    if not has_clap:
        for items in phrases.values():
            for ph in items or []:
                if ph.get("labels"):
                    has_clap = True
                    break
            if has_clap:
                break
    return {
        "schema_ver": str(data.get("schema_version", "")),
        "duration_sec": (data.get("source") or {}).get("duration_sec"),
        "bpm_global": (data.get("tempo") or {}).get("bpm_global"),
        "lufs_integrated": data.get("lufs_integrated"),
        "has_stems": int(has_stems),
        "has_clap": int(has_clap),
        "has_drum_cls": int(has_drum_cls),
    }


def _iter_song_dirs(storage_root: Path) -> Iterable[Path]:
    songs_dir = storage_root / "songs"
    if not songs_dir.exists():
        return ()
    return (d for d in sorted(songs_dir.iterdir()) if d.is_dir())


def _ingest_song_dir(db: sqlite3.Connection, song_dir: Path) -> None:
    """Insert / replace one song's rows from its directory.

    Cascading FK on analyses takes care of analyses + loop_regions tied
    to this song when the song row is deleted first.
    """
    sid = song_dir.name
    db.execute("DELETE FROM songs WHERE id=?", (sid,))

    title = _read_optional_text(song_dir / "title.txt") or sid
    source_url = _read_optional_text(song_dir / "source_url.txt")
    source_ext = _source_ext(song_dir) or "bin"
    trashed_at = _read_optional_text(song_dir / ".trashed_at")
    has_thumbnail = int((song_dir / "thumbnail.jpg").exists())
    added_at = _utc_iso_from_mtime(song_dir)

    analyses_dir = song_dir / "analyses"
    latest: dict | None = None
    analysis_rows: list[tuple] = []
    loop_rows: list[tuple] = []
    if analyses_dir.exists():
        for adir in sorted(
            analyses_dir.iterdir(),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        ):
            if not adir.is_dir():
                continue
            summary = _summarise_analysis(adir)
            if summary is None:
                continue
            created_at = _utc_iso_from_mtime(adir)
            analysis_rows.append(
                (
                    adir.name,
                    sid,
                    created_at,
                    summary["has_stems"],
                    summary["has_clap"],
                    summary["has_drum_cls"],
                    summary["schema_ver"],
                )
            )
            latest = summary
            loops = adir / "loops.json"
            if loops.exists():
                try:
                    ld = json.loads(loops.read_text(encoding="utf-8"))
                    loop_rows.append(
                        (
                            sid,
                            adir.name,
                            float(ld["loop_in"]),
                            float(ld["loop_out"]),
                            int(bool(ld.get("enabled", True))),
                            ld.get(
                                "updated_at",
                                datetime.now(tz=timezone.utc).isoformat(),
                            ),
                        )
                    )
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    pass

    db.execute(
        """INSERT INTO songs
           (id, title, source_url, source_ext, duration_sec, bpm_global,
            lufs_integrated, added_at, trashed_at, has_thumbnail)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            sid,
            title,
            source_url,
            source_ext,
            (latest or {}).get("duration_sec"),
            (latest or {}).get("bpm_global"),
            (latest or {}).get("lufs_integrated"),
            added_at,
            trashed_at,
            has_thumbnail,
        ),
    )
    if analysis_rows:
        db.executemany(
            """INSERT INTO analyses
               (id, song_id, created_at, has_stems, has_clap,
                has_drum_cls, schema_ver)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            analysis_rows,
        )
    if loop_rows:
        db.executemany(
            """INSERT INTO loop_regions
               (song_id, analysis_id, loop_in, loop_out, enabled, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            loop_rows,
        )


def rebuild(db: sqlite3.Connection, storage_root: Path) -> None:
    """Drop user data tables and reinsert from filesystem.

    Caller must have already invoked schema.create_all(db). On the very
    first rebuild that's an empty schema; on subsequent rebuilds (after
    a schema bump) drop_all + create_all should run before this is called.
    """
    db.execute("DELETE FROM loop_regions")
    db.execute("DELETE FROM analyses")
    db.execute("DELETE FROM songs")
    for song_dir in _iter_song_dirs(storage_root):
        _ingest_song_dir(db, song_dir)
    db.commit()


def refresh_song(
    db: sqlite3.Connection, storage_root: Path, song_id: str
) -> None:
    """Re-ingest a single song's directory after upload / analyze writes."""
    song_dir = storage_root / "songs" / song_id
    if not song_dir.is_dir():
        db.execute("DELETE FROM songs WHERE id=?", (song_id,))
        db.commit()
        return
    _ingest_song_dir(db, song_dir)
    db.commit()


def ensure_current(db: sqlite3.Connection, storage_root: Path) -> bool:
    """Bring the DB to SCHEMA_VERSION. Return True if a rebuild happened."""
    (uv,) = db.execute("PRAGMA user_version").fetchone()
    if uv == schema.SCHEMA_VERSION:
        return False
    schema.drop_all(db)
    schema.create_all(db)
    rebuild(db, storage_root)
    return True


def set_trashed(
    db: sqlite3.Connection, storage_root: Path, song_id: str, *, trashed: bool
) -> str | None:
    """Write-through trash flag. Returns the timestamp written (None if cleared)."""
    sidecar = storage_root / "songs" / song_id / ".trashed_at"
    if not sidecar.parent.exists():
        raise FileNotFoundError(f"song dir missing: {song_id}")
    if trashed:
        ts = datetime.now(tz=timezone.utc).isoformat()
        sidecar.write_text(ts, encoding="utf-8")
        db.execute("UPDATE songs SET trashed_at=? WHERE id=?", (ts, song_id))
    else:
        sidecar.unlink(missing_ok=True)
        ts = None
        db.execute("UPDATE songs SET trashed_at=NULL WHERE id=?", (song_id,))
    db.commit()
    return ts


def set_loop(
    db: sqlite3.Connection,
    storage_root: Path,
    song_id: str,
    analysis_id: str,
    loop: dict,
) -> dict:
    """Write-through loop region. Returns the persisted dict (with updated_at)."""
    analysis_dir = storage_root / "songs" / song_id / "analyses" / analysis_id
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"analysis dir missing: {song_id}/{analysis_id}")
    payload = {
        "loop_in": float(loop["loop_in"]),
        "loop_out": float(loop["loop_out"]),
        "enabled": bool(loop.get("enabled", True)),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (analysis_dir / "loops.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    db.execute(
        """INSERT INTO loop_regions
           (song_id, analysis_id, loop_in, loop_out, enabled, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(song_id, analysis_id) DO UPDATE SET
             loop_in    = excluded.loop_in,
             loop_out   = excluded.loop_out,
             enabled    = excluded.enabled,
             updated_at = excluded.updated_at""",
        (
            song_id,
            analysis_id,
            payload["loop_in"],
            payload["loop_out"],
            int(payload["enabled"]),
            payload["updated_at"],
        ),
    )
    db.commit()
    return payload


def delete_song(
    db: sqlite3.Connection, storage_root: Path, song_id: str
) -> None:
    """Hard delete: refuse unless `.trashed_at` sidecar present."""
    song_dir = storage_root / "songs" / song_id
    if not (song_dir / ".trashed_at").exists():
        raise ValueError("song must be moved to trash before hard-delete")
    if song_dir.exists():
        shutil.rmtree(song_dir)
    db.execute("DELETE FROM songs WHERE id=?", (song_id,))
    db.commit()


def empty_trash(
    db: sqlite3.Connection, storage_root: Path
) -> dict[str, int | list[str]]:
    rows = db.execute(
        "SELECT id FROM songs WHERE trashed_at IS NOT NULL"
    ).fetchall()
    deleted: list[str] = []
    skipped: list[str] = []
    for (sid,) in rows:
        try:
            delete_song(db, storage_root, sid)
            deleted.append(sid)
        except (OSError, ValueError):
            skipped.append(sid)
    return {"deleted": len(deleted), "skipped": skipped}
