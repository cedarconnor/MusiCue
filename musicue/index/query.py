"""Read-only query helpers used by the routes layer."""
from __future__ import annotations

import sqlite3
from typing import Any, Iterable

_SORT_FIELDS = {"added_at", "title", "duration_sec", "bpm_global"}
_DEFAULT_SORT = "added_at"


def _row_to_song(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "source_url": row["source_url"],
        "source_ext": row["source_ext"],
        "duration_sec": row["duration_sec"],
        "bpm_global": row["bpm_global"],
        "lufs_integrated": row["lufs_integrated"],
        "added_at": row["added_at"],
        "trashed_at": row["trashed_at"],
        "has_thumbnail": bool(row["has_thumbnail"]),
    }


def _fts_query(q: str) -> str:
    """Quote each whitespace-split token; append `*` for prefix match.

    Any FTS5 metacharacters in user input are neutralized by the surrounding
    double quotes (we strip embedded quotes to avoid breaking out)."""
    tokens = [t.replace('"', "") for t in q.split() if t]
    if not tokens:
        return '""'
    return " ".join(f'"{t}"*' for t in tokens)


def _has_fts5(db: sqlite3.Connection) -> bool:
    return bool(
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='songs_fts'"
        ).fetchone()
    )


def list_songs(
    db: sqlite3.Connection,
    *,
    q: str | None = None,
    filters: Iterable[str] = (),
    sort: str = _DEFAULT_SORT,
    trashed: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    db.row_factory = sqlite3.Row
    where = ["trashed_at IS " + ("NOT NULL" if trashed else "NULL")]
    params: list[Any] = []

    if q:
        if _has_fts5(db):
            where.append(
                "id IN (SELECT song_id FROM songs_fts WHERE songs_fts MATCH ?)"
            )
            params.append(_fts_query(q))
        else:
            like = f"%{q}%"
            where.append("(title LIKE ? OR COALESCE(source_url,'') LIKE ?)")
            params.extend([like, like])

    fset = set(filters)
    if "has_stems" in fset:
        where.append(
            "EXISTS (SELECT 1 FROM analyses a "
            "WHERE a.song_id = songs.id AND a.has_stems = 1)"
        )
    if "has_clap" in fset:
        where.append(
            "EXISTS (SELECT 1 FROM analyses a "
            "WHERE a.song_id = songs.id AND a.has_clap = 1)"
        )
    if "has_url" in fset:
        where.append("source_url IS NOT NULL")
    if "bpm_80_120" in fset:
        where.append("bpm_global >= 80 AND bpm_global < 120")
    if "bpm_120_140" in fset:
        where.append("bpm_global >= 120 AND bpm_global < 140")
    if "bpm_140_plus" in fset:
        where.append("bpm_global >= 140")
    if "recent_24h" in fset:
        where.append("added_at >= datetime('now','-1 day')")
    if "recent_7d" in fset:
        where.append("added_at >= datetime('now','-7 days')")

    if sort not in _SORT_FIELDS:
        sort = _DEFAULT_SORT
    direction = "ASC" if sort == "title" else "DESC"

    sql = (
        f"SELECT id,title,source_url,source_ext,duration_sec,bpm_global,"
        f"lufs_integrated,added_at,trashed_at,has_thumbnail "
        f"FROM songs WHERE {' AND '.join(where)} "
        f"ORDER BY {sort} {direction} LIMIT ? OFFSET ?"
    )
    rows = db.execute(sql, [*params, int(limit), int(offset)]).fetchall()
    return [_row_to_song(r) for r in rows]


def get_loop(
    db: sqlite3.Connection, song_id: str, analysis_id: str
) -> dict[str, Any] | None:
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT loop_in, loop_out, enabled, updated_at FROM loop_regions "
        "WHERE song_id = ? AND analysis_id = ?",
        (song_id, analysis_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "loop_in": row["loop_in"],
        "loop_out": row["loop_out"],
        "enabled": bool(row["enabled"]),
        "updated_at": row["updated_at"],
    }


def get_analysis_count(db: sqlite3.Connection, song_id: str) -> int:
    (n,) = db.execute(
        "SELECT COUNT(*) FROM analyses WHERE song_id=?", (song_id,)
    ).fetchone()
    return int(n)
