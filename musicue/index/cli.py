"""`musicue index` subcommand helpers.

Wired into the top-level Typer app in ``musicue/cli.py``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from musicue.index import index as indexer
from musicue.index import schema


def default_root() -> Path:
    return Path.home() / ".musicue"


def _open_db(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "index.db")
    db.execute("PRAGMA foreign_keys = ON")
    return db


def cmd_status(root: Path) -> int:
    db = _open_db(root)
    if not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='songs'"
    ).fetchone():
        print(f"index db: {root / 'index.db'} (not built)")
        return 0
    (n_songs,) = db.execute("SELECT COUNT(*) FROM songs").fetchone()
    (n_analyses,) = db.execute("SELECT COUNT(*) FROM analyses").fetchone()
    (n_loops,) = db.execute("SELECT COUNT(*) FROM loop_regions").fetchone()
    (uv,) = db.execute("PRAGMA user_version").fetchone()
    print(f"index db: {root / 'index.db'}")
    print(f"schema_version: {uv}")
    print(f"songs: {n_songs}")
    print(f"analyses: {n_analyses}")
    print(f"loop_regions: {n_loops}")
    return 0


def cmd_rebuild(root: Path) -> int:
    db = _open_db(root)
    schema.drop_all(db)
    schema.create_all(db)
    indexer.rebuild(db, root)
    (n_songs,) = db.execute("SELECT COUNT(*) FROM songs").fetchone()
    print(f"rebuilt index at {root / 'index.db'}: {n_songs} songs indexed")
    return 0
