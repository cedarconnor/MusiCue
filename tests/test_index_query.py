"""query.py: list_songs filter/sort/search and loop accessors."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from musicue.index import query, schema


def _seed(db: sqlite3.Connection) -> None:
    schema.create_all(db)
    rows = [
        # id, title,         url,           ext,   dur,   bpm,   lufs,   added,                  trashed,                thumb
        ("a", "Alpha song",   "https://x/a", "wav", 60.0,  100.0, -10.0, "2026-05-08T00:00:00Z", None,                   1),
        ("b", "Bravo Tango",  None,           "m4a", 120.0, 130.0, -7.0,  "2026-05-09T00:00:00Z", None,                   0),
        ("c", "Charlie",      "https://x/c", "wav", 200.0, 150.0, -5.0,  "2026-05-09T01:00:00Z", "2026-05-09T02:00:00Z", 1),
    ]
    db.executemany(
        "INSERT INTO songs (id,title,source_url,source_ext,duration_sec,"
        "bpm_global,lufs_integrated,added_at,trashed_at,has_thumbnail) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.executemany(
        "INSERT INTO analyses (id,song_id,created_at,has_stems,has_clap,"
        "has_drum_cls,schema_ver) VALUES (?,?,?,?,?,?,?)",
        [
            ("an_a", "a", "2026-05-08T00:00:00Z", 1, 0, 0, "v3"),
            ("an_b", "b", "2026-05-09T00:00:00Z", 1, 1, 1, "v3"),
        ],
    )
    db.commit()


def test_list_songs_default_excludes_trashed(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db)
    ids = [r["id"] for r in out]
    assert ids == ["b", "a"]


def test_list_songs_trashed_only(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, trashed=True)
    assert [r["id"] for r in out] == ["c"]


def test_list_songs_filter_has_clap(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, filters=("has_clap",))
    assert [r["id"] for r in out] == ["b"]


def test_list_songs_filter_bpm_band(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, filters=("bpm_120_140",))
    assert [r["id"] for r in out] == ["b"]


def test_list_songs_sort_title(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, sort="title")
    assert [r["id"] for r in out] == ["a", "b"]


def test_list_songs_unknown_sort_falls_back(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, sort="DROP TABLE songs")
    assert [r["id"] for r in out] == ["b", "a"]


def test_list_songs_search_fts_or_like(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    _seed(db)
    out = query.list_songs(db, q="brav")
    assert [r["id"] for r in out] == ["b"]


def test_get_loop_round_trip(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "i.db")
    schema.create_all(db)
    db.execute(
        "INSERT INTO songs (id,title,source_ext,added_at) "
        "VALUES ('s','t','wav','2026-05-09T00:00:00Z')"
    )
    db.execute(
        "INSERT INTO loop_regions (song_id,analysis_id,loop_in,loop_out,"
        "enabled,updated_at) VALUES ('s','a',1.5,3.5,1,'2026-05-09T00:00:00Z')"
    )
    db.commit()
    loop = query.get_loop(db, "s", "a")
    assert loop == {
        "loop_in": 1.5,
        "loop_out": 3.5,
        "enabled": True,
        "updated_at": "2026-05-09T00:00:00Z",
    }
    assert query.get_loop(db, "s", "missing") is None
