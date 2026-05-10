"""SQLite schema and FTS5 feature detection."""
from __future__ import annotations

import sqlite3

# Bump together with PRAGMA user_version. On a startup mismatch the index
# is dropped and rebuilt from filesystem.
SCHEMA_VERSION = 1


def has_fts5(db: sqlite3.Connection) -> bool:
    """Return True if the bundled SQLite supports FTS5."""
    try:
        db.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        db.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def create_all(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS songs (
          id              TEXT PRIMARY KEY,
          title           TEXT NOT NULL,
          source_url      TEXT,
          source_ext      TEXT NOT NULL,
          duration_sec    REAL,
          bpm_global      REAL,
          lufs_integrated REAL,
          added_at        TEXT NOT NULL,
          trashed_at      TEXT,
          has_thumbnail   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS songs_title_idx   ON songs(title);
        CREATE INDEX IF NOT EXISTS songs_trashed_idx ON songs(trashed_at);

        CREATE TABLE IF NOT EXISTS analyses (
          id            TEXT PRIMARY KEY,
          song_id       TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
          created_at    TEXT NOT NULL,
          has_stems     INTEGER NOT NULL DEFAULT 0,
          has_clap      INTEGER NOT NULL DEFAULT 0,
          has_drum_cls  INTEGER NOT NULL DEFAULT 0,
          schema_ver    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS analyses_song_idx ON analyses(song_id);

        CREATE TABLE IF NOT EXISTS loop_regions (
          song_id       TEXT NOT NULL,
          analysis_id   TEXT NOT NULL,
          loop_in       REAL NOT NULL,
          loop_out      REAL NOT NULL,
          enabled       INTEGER NOT NULL,
          updated_at    TEXT NOT NULL,
          PRIMARY KEY (song_id, analysis_id),
          FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );
        """
    )

    if has_fts5(db):
        db.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS songs_fts USING fts5(
              song_id UNINDEXED,
              title,
              source_url,
              tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE TRIGGER IF NOT EXISTS songs_ai AFTER INSERT ON songs BEGIN
              INSERT INTO songs_fts(song_id, title, source_url)
              VALUES (new.id, new.title, COALESCE(new.source_url, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS songs_ad AFTER DELETE ON songs BEGIN
              DELETE FROM songs_fts WHERE song_id = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS songs_au AFTER UPDATE ON songs BEGIN
              UPDATE songs_fts
                 SET title = new.title,
                     source_url = COALESCE(new.source_url, '')
               WHERE song_id = old.id;
            END;
            """
        )

    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    db.commit()


def drop_all(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        DROP TRIGGER IF EXISTS songs_au;
        DROP TRIGGER IF EXISTS songs_ad;
        DROP TRIGGER IF EXISTS songs_ai;
        DROP TABLE IF EXISTS songs_fts;
        DROP TABLE IF EXISTS loop_regions;
        DROP TABLE IF EXISTS analyses;
        DROP INDEX IF EXISTS songs_trashed_idx;
        DROP INDEX IF EXISTS songs_title_idx;
        DROP TABLE IF EXISTS songs;
        """
    )
    db.execute("PRAGMA user_version = 0")
    db.commit()
