"""SQLite index for the Web UI Library.

Filesystem-canonical: this DB is a derived cache. On schema bump the DB
is dropped and rebuilt from disk. User-data writes (trash, loop) are
write-through to sidecar files so a rebuild is lossless.
"""
