from pathlib import Path

from musicue.ui.storage import (
    SongSummary,
    UIStorage,
    sha256_of_file,
)


def test_sha256_of_file_is_deterministic(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    assert sha256_of_file(f) == sha256_of_file(f)
    assert len(sha256_of_file(f)) == 64


def test_register_source_creates_layout(tmp_path: Path):
    storage = UIStorage(tmp_path / ".musicue")
    src = tmp_path / "Some Song.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100)

    record = storage.register_source(src, title="Some Song")

    assert (tmp_path / ".musicue" / "songs" / record.id / "source.wav").exists()
    title_text = (tmp_path / ".musicue" / "songs" / record.id / "title.txt").read_text(
        encoding="utf-8"
    ).strip()
    assert title_text == "Some Song"
    assert record.id == sha256_of_file(src)


def test_list_songs_scans_filesystem(tmp_path: Path):
    storage = UIStorage(tmp_path / ".musicue")
    src = tmp_path / "song.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 200)
    storage.register_source(src, title="song")

    songs = storage.list_songs()

    assert len(songs) == 1
    assert isinstance(songs[0], SongSummary)
    assert songs[0].title == "song"
    assert songs[0].has_analysis is False


def test_list_songs_with_analysis(tmp_path: Path):
    storage = UIStorage(tmp_path / ".musicue")
    src = tmp_path / "s.wav"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 200)
    rec = storage.register_source(src, title="s")
    (storage.analyses_dir(rec.id) / "abc123").mkdir(parents=True)

    songs = storage.list_songs()

    assert songs[0].has_analysis is True
    assert songs[0].analysis_ids == ["abc123"]


def test_get_song_returns_none_for_unknown(tmp_path: Path):
    storage = UIStorage(tmp_path / ".musicue")
    assert storage.get_song("deadbeef") is None
