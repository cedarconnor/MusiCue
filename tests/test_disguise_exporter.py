"""Tests for the disguise cue-table exporter.

disguise (d3) expects a tab-separated cue table with a specific header
line and a ``<name>_cue_table.txt`` filename. These tests pin the format
so refactors can't quietly break d3 import.
"""
from musicue.exporters.disguise import _seconds_to_track_time, export


def _read(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_first_line_is_cue_table_header(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    lines = _read(out)
    assert lines[0].startswith("Cue table for "), lines[0]


def test_second_line_is_tab_separated_column_headers(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    lines = _read(out)
    assert lines[1] == "Beat\tTag\tNote\tTrack_Time\tTC_Time\tSection_Break"


def test_data_rows_have_six_tab_separated_columns(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    lines = _read(out)
    data_rows = lines[2:]
    assert len(data_rows) > 0
    for row in data_rows:
        assert row.count("\t") == 5, f"row should have 5 tabs / 6 cells: {row!r}"


def test_beat_column_is_seconds_with_five_decimals(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    lines = _read(out)
    for row in lines[2:]:
        beat = row.split("\t")[0]
        # 5 decimals → after the dot there are exactly 5 digits
        assert "." in beat
        assert len(beat.split(".")[1]) == 5, beat


def test_tag_tc_time_section_break_are_empty(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    lines = _read(out)
    for row in lines[2:]:
        cells = row.split("\t")
        # Beat, Tag, Note, Track_Time, TC_Time, Section_Break
        assert cells[1] == "", f"Tag should be empty, got {cells[1]!r}"
        assert cells[4] == "", f"TC_Time should be empty, got {cells[4]!r}"
        assert cells[5] == "", f"Section_Break should be empty, got {cells[5]!r}"


def test_track_time_is_hh_mm_ss_ff_at_30_fps(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out, fps=30.0)
    lines = _read(out)
    for row in lines[2:]:
        track_time = row.split("\t")[3]
        # HH:MM:SS.FF
        assert len(track_time) == 11, f"expected HH:MM:SS.FF, got {track_time!r}"
        assert track_time[2] == ":" and track_time[5] == ":" and track_time[8] == "."


def test_track_time_matches_sample_at_30_fps():
    # The disguise sample shows 185.23857s → 00:03:05.07 at 30 fps
    # (5557 floor-frames; 5557 - 5550 = 7 leftover frames).
    assert _seconds_to_track_time(185.23857, 30.0) == "00:03:05.07"
    # Other anchors from the sample.
    assert _seconds_to_track_time(0.0, 30.0) == "00:00:00.00"
    assert _seconds_to_track_time(30.0, 30.0) == "00:00:30.00"
    assert _seconds_to_track_time(1800.0, 30.0) == "00:30:00.00"
    assert _seconds_to_track_time(1920.0, 30.0) == "00:32:00.00"


def test_rows_are_sorted_by_beat_ascending(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    beats = [float(row.split("\t")[0]) for row in _read(out)[2:]]
    assert beats == sorted(beats), beats


def test_note_column_is_non_empty_for_data_rows(tmp_path, full_cuesheet):
    """Note carries the cue identifier so the operator can see what each
    row represents in d3. Empty notes would be useless."""
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out)
    notes = [row.split("\t")[2] for row in _read(out)[2:]]
    assert all(n for n in notes), "every cue should have a Note string"


def test_title_defaults_to_filename_stem(tmp_path, full_cuesheet):
    out = tmp_path / "neon queens_cue_table.txt"
    export(full_cuesheet, out)
    first = _read(out)[0]
    assert first == "Cue table for neon queens_cue_table"


def test_explicit_title_override(tmp_path, full_cuesheet):
    out = tmp_path / "track 1_cue_table.txt"
    export(full_cuesheet, out, title="track 3")
    assert _read(out)[0] == "Cue table for track 3"
