import csv

from musicue.exporters.touchdesigner import export


def test_td_export_creates_chop_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_td_export_chop_has_time_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "time" in headers


def test_td_export_chop_has_all_tracks(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "kick" in headers
    assert "energy" in headers


def test_td_export_events_csv_created(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    events_csv = tmp_path / "cuesheet_events.csv"
    assert events_csv.exists()


def test_td_events_csv_has_track_time_strength(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    events_csv = tmp_path / "cuesheet_events.csv"
    with open(events_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    assert "track" in headers
    assert "t" in headers
    assert "strength" in headers
    assert len(rows) > 0
