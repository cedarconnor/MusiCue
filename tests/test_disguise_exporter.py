import csv

from musicue.exporters.disguise import export


def test_disguise_export_creates_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_disguise_csv_has_timecode_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert any("timecode" in h.lower() or "tc" in h.lower() for h in headers)


def test_disguise_csv_has_cue_name_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert any("name" in h.lower() or "cue" in h.lower() for h in headers)


def test_disguise_csv_timecode_format(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    tc_col = next(k for k in rows[0] if "timecode" in k.lower() or "tc" in k.lower())
    tc = rows[0][tc_col]
    parts = tc.split(":")
    assert len(parts) == 4


def test_disguise_csv_kick_events_present(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    cue_names = [list(r.values())[1] for r in rows]
    kick_cues = [n for n in cue_names if "kick" in n.lower()]
    assert len(kick_cues) >= 3
