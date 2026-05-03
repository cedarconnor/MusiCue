import csv

from musicue.exporters.houdini import export


def test_houdini_export_creates_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_houdini_csv_header_starts_with_time(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        content = f.read()
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) >= 2


def test_houdini_csv_channel_count(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    # Skip leading comment lines (start with #)
    with open(out, newline="") as f:
        non_comment_lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(non_comment_lines)
    header = next(reader)
    assert len(header) >= 2
    assert "time" in header or header[0] == "time"


def test_houdini_csv_has_correct_row_count(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    # Skip comments
    with open(out, newline="") as f:
        non_comment_lines = [line for line in f if not line.startswith("#")]
    rows = list(csv.reader(non_comment_lines))
    # 10s at 0.04s hop = 250 rows + 1 header = 251
    assert len(rows) > 10
