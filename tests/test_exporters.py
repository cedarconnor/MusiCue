import csv
import json
import pytest
from pathlib import Path
from musicue.schemas import CueSheet, CueTrack


def _make_cuesheet() -> CueSheet:
    return CueSheet(
        source_sha256="abc",
        grammar="concert_visuals",
        duration_sec=5.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="drums",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
                    {"t": 2.0, "strength": 0.7, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
                ],
            ),
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=1.0,
                values=[-20.0, -18.0, -22.0, -19.0, -21.0],
            ),
        ],
    )


def test_json_export_creates_valid_file(tmp_path):
    from musicue.exporters.json_export import export
    out = tmp_path / "cuesheet.json"
    export(_make_cuesheet(), out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["grammar"] == "concert_visuals"
    assert len(data["tracks"]) == 2


def test_json_export_roundtrip(tmp_path):
    from musicue.exporters.json_export import export
    out = tmp_path / "cuesheet.json"
    export(_make_cuesheet(), out)
    cs2 = CueSheet.model_validate_json(out.read_text())
    assert cs2.duration_sec == pytest.approx(5.0)
    assert cs2.tracks[0].name == "drums"


def test_csv_export_creates_file(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    assert out.exists()


def test_csv_has_time_sec_column(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "time_sec" in headers


def test_csv_has_track_columns(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "energy" in headers
    assert "drums" in headers


def test_csv_row_count_matches_continuous_track(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # energy has 5 values at 1.0s hop → 5 rows
    assert len(rows) == 5


def test_csv_impulse_column_fires_at_event_time(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # event at t=0.5s → should be non-zero in the row nearest 0.5s
    drums_col = [float(r["drums"]) for r in rows]
    assert max(drums_col) > 0
