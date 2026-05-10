"""Round-trip tests for the /api/.../export route.

Each test plants a hand-crafted analysis.json on disk in the storage layout
the route expects, then exercises the route via TestClient and checks the
resulting download has format-appropriate content.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    OnsetEvent,
    SourceInfo,
    TimedCurve,
)
from musicue.ui.server import create_app

SONG_ID = "test-song-id"
ANALYSIS_ID = "test-analysis-id"


def _plant_analysis(storage_root: Path) -> None:
    """Write a minimal-but-valid analysis.json into the storage layout."""
    adir = storage_root / "songs" / SONG_ID / "analyses" / ANALYSIS_ID
    adir.mkdir(parents=True, exist_ok=True)
    analysis = AnalysisResult(
        source=SourceInfo(
            path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100
        ),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={
            "drums": [
                OnsetEvent(t=0.5, strength=0.8, drum_class="kick"),
                OnsetEvent(t=1.0, strength=0.6, drum_class="snare"),
            ]
        },
        curves={"lufs": TimedCurve(hop_sec=0.04, values=[-20.0] * 250)},
    )
    (adir / "analysis.json").write_text(
        analysis.model_dump_json(), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Happy-path round trips: one per format that takes options.
# ---------------------------------------------------------------------------


def test_export_csv_default(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={"format": "csv", "grammar": "concert_visuals"},
    )
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    # CSV exporter writes a header that includes the time column.
    assert "time_sec" in body.splitlines()[0]
    cd = r.headers.get("content-disposition", "")
    assert ".csv" in cd


def test_export_after_effects_with_fps(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={
            "format": "after_effects",
            "grammar": "concert_visuals",
            "fps": 30.0,
        },
    )
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    # ExtendScript output starts with a comment block and contains layers.
    assert "//" in body or "app." in body
    assert ".jsx" in r.headers.get("content-disposition", "")


def test_export_midi_with_ticks(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={
            "format": "midi",
            "grammar": "concert_visuals",
            "ticks_per_beat": 960,
        },
    )
    assert r.status_code == 200
    # Standard MIDI files start with the "MThd" header chunk.
    assert r.content[:4] == b"MThd"


def test_export_osc_with_host_port(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={
            "format": "osc",
            "grammar": "concert_visuals",
            "osc_host": "10.0.0.1",
            "osc_port": 9001,
        },
    )
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    # The OSC bundle is JSON with host+port baked in.
    assert "10.0.0.1" in body
    assert "9001" in body


def test_export_filename_used_in_disposition(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={
            "format": "csv",
            "grammar": "concert_visuals",
            "filename": "my-mix",
        },
    )
    assert r.status_code == 200
    cd = r.headers.get("content-disposition", "")
    assert "my-mix.csv" in cd


def test_export_filename_sanitized(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={
            "format": "csv",
            "grammar": "concert_visuals",
            "filename": "../../etc/passwd",
        },
    )
    assert r.status_code == 200
    cd = r.headers.get("content-disposition", "")
    # Path separators must have been stripped.
    assert "/" not in cd.split("filename=", 1)[1]
    assert "\\" not in cd.split("filename=", 1)[1]


# ---------------------------------------------------------------------------
# 4xx cases.
# ---------------------------------------------------------------------------


def test_export_csv_includes_frame_number_column(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={"format": "csv", "grammar": "concert_visuals", "fps": 30},
    )
    assert r.status_code == 200
    body = r.content.decode("utf-8")
    header = body.splitlines()[0]
    assert "frame_number" in header
    # First data row's frame_number for time 0 must be 0.
    first_data = body.splitlines()[1].split(",")
    cols = header.split(",")
    frame_idx = cols.index("frame_number")
    assert first_data[frame_idx] == "0"


def test_export_unknown_format(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={"format": "nope", "grammar": "concert_visuals"},
    )
    assert r.status_code == 400
    assert "unknown format" in r.json()["detail"].lower()


def test_export_unknown_grammar(tmp_path):
    _plant_analysis(tmp_path)
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={"format": "csv", "grammar": "nope"},
    )
    assert r.status_code == 400
    assert "unknown grammar" in r.json()["detail"].lower()


def test_export_missing_analysis(tmp_path):
    # Don't plant anything.
    client = TestClient(create_app(storage_root=tmp_path))
    r = client.post(
        f"/api/songs/{SONG_ID}/analyses/{ANALYSIS_ID}/export",
        json={"format": "csv", "grammar": "concert_visuals"},
    )
    assert r.status_code == 404
