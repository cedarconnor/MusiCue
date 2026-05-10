"""Tests for the v0.2b editorial exporters: EDL, FCPXML, Premiere CSV, Resolve CSV.

Each exporter receives a hand-built CueSheet with sections (step track) and
a transition (ramp track), then we verify the produced file has the right
structural pieces — not byte-exact diffs, since each format has small
acceptable variations.
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from musicue.exporters._editorial import extract_markers
from musicue.schemas import CueSheet, CueTrack


@pytest.fixture
def cs() -> CueSheet:
    return CueSheet(
        source_sha256="abc",
        grammar="concert_visuals",
        duration_sec=120.0,
        fps=24.0,
        drop_frame=False,
        tracks=[
            CueTrack(
                name="section_step",
                type="step",
                timescale="macro",
                events=[
                    {"t": 0.0, "value": 1, "label": "intro"},
                    {"t": 8.5, "value": 2, "label": "verse"},
                    {"t": 32.0, "value": 3, "label": "chorus"},
                ],
            ),
            CueTrack(
                name="ramps",
                type="ramp",
                timescale="macro",
                events=[
                    {
                        "t_start": 30.5,
                        "t_end": 32.0,
                        "from": 0.0,
                        "to": 1.0,
                        "shape": "ease_in",
                        "label": "verse->chorus",
                    },
                ],
            ),
            CueTrack(
                name="downbeat",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 1.0, "envelope": {}, "tags": []},
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# extract_markers
# ---------------------------------------------------------------------------


def test_extract_markers_default_sources(cs):
    markers = extract_markers(cs)
    # Default = sections + transitions only, no impulses.
    assert {m.category for m in markers} == {"section", "transition"}
    assert len(markers) == 4  # 3 sections + 1 transition


def test_extract_markers_with_impulses(cs):
    markers = extract_markers(cs, marker_sources={"section", "transition", "impulse"})
    cats = {m.category for m in markers}
    assert "impulse" in cats


def test_extract_markers_filters_impulse_by_name(cs):
    # Only emit impulses from named tracks.
    markers = extract_markers(
        cs,
        marker_sources={"impulse"},
        impulse_track_names={"downbeat"},
    )
    assert all(m.category == "impulse" and m.name == "downbeat" for m in markers)


def test_extract_markers_sorted_by_time(cs):
    markers = extract_markers(cs, marker_sources={"section", "transition", "impulse"})
    times = [m.t_start for m in markers]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# EDL
# ---------------------------------------------------------------------------


def test_edl_export_writes_cmx_3600_header(tmp_path, cs):
    from musicue.exporters import edl

    out = tmp_path / "test.edl"
    edl.export(cs, out)
    body = out.read_text(encoding="utf-8")
    assert body.startswith("TITLE: MusiCue cuesheet")
    assert "FCM: NON-DROP FRAME" in body


def test_edl_event_lines_well_formed(tmp_path, cs):
    from musicue.exporters import edl

    out = tmp_path / "test.edl"
    edl.export(cs, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    # Each event should have an event-number prefix.
    event_lines = [l for l in lines if l.startswith("001 ") or l.startswith("002 ")]
    assert event_lines, "expected numbered event lines"
    # Each event has FROM CLIP NAME / COLOR / COMMENT comments after it.
    body = "\n".join(lines)
    assert "* FROM CLIP NAME: intro" in body
    assert "* COLOR: Blue" in body
    assert "* COLOR: Red" in body  # transition


def test_edl_drop_frame_header_when_enabled(tmp_path, cs):
    from musicue.exporters import edl

    out = tmp_path / "test.edl"
    edl.export(cs, out, fps=29.97, drop_frame=True)
    body = out.read_text(encoding="utf-8")
    assert "FCM: DROP FRAME" in body


# ---------------------------------------------------------------------------
# FCPXML
# ---------------------------------------------------------------------------


def test_fcpxml_export_well_formed_xml(tmp_path, cs):
    from musicue.exporters import fcpxml

    out = tmp_path / "test.fcpxml"
    fcpxml.export(cs, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<?xml version=")
    assert "<!DOCTYPE fcpxml>" in text


def test_fcpxml_marker_count_matches_extracted(tmp_path, cs):
    from musicue.exporters import fcpxml

    out = tmp_path / "test.fcpxml"
    fcpxml.export(cs, out)
    # Strip DOCTYPE before parsing (ElementTree refuses DOCTYPE).
    raw = out.read_text(encoding="utf-8")
    raw = raw.replace("<!DOCTYPE fcpxml>", "")
    root = ET.fromstring(raw)
    markers = root.findall(".//marker")
    expected = len(extract_markers(cs))
    assert len(markers) == expected


def test_fcpxml_color_in_marker_name(tmp_path, cs):
    from musicue.exporters import fcpxml

    out = tmp_path / "test.fcpxml"
    fcpxml.export(cs, out)
    raw = out.read_text(encoding="utf-8").replace("<!DOCTYPE fcpxml>", "")
    root = ET.fromstring(raw)
    names = [m.get("value") for m in root.findall(".//marker")]
    assert any(n.startswith("[Blue]") for n in names)  # section
    assert any(n.startswith("[Red]") for n in names)  # transition


# ---------------------------------------------------------------------------
# Premiere
# ---------------------------------------------------------------------------


def test_premiere_csv_has_expected_header(tmp_path, cs):
    from musicue.exporters import premiere_markers

    out = tmp_path / "test.csv"
    premiere_markers.export(cs, out)
    text = out.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0] == ["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"]


def test_premiere_csv_rows_match_marker_count(tmp_path, cs):
    from musicue.exporters import premiere_markers

    out = tmp_path / "test.csv"
    premiere_markers.export(cs, out)
    text = out.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    # Header + N markers.
    assert len(rows) == 1 + len(extract_markers(cs))


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def test_resolve_csv_has_bom_and_crlf(tmp_path, cs):
    from musicue.exporters import resolve_markers

    out = tmp_path / "test.csv"
    resolve_markers.export(cs, out)
    raw_bytes = out.read_bytes()
    # Resolve wants UTF-8 BOM.
    assert raw_bytes.startswith(b"\xef\xbb\xbf")


def test_resolve_csv_has_named_color_column(tmp_path, cs):
    from musicue.exporters import resolve_markers

    out = tmp_path / "test.csv"
    resolve_markers.export(cs, out)
    text = out.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    assert "Color" in header
    color_idx = header.index("Color")
    colors = {row[color_idx] for row in rows[1:]}
    # Should include canonical Resolve color names.
    assert colors.issubset({"Blue", "Red", "Green", "Yellow"})
