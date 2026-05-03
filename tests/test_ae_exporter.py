from musicue.exporters.aftereffects import export


def test_ae_export_creates_jsx_file(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_ae_export_is_valid_jsx(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "app.project" in content or "function" in content


def test_ae_export_contains_track_names(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "MusiCue_kick" in content
    assert "MusiCue_energy" in content


def test_ae_export_contains_composition_markers(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "marker" in content.lower() or "Marker" in content


def test_ae_export_contains_keyframes_for_continuous(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "setValue" in content or "setValueAtTime" in content
