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


def test_ae_export_handles_special_chars_in_track_name(tmp_path):
    """Track names with spaces, dots, or other JS-invalid chars must produce valid JSX."""
    from musicue.exporters.aftereffects import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=5.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick.snare-hat 1",  # dots, hyphen, space, digit
                type="impulse",
                timescale="micro",
                events=[{
                    "t": 0.5,
                    "strength": 0.9,
                    "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                    "tags": [],
                }],
            ),
        ],
    )
    out = tmp_path / "cuesheet.jsx"
    export(cs, out)
    content = out.read_text(encoding="utf-8")
    # Variable names must not contain dots or spaces
    assert "var layer_kick.snare" not in content
    assert "var layer_kick snare" not in content
    # Layer display name (a string) is allowed to keep the original
    assert 'MusiCue_kick.snare-hat 1' in content or 'MusiCue_kick.snare-hat 1' in content


def test_ae_export_handles_leading_digit_track_name(tmp_path):
    from musicue.exporters.aftereffects import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=5.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="808",  # starts with digit
                type="impulse",
                timescale="micro",
                events=[{
                    "t": 0.5,
                    "strength": 0.9,
                    "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                    "tags": [],
                }],
            ),
        ],
    )
    out = tmp_path / "cuesheet.jsx"
    export(cs, out)
    content = out.read_text(encoding="utf-8")
    # Should be `var layer__808` (leading underscore)
    assert "var layer__808" in content
    # Original number must NOT appear as a variable identifier prefix
    assert "var layer_808" not in content


def test_ae_continuous_normalized_input_does_not_saturate(tmp_path):
    """Continuous values in [0, 1] should NOT all map to slider 100."""
    from musicue.exporters.aftereffects import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=1.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.2,
                values=[0.0, 0.25, 0.5, 0.75, 1.0],
            ),
        ],
    )
    out = tmp_path / "cuesheet.jsx"
    export(cs, out)
    content = out.read_text(encoding="utf-8")
    # Should see varying slider values, not all 100.00
    import re
    energy_block_match = re.search(
        r"// Track: energy.*?(?=// Track:|app\.endUndoGroup)",
        content, re.DOTALL,
    )
    assert energy_block_match, "Energy track block not found"
    energy_block = energy_block_match.group(0)
    energy_calls = re.findall(r"setValueAtTime\([\d.]+,\s*([\d.]+)\)", energy_block)
    energy_values = [float(v) for v in energy_calls]
    # 5 values rescaled: should span 0..100, not all be 100
    assert min(energy_values) == 0.0
    assert max(energy_values) == 100.0
    assert len(set(energy_values)) > 1  # not flatlined
