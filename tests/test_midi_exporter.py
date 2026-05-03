import pytest

mido = pytest.importorskip("mido")

from musicue.exporters.midi import export  # noqa: E402


def test_midi_export_creates_file(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_midi_export_is_valid_midi(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) >= 1


def test_midi_export_has_correct_tempo(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    tempo_msgs = [m for t in mid.tracks for m in t if m.type == "set_tempo"]
    assert len(tempo_msgs) >= 1
    # 120 BPM = 500000 microseconds per beat
    assert tempo_msgs[0].tempo == pytest.approx(500000, abs=5000)


def test_midi_export_has_note_events(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    note_ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert len(note_ons) >= 3  # at least the 3 kick events


def test_midi_export_continuous_as_cc(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    cc_msgs = [m for t in mid.tracks for m in t if m.type == "control_change"]
    assert len(cc_msgs) > 0  # energy track -> CC messages


def test_midi_export_step_as_text_marker(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    markers = [m for t in mid.tracks for m in t if m.type == "marker"]
    labels = [m.text for m in markers]
    assert any("intro" in lbl or "chorus" in lbl for lbl in labels)


def test_midi_export_continuous_downsamples_to_10hz(tmp_path):
    """Continuous tracks finer than 10 Hz should NOT emit at source rate."""
    from musicue.exporters.midi import export
    from musicue.schemas import CueSheet, CueTrack

    # 100 Hz source: 1000 samples over 10 seconds at hop_sec=0.01
    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=10.0,
        tempo_map=[{"t": 0.0, "bpm": 120.0}],
        tracks=[
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.01,
                values=[float(i) / 1000 - 0.5 for i in range(1000)],
            ),
        ],
    )
    out = tmp_path / "cuesheet.mid"
    export(cs, out)
    mid = mido.MidiFile(str(out))
    cc_msgs = [m for t in mid.tracks for m in t if m.type == "control_change"]
    # 10 seconds at 10 Hz target should give ~100 CC messages, not 1000.
    assert 80 <= len(cc_msgs) <= 120, f"Expected ~100 CC messages, got {len(cc_msgs)}"


def test_midi_continuous_normalized_input_does_not_saturate(tmp_path):
    """Continuous values already in [0, 1] should NOT all map to CC 127."""
    from musicue.exporters.midi import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=2.0,
        tempo_map=[{"t": 0.0, "bpm": 120.0}],
        tracks=[
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.1,
                values=[0.0, 0.25, 0.5, 0.75, 1.0] + [0.0] * 15,  # 20 values, 2s at hop=0.1
            ),
        ],
    )
    out = tmp_path / "cuesheet.mid"
    export(cs, out)
    mid = mido.MidiFile(str(out))
    cc_msgs = [m for t in mid.tracks for m in t if m.type == "control_change"]
    cc_values = [m.value for m in cc_msgs]
    # With auto-rescale, [0, 0.25, 0.5, 0.75, 1.0] maps to [0, 31, 63, 95, 127]
    assert min(cc_values) == 0
    assert max(cc_values) == 127
    # Not all 127 -- the saturation bug should not recur
    assert len(set(cc_values)) > 1


def test_midi_continuous_lufs_range_input_works(tmp_path):
    """Continuous LUFS values in [-70, 0] should still rescale to full CC range."""
    from musicue.exporters.midi import export
    from musicue.schemas import CueSheet, CueTrack

    cs = CueSheet(
        source_sha256="x",
        grammar="g",
        duration_sec=2.0,
        tempo_map=[{"t": 0.0, "bpm": 120.0}],
        tracks=[
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.1,
                values=[-70.0, -50.0, -30.0, -10.0, 0.0] + [-30.0] * 15,
            ),
        ],
    )
    out = tmp_path / "cuesheet.mid"
    export(cs, out)
    mid = mido.MidiFile(str(out))
    cc_msgs = [m for t in mid.tracks for m in t if m.type == "control_change"]
    cc_values = [m.value for m in cc_msgs]
    assert min(cc_values) == 0
    assert max(cc_values) == 127
