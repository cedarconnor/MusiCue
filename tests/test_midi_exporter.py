import mido
import pytest

from musicue.exporters.midi import export


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
