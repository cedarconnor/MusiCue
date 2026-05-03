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
