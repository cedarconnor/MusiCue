import pytest

from musicue.analysis.phrases import group_into_phrases


def _notes(times_and_pitches):
    return [
        {"t": t, "duration": 0.3, "pitch": p, "velocity": 80}
        for t, p in times_and_pitches
    ]


def test_single_phrase_no_gaps():
    notes = _notes([(0.0, 60), (0.3, 62), (0.6, 64)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert len(phrases) == 1
    p = phrases[0]
    assert p["t_start"] == pytest.approx(0.0)
    assert p["note_count"] == 3


def test_gap_splits_into_two_phrases():
    notes = _notes([(0.0, 60), (0.3, 62), (2.0, 67), (2.3, 69)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert len(phrases) == 2
    assert phrases[0]["note_count"] == 2
    assert phrases[1]["t_start"] == pytest.approx(2.0)


def test_phrase_pitch_features():
    notes = _notes([(0.0, 60), (0.3, 67), (0.6, 64)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    p = phrases[0]
    assert p["pitch_peak"] == 67
    assert p["pitch_low"] == 60
    assert len(p["pitch_contour"]) > 0


def test_phrase_timescale():
    notes = _notes([(0.0, 60), (0.3, 62)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert phrases[0]["timescale"] == "meso"


def test_phrase_t_end():
    notes = _notes([(0.0, 60), (0.5, 62)])
    # note at 0.5 with duration 0.3 ends at 0.8
    notes[1]["duration"] = 0.3
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert phrases[0]["t_end"] == pytest.approx(0.8)


def test_empty_notes_returns_empty():
    assert group_into_phrases([], gap_sec=0.6) == []
