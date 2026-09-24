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


def test_energy_curve_falls_back_to_note_velocity():
    notes = [
        {"t": 0.0, "duration": 0.2, "pitch": 60, "velocity": 64},
        {"t": 0.2, "duration": 0.2, "pitch": 62, "velocity": 127},
    ]
    (phrase,) = group_into_phrases(notes, gap_sec=0.6)
    curve = phrase["energy_curve"]
    assert curve["values"], "energy_curve must not be empty"
    assert max(curve["values"]) == pytest.approx(1.0)
    assert min(curve["values"]) == pytest.approx(64 / 127)


def test_energy_curve_uses_stem_rms_over_phrase_span():
    hop = 0.04
    # 0.2 RMS for 2 s, then 0.4 (the stem's loudest level).
    rms = {"hop_sec": hop, "values": [0.2] * 50 + [0.4] * 50}
    notes = _notes([(0.0, 60), (0.3, 62)]) + _notes([(2.4, 60), (2.7, 62)])
    phrases = group_into_phrases(notes, gap_sec=0.6, rms_curve=rms)
    assert len(phrases) == 2
    quiet, loud = phrases
    assert quiet["energy_curve"]["hop_sec"] == pytest.approx(hop)
    # Span 0.0..0.6 s → 15 hops.
    assert len(quiet["energy_curve"]["values"]) == 15
    assert max(quiet["energy_curve"]["values"]) == pytest.approx(0.5)
    assert max(loud["energy_curve"]["values"]) == pytest.approx(1.0)


def test_vocal_phrase_envelope_strength_nonzero():
    """The concert_visuals vocal_phrase track scores max(energy_curve)."""
    from musicue.compile.scoring import compute_score

    (phrase,) = group_into_phrases(_notes([(0.0, 60), (0.3, 62)]), gap_sec=0.6)
    score = compute_score({"base": "max(energy_curve)"}, phrase)
    assert score > 0.0
