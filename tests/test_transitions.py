import numpy as np
import pytest

from musicue.analysis.transitions import derive_transitions


def _make_sections():
    return [
        {"start": 0.0, "end": 17.2, "label": "intro", "confidence": 0.9, "timescale": "macro"},
        {"start": 17.2, "end": 51.6, "label": "verse", "confidence": 0.9, "timescale": "macro"},
        {"start": 51.6, "end": 86.0, "label": "chorus", "confidence": 0.9, "timescale": "macro"},
    ]


def _make_flux(hop_sec=0.04, n=2500):
    values = [0.1] * n
    # Add a rise before each transition
    for trans_t in (17.2, 51.6):
        idx = int(trans_t / hop_sec)
        for j in range(max(0, idx - 35), idx):
            values[j] = 0.8 + (j - (idx - 35)) * 0.005
    return {"hop_sec": hop_sec, "values": values}


def _make_lufs(hop_sec=0.04, n=2500):
    return {"hop_sec": hop_sec, "values": [-20.0] * n}


def test_derive_transitions_count(synthetic_wav):
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    # 2 transitions: intro→verse, verse→chorus
    assert len(transitions) == 2


def test_derive_transitions_fields():
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    t = transitions[0]
    assert "t" in t
    assert "from" in t or "from_section" in t
    assert "to" in t
    assert "ramp" in t
    assert "ramp_evidence" in t
    assert t["t"] == pytest.approx(17.2)


def test_derive_transitions_to_from_labels():
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    assert transitions[0]["to"] == "verse"
    assert transitions[1]["to"] == "chorus"


def test_derive_transitions_ramp_evidence_keys():
    transitions = derive_transitions(_make_sections(), _make_flux(), _make_lufs())
    ev = transitions[0]["ramp_evidence"]
    assert "spectral_flux_rise" in ev
    assert "lufs_rise_db" in ev


def test_no_sections_returns_empty():
    assert derive_transitions([], _make_flux(), _make_lufs()) == []


def _flux_steps(levels: list[tuple[float, float]], hop_sec=0.04, n=2500):
    """Flux curve that sits at ``level`` from each ``(t_from, level)`` on."""
    rng = np.random.default_rng(0)
    values = np.zeros(n)
    for t_from, level in levels:
        values[int(t_from / hop_sec):] = level
    values += 0.02 * rng.standard_normal(n)
    return {"hop_sec": hop_sec, "values": values.tolist()}


def test_spectral_flux_rise_is_informative():
    # intro (quiet) -> verse (busier) -> chorus (same as verse)
    flux = _flux_steps([(0.0, 0.2), (17.2, 1.0)])
    transitions = derive_transitions(_make_sections(), flux, _make_lufs())
    rise_into_verse = transitions[0]["ramp_evidence"]["spectral_flux_rise"]
    rise_into_chorus = transitions[1]["ramp_evidence"]["spectral_flux_rise"]
    assert rise_into_verse > 0.8
    assert rise_into_chorus == pytest.approx(0.5, abs=0.1)


def test_spectral_flux_rise_below_half_for_drop():
    flux = _flux_steps([(0.0, 1.0), (51.6, 0.2)])
    transitions = derive_transitions(_make_sections(), flux, _make_lufs())
    assert transitions[1]["ramp_evidence"]["spectral_flux_rise"] < 0.2
    for tr in transitions:
        assert 0.0 <= tr["ramp_evidence"]["spectral_flux_rise"] <= 1.0


def test_section_ramp_filters_follow_new_scale():
    """Grammar thresholds sit above 0.5 (= no change) after the rescale."""
    from musicue.compile.grammar import load_grammar
    from musicue.compile.scoring import evaluate_filter

    flat = {"ramp_evidence": {"spectral_flux_rise": 0.5}}
    rise = {"ramp_evidence": {"spectral_flux_rise": 0.9}}
    for name in ("concert_visuals", "camera_edit"):
        grammar = load_grammar(name)
        (track,) = [t for t in grammar.tracks if t.name == "section_ramp"]
        assert evaluate_filter(track.filter, rise) is True
        assert evaluate_filter(track.filter, flat) is False


# ---- Build-window ramps ----


def _lufs_steps(levels: list[tuple[float, float]], hop_sec=0.04, n=2500):
    values = np.full(n, -20.0)
    for t_from, level in levels:
        values[int(t_from / hop_sec):] = level
    return {"hop_sec": hop_sec, "values": values.tolist()}


def _grid(bar=2.0, until=100.0):
    return [i * bar for i in range(int(until / bar) + 1)]


def test_build_ramp_spans_eight_bars_on_the_beat_grid():
    sections = [
        {"start": 0.0, "end": 40.0, "label": "verse"},
        {"start": 40.0, "end": 80.0, "label": "chorus"},
    ]
    lufs = _lufs_steps([(0.0, -24.0), (40.0, -10.0)])
    (tr,) = derive_transitions(sections, _make_flux(), lufs, downbeats=_grid(), bpm=120.0)
    assert tr["ramp"]["t_start"] == pytest.approx(24.0)
    assert tr["ramp"]["t_end"] == pytest.approx(40.0)
    assert tr["ramp"]["shape"] == "ease_in"


def test_non_build_and_gridless_ramps_keep_short_default():
    sections = [
        {"start": 0.0, "end": 40.0, "label": "chorus"},
        {"start": 40.0, "end": 80.0, "label": "verse"},
    ]
    drop = _lufs_steps([(0.0, -10.0), (40.0, -24.0)])
    (tr,) = derive_transitions(sections, _make_flux(), drop, downbeats=_grid(), bpm=120.0)
    assert tr["ramp"]["t_start"] == pytest.approx(40.0 - 1.2)

    rise = _lufs_steps([(0.0, -24.0), (40.0, -10.0)])
    (tr,) = derive_transitions(sections, _make_flux(), rise)  # no grid supplied
    assert tr["ramp"]["t_start"] == pytest.approx(40.0 - 1.2)
