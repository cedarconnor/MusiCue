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
