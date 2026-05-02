"""Tests for musicue.analysis.transcription (Basic Pitch wrapper).

``basic-pitch`` is an optional dependency under the ``models`` extra. The
production code lazy-imports it inside ``transcribe_stem``, so we stub the
``basic_pitch`` and ``basic_pitch.inference`` modules into ``sys.modules``
before patching ``predict``. This keeps the unit tests focused on the
adapter logic (note dict shape, velocity clamping, sort order) and runs
without basic-pitch actually installed.
"""
import sys
import types
from unittest.mock import patch

import pytest

from musicue.analysis.transcription import basic_pitch_version, transcribe_stem


@pytest.fixture
def fake_basic_pitch():
    """Insert stub ``basic_pitch`` and ``basic_pitch.inference`` modules.

    Yields the inference submodule so individual tests can do
    ``patch.object(fake_basic_pitch, 'predict', return_value=...)``.
    """
    bp = types.ModuleType("basic_pitch")
    bp_inf = types.ModuleType("basic_pitch.inference")
    bp.ICASSP_2022_MODEL_PATH = "fake/path/icassp.pth"  # type: ignore[attr-defined]
    bp_inf.predict = lambda *a, **kw: ({}, None, [])  # type: ignore[attr-defined]
    prior = {k: sys.modules.get(k) for k in ("basic_pitch", "basic_pitch.inference")}
    sys.modules["basic_pitch"] = bp
    sys.modules["basic_pitch.inference"] = bp_inf
    try:
        yield bp_inf
    finally:
        for k, v in prior.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _mock_bp_output():
    # basic_pitch returns (model_output, midi_data, note_events)
    # note_events is a list of (start_time, end_time, pitch, amplitude, pitch_bends)
    note_events = [
        (0.5, 1.0, 64, 0.8, None),
        (1.5, 2.0, 67, 0.7, None),
        (2.5, 3.0, 60, 0.9, None),
    ]
    return ({}, None, note_events)


def test_basic_pitch_version_returns_string():
    v = basic_pitch_version()
    assert isinstance(v, str)


def test_transcribe_returns_midi_list(synthetic_wav, fake_basic_pitch):
    with patch.object(fake_basic_pitch, "predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    assert isinstance(notes, list)
    assert len(notes) == 3


def test_transcribe_note_fields(synthetic_wav, fake_basic_pitch):
    with patch.object(fake_basic_pitch, "predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    n = notes[0]
    assert "t" in n and "duration" in n and "pitch" in n and "velocity" in n
    assert n["t"] == pytest.approx(0.5)
    assert n["pitch"] == 64
    assert 0 < n["velocity"] <= 127


def test_transcribe_notes_sorted_by_time(synthetic_wav, fake_basic_pitch):
    with patch.object(fake_basic_pitch, "predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    times = [n["t"] for n in notes]
    assert times == sorted(times)
