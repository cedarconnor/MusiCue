"""Tests for musicue.analysis.structure (beat/downbeat/section detection).

Note: ``allin1`` is an optional dependency listed under the ``models`` extra in
pyproject.toml. It is not always installable on Windows, so these tests stub
``allin1`` into ``sys.modules`` before patching its ``analyze`` function. This
keeps the unit tests focused on branching logic in ``detect_structure`` (the
librosa fallback is exercised separately by forcing the allin1 path to raise).
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from musicue.analysis.structure import allin1_version, detect_structure


@pytest.fixture
def fake_allin1():
    """Insert a stub ``allin1`` module into sys.modules so patches can attach.

    Yields the stub module; ``patch.object(stub, 'analyze', ...)`` is the recommended
    way to control its behavior in individual tests.
    """
    existing = sys.modules.get("allin1")
    stub = types.ModuleType("allin1")
    stub.analyze = MagicMock()  # type: ignore[attr-defined]
    sys.modules["allin1"] = stub
    try:
        yield stub
    finally:
        if existing is not None:
            sys.modules["allin1"] = existing
        else:
            sys.modules.pop("allin1", None)


def test_allin1_version_returns_string():
    v = allin1_version()
    assert isinstance(v, str)


def test_detect_structure_returns_expected_keys(synthetic_wav, fake_allin1):
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = [0.5, 1.0, 1.5, 2.0]
    mock_result.downbeats = [0.5, 2.0]
    mock_result.segments = [
        MagicMock(start=0.0, end=5.0, label="intro"),
        MagicMock(start=5.0, end=10.0, label="verse"),
    ]

    with patch.object(fake_allin1, "analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    assert "tempo" in result
    assert "beats" in result
    assert "sections" in result
    assert result["tempo"]["bpm_global"] == pytest.approx(120.0)


def test_detect_structure_beat_fields(synthetic_wav, fake_allin1):
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = [0.5, 1.0]
    mock_result.downbeats = [0.5]
    mock_result.segments = []

    with patch.object(fake_allin1, "analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    beats = result["beats"]
    assert len(beats) == 2
    b = beats[0]
    assert "t" in b and "is_downbeat" in b and "timescale" in b and "confidence" in b
    assert b["is_downbeat"] is True
    assert b["timescale"] == "micro"


def test_detect_structure_section_fields(synthetic_wav, fake_allin1):
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = []
    mock_result.downbeats = []
    mock_result.segments = [MagicMock(start=0.0, end=5.0, label="chorus")]

    with patch.object(fake_allin1, "analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    sections = result["sections"]
    assert len(sections) == 1
    s = sections[0]
    assert s["label"] == "chorus"
    assert s["timescale"] == "macro"
    assert "start" in s and "end" in s


def test_detect_structure_falls_back_to_librosa_on_error(synthetic_wav, fake_allin1):
    with patch.object(fake_allin1, "analyze", side_effect=RuntimeError("allin1 failed")):
        result = detect_structure(synthetic_wav, backend="allin1")
    # Should not raise — should fall back to librosa
    assert "beats" in result
    assert "tempo" in result
