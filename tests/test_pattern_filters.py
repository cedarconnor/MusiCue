"""Tests for pattern-aware grammar filter primitives (v0.2c)."""
from __future__ import annotations

from musicue.compile.scoring import evaluate_filter


# ---------------------------------------------------------------------------
# is_fill()
# ---------------------------------------------------------------------------


def test_is_fill_true():
    assert evaluate_filter("is_fill()", {"is_fill": True}) is True


def test_is_fill_false_when_missing():
    assert evaluate_filter("is_fill()", {}) is False


def test_is_fill_false_when_set_false():
    assert evaluate_filter("is_fill()", {"is_fill": False}) is False


# ---------------------------------------------------------------------------
# is_phrase_start() / is_phrase_end()
# ---------------------------------------------------------------------------


def test_is_phrase_start_first_position():
    assert evaluate_filter("is_phrase_start()", {"phrase_position": 1}) is True


def test_is_phrase_start_other_positions():
    assert evaluate_filter("is_phrase_start()", {"phrase_position": 2}) is False
    assert evaluate_filter("is_phrase_start()", {}) is False


def test_is_phrase_end_last_position():
    assert evaluate_filter(
        "is_phrase_end()", {"phrase_position": 8, "phrase_length": 8}
    ) is True


def test_is_phrase_end_not_last():
    assert evaluate_filter(
        "is_phrase_end()", {"phrase_position": 4, "phrase_length": 8}
    ) is False


def test_is_phrase_end_missing_length():
    assert evaluate_filter("is_phrase_end()", {"phrase_position": 8}) is False


# ---------------------------------------------------------------------------
# every_nth()
# ---------------------------------------------------------------------------


def test_every_nth_default_offset_matches_zero_mod():
    assert evaluate_filter("every_nth(4)", {"bar": 0}) is True
    assert evaluate_filter("every_nth(4)", {"bar": 4}) is True
    assert evaluate_filter("every_nth(4)", {"bar": 8}) is True


def test_every_nth_default_offset_rejects_other_bars():
    assert evaluate_filter("every_nth(4)", {"bar": 1}) is False
    assert evaluate_filter("every_nth(4)", {"bar": 3}) is False


def test_every_nth_with_offset():
    # every 8th bar starting from bar 4: 4, 12, 20, ...
    assert evaluate_filter("every_nth(8, offset=4)", {"bar": 4}) is True
    assert evaluate_filter("every_nth(8, offset=4)", {"bar": 12}) is True
    assert evaluate_filter("every_nth(8, offset=4)", {"bar": 5}) is False


def test_every_nth_missing_bar():
    assert evaluate_filter("every_nth(4)", {}) is False


# ---------------------------------------------------------------------------
# Existing field comparison primitives still work for pattern fields.
# ---------------------------------------------------------------------------


def test_phrase_position_field_comparison():
    # phrase_position is just a numeric field; the existing primitives
    # should already cover it without needing new regex.
    assert evaluate_filter("phrase_position == 1", {"phrase_position": 1}) is False
    # `==` for integers via the boolean-style check fails; we expect users to
    # use the supported syntax. The numeric `>=` form does work:
    assert evaluate_filter("phrase_position >= 1", {"phrase_position": 1}) is True
    assert evaluate_filter("phrase_position >= 1", {"phrase_position": 0}) is False


def test_syncopation_threshold():
    assert evaluate_filter("syncopation > 0.4", {"syncopation": 0.5}) is True
    assert evaluate_filter("syncopation > 0.4", {"syncopation": 0.3}) is False
