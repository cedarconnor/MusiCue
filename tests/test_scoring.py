import math

import pytest

from musicue.compile.scoring import (
    RarityTracker,
    compute_score,
    evaluate_filter,
)


def _event(drum_class=None, strength=0.8, labels=None, timescale="micro", is_downbeat=False):
    return {
        "t": 1.0,
        "strength": strength,
        "drum_class": drum_class,
        "timescale": timescale,
        "is_downbeat": is_downbeat,
        "labels": labels or [],
    }


# --- filter evaluation ---


def test_filter_drum_class_eq_true():
    assert evaluate_filter("drum_class == 'kick'", _event(drum_class="kick")) is True


def test_filter_drum_class_eq_false():
    assert evaluate_filter("drum_class == 'kick'", _event(drum_class="snare")) is False


def test_filter_is_downbeat_true():
    assert evaluate_filter("is_downbeat == true", _event(is_downbeat=True)) is True


def test_filter_is_downbeat_false():
    assert evaluate_filter("is_downbeat == true", _event(is_downbeat=False)) is False


def test_filter_none_matches_all():
    assert evaluate_filter(None, _event()) is True


def test_filter_any_label_match():
    event = _event(labels=[{"label": "sub bass drop", "score": 0.75, "source": "clap"}])
    assert evaluate_filter("any_label('sub bass drop', min_score=0.6)", event) is True


def test_filter_any_label_below_threshold():
    event = _event(labels=[{"label": "sub bass drop", "score": 0.4, "source": "clap"}])
    assert evaluate_filter("any_label('sub bass drop', min_score=0.6)", event) is False


# --- scoring ---


def test_compute_score_base_literal():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(strength=0.8), timescale_weight=1.0, rarity_bonus=1.0)
    assert s == pytest.approx(1.0)


def test_compute_score_base_field():
    score_cfg = {"base": "strength"}
    s = compute_score(score_cfg, _event(strength=0.75), timescale_weight=1.0, rarity_bonus=1.0)
    assert s == pytest.approx(0.75)


def test_compute_score_with_timescale_weight():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(), timescale_weight=1.5, rarity_bonus=1.0)
    assert s == pytest.approx(1.5)


def test_compute_score_with_rarity_bonus():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(), timescale_weight=1.0, rarity_bonus=0.5)
    assert s == pytest.approx(0.5)


# --- rarity tracker ---


def test_rarity_bonus_no_recent_events():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    bonus = tracker.bonus(t=5.0)
    assert bonus == pytest.approx(1.0)  # exp(0) = 1


def test_rarity_bonus_one_recent_event():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    tracker.record(t=4.5)
    bonus = tracker.bonus(t=5.0)
    # one event in window -> exp(-1/4) ~ 0.778
    assert bonus == pytest.approx(math.exp(-1 / 4.0), abs=0.01)


def test_rarity_bonus_outside_window():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    tracker.record(t=3.0)  # 2s ago - outside 1s window
    bonus = tracker.bonus(t=5.0)
    assert bonus == pytest.approx(1.0)
