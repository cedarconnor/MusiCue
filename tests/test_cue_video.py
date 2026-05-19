"""Tests for musicue.visualize."""
from __future__ import annotations

import math

import pytest


def test_sample_adsr_attack_rises_linearly() -> None:
    from musicue.visualize.envelopes import sample_adsr

    env = {"a": 0.1, "d": 0.0, "s": 1.0, "r": 0.0}
    assert sample_adsr(env, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert sample_adsr(env, 0.05) == pytest.approx(0.5, abs=1e-6)
    assert sample_adsr(env, 0.1) == pytest.approx(1.0, abs=1e-6)


def test_sample_adsr_decay_falls_to_sustain() -> None:
    from musicue.visualize.envelopes import sample_adsr

    env = {"a": 0.0, "d": 0.2, "s": 0.4, "r": 0.0}
    assert sample_adsr(env, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert sample_adsr(env, 0.1) == pytest.approx(0.7, abs=1e-6)
    assert sample_adsr(env, 0.2) == pytest.approx(0.4, abs=1e-6)


def test_sample_adsr_after_release_is_zero() -> None:
    from musicue.visualize.envelopes import sample_adsr

    env = {"a": 0.01, "d": 0.05, "s": 0.5, "r": 0.05}
    assert sample_adsr(env, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_sample_adsr_envelope_value_outside_clamps_to_zero() -> None:
    from musicue.visualize.envelopes import sample_adsr

    env = {"a": 0.01, "d": 0.05, "s": 0.0, "r": 0.0}
    assert sample_adsr(env, -1.0) == 0.0
    assert sample_adsr(env, 99.0) == 0.0


def test_ease_linear() -> None:
    from musicue.visualize.envelopes import ease

    assert ease("linear", 0.0) == 0.0
    assert ease("linear", 0.5) == 0.5
    assert ease("linear", 1.0) == 1.0


def test_ease_in_out_symmetric_about_half() -> None:
    from musicue.visualize.envelopes import ease

    assert ease("ease_in_out", 0.25) == pytest.approx(0.125, abs=1e-6)
    assert ease("ease_in_out", 0.5) == pytest.approx(0.5, abs=1e-6)
    assert ease("ease_in_out", 0.75) == pytest.approx(0.875, abs=1e-6)


def test_ease_unknown_shape_falls_back_to_linear() -> None:
    from musicue.visualize.envelopes import ease

    assert ease("nonsense", 0.5) == 0.5
