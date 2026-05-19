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


def test_track_color_is_deterministic() -> None:
    from musicue.visualize.colors import track_color

    a = track_color("kick")
    b = track_color("kick")
    assert a == b
    assert len(a) == 3
    assert all(0.0 <= c <= 1.0 for c in a)


def test_track_color_differs_between_tracks() -> None:
    from musicue.visualize.colors import track_color

    assert track_color("kick") != track_color("snare")


def test_section_palette_known_labels() -> None:
    from musicue.visualize.colors import section_palette

    assert section_palette("intro")[2] > section_palette("intro")[0]  # blueish
    assert section_palette("verse")[1] > section_palette("verse")[0]  # greenish
    assert section_palette("CHORUS")[0] > 0.7  # yellow has high red
    assert section_palette("verse_2") == section_palette("verse")  # fuzzy match


def test_section_palette_unknown_label_is_grey() -> None:
    from musicue.visualize.colors import section_palette

    r, g, b = section_palette("interlude")
    assert abs(r - g) < 0.1 and abs(g - b) < 0.1  # roughly grey


def test_type_tint_known_types() -> None:
    from musicue.visualize.colors import type_tint

    assert type_tint("impulse") != type_tint("envelope")
    assert type_tint("ramp") != type_tint("continuous")
    for t in ("impulse", "envelope", "step", "ramp", "continuous"):
        r, g, b = type_tint(t)
        assert (r + g + b) / 3 < 0.4
