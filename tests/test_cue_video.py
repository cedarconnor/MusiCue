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


def test_plan_frames_ntsc_rate() -> None:
    from musicue.visualize.cue_video import plan_frames

    # 2.05 s at 29.97 fps (30000/1001) -> ~61.45 frames -> 61 rounded
    n = plan_frames(duration_sec=2.05, fps=30000 / 1001)
    assert n == 61


def test_plan_frames_clean_30_fps() -> None:
    from musicue.visualize.cue_video import plan_frames

    assert plan_frames(duration_sec=2.0, fps=30.0) == 60
    assert plan_frames(duration_sec=10.0, fps=24.0) == 240


def test_frame_time_ntsc() -> None:
    from musicue.visualize.cue_video import frame_time

    fps = 30000 / 1001
    # Frame 30 at 29.97 -> 30 * 1001 / 30000 = 1.001 s
    assert frame_time(30, fps) == pytest.approx(1.001, abs=1e-6)


def test_events_in_window_returns_intersecting_only() -> None:
    from musicue.visualize.cue_video import events_in_window

    events = [
        {"t": 0.0}, {"t": 1.0}, {"t": 2.0}, {"t": 3.0}, {"t": 10.0},
    ]
    sliced = events_in_window(events, t_now=1.5, window_sec=2.0, key="t")
    assert [e["t"] for e in sliced] == [1.0, 2.0]


def test_spanning_events_in_window() -> None:
    from musicue.visualize.cue_video import spanning_events_in_window

    events = [
        {"t_start": 0.0, "t_end": 0.4},   # before window
        {"t_start": 0.5, "t_end": 2.5},   # straddles window
        {"t_start": 1.0, "t_end": 2.0},   # fully inside
        {"t_start": 10.0, "t_end": 11.0}, # after window
    ]
    sliced = spanning_events_in_window(
        events, t_now=1.5, window_sec=2.0,
        start_key="t_start", end_key="t_end",
    )
    assert len(sliced) == 2
    assert sliced[0]["t_start"] == 0.5
    assert sliced[1]["t_start"] == 1.0


def test_events_in_window_uses_bisect_on_large_input() -> None:
    from musicue.visualize.cue_video import events_in_window

    events = [{"t": i * 0.01} for i in range(10000)]  # 0..99.99 s
    sliced = events_in_window(events, t_now=50.0, window_sec=0.2, key="t")
    ts = [e["t"] for e in sliced]
    assert all(49.9 <= t <= 50.1 for t in ts)
    assert len(ts) == 21


def _impulse_track(events_t):
    from musicue.schemas import CueTrack

    env = {"a": 0.02, "d": 0.20, "s": 0.0, "r": 0.0}
    return CueTrack(
        name="kick",
        type="impulse",
        timescale="micro",
        events=[
            {"t": t, "strength": 1.0, "envelope": env} for t in events_t
        ],
    )


def _envelope_track():
    from musicue.schemas import CueTrack

    return CueTrack(
        name="vocal_phrase",
        type="envelope",
        timescale="meso",
        events=[
            {
                "t_start": 1.0,
                "t_end": 3.0,
                "strength": 1.0,
                "envelope": {"a": 0.2, "d": 0.2, "s": 0.7, "r": 0.5},
            }
        ],
    )


def _step_track():
    from musicue.schemas import CueTrack

    return CueTrack(
        name="section_change",
        type="step",
        timescale="macro",
        events=[
            {"t": 0.0, "value": 1, "label": "intro"},
            {"t": 2.0, "value": 2, "label": "chorus"},
        ],
    )


def _ramp_track():
    from musicue.schemas import CueTrack

    return CueTrack(
        name="section_ramp",
        type="ramp",
        timescale="macro",
        events=[
            {
                "t_start": 1.0,
                "t_end": 2.0,
                "from": 0.0,
                "to": 1.0,
                "shape": "linear",
                "label": "intro->chorus",
            }
        ],
    )


def _continuous_track():
    from musicue.schemas import CueTrack

    return CueTrack(
        name="energy",
        type="continuous",
        timescale="macro",
        hop_sec=0.5,
        values=[0.0, 0.5, 1.0],
    )


def test_fire_brightness_impulse_zero_before_event() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _impulse_track([1.0])
    assert fire_brightness(track, t_now=0.5) == 0.0


def test_fire_brightness_impulse_peaks_at_attack_end() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _impulse_track([1.0])
    val = fire_brightness(track, t_now=1.02)
    assert val == pytest.approx(1.0, abs=1e-2)


def test_fire_brightness_envelope_pulses_at_t_start() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _envelope_track()
    assert fire_brightness(track, t_now=0.5) == 0.0
    assert fire_brightness(track, t_now=1.2) == pytest.approx(1.0, abs=1e-2)


def test_fire_brightness_step_flashes_after_label_change() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _step_track()
    assert fire_brightness(track, t_now=1.9) == 0.0
    assert fire_brightness(track, t_now=2.0) == pytest.approx(1.0, abs=1e-2)
    assert fire_brightness(track, t_now=2.5) == 0.0


def test_fire_brightness_ramp_glows_during_active_window() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _ramp_track()
    assert fire_brightness(track, t_now=1.5) == pytest.approx(0.5, abs=1e-2)


def test_fire_brightness_continuous_tracks_value() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _continuous_track()
    assert fire_brightness(track, t_now=0.5) == pytest.approx(0.5, abs=1e-2)
    assert fire_brightness(track, t_now=1.0) == pytest.approx(1.0, abs=1e-2)


def test_fire_brightness_continuous_outside_range_is_zero() -> None:
    from musicue.visualize.lanes import fire_brightness

    track = _continuous_track()
    assert fire_brightness(track, t_now=5.0) == 0.0


def test_lane_renderer_impulse_draws_ticks_and_flash() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from musicue.visualize.lanes import draw_impulse_lane

    track = _impulse_track([1.0, 1.5, 2.0])
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_impulse_lane(ax, track, t_now=1.02, window_sec=5.0)

    # Vertical ticks from axvline — find Line2D children with xs[0]==xs[1]
    tick_xs = []
    for line in ax.get_lines():
        xs = line.get_xdata()
        if len(xs) == 2 and xs[0] == xs[1]:
            tick_xs.append(float(xs[0]))
    tick_xs_sorted = sorted(tick_xs)
    assert len(tick_xs_sorted) >= 3
    assert tick_xs_sorted[:3] == pytest.approx([1.0, 1.5, 2.0], abs=0.01)

    # Flash overlay: a Rectangle with non-zero alpha
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert any(p.get_alpha() and p.get_alpha() > 0 for p in rects)

    plt.close(fig)


def test_lane_renderer_impulse_no_overlay_when_no_event_firing() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from musicue.visualize.lanes import draw_impulse_lane

    track = _impulse_track([10.0])  # event far in the future
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_impulse_lane(ax, track, t_now=1.0, window_sec=2.0)

    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert all((not p.get_alpha()) or p.get_alpha() < 0.01 for p in rects)

    plt.close(fig)
