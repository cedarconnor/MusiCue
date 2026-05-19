"""Tests for musicue.visualize."""
from __future__ import annotations

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


def test_lane_renderer_envelope_draws_filled_shape() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from musicue.visualize.lanes import draw_envelope_lane

    track = _envelope_track()
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_envelope_lane(ax, track, t_now=2.0, window_sec=5.0)

    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) >= 1

    plt.close(fig)


def test_lane_renderer_envelope_skips_out_of_window() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from musicue.visualize.lanes import draw_envelope_lane

    track = _envelope_track()
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_envelope_lane(ax, track, t_now=20.0, window_sec=1.0)
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) == 0

    plt.close(fig)


def test_lane_renderer_step_draws_segments_and_current_outline() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from musicue.visualize.lanes import draw_step_lane

    track = _step_track()  # intro at t=0, chorus at t=2
    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_step_lane(ax, track, t_now=2.5, window_sec=6.0)

    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    big = [p for p in rects if p.get_width() > 0.5]
    assert len(big) >= 2

    has_outline = any(
        (p.get_edgecolor() and len(p.get_edgecolor()) == 4 and p.get_edgecolor()[3] > 0)
        for p in big
    )
    assert has_outline

    labels = {t.get_text() for t in ax.texts}
    assert "intro" in labels and "chorus" in labels

    plt.close(fig)


def test_lane_renderer_ramp_draws_line() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from musicue.visualize.lanes import draw_ramp_lane

    track = _ramp_track()
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 1.0)

    draw_ramp_lane(ax, track, t_now=1.5, window_sec=5.0)

    lines = [a for a in ax.get_lines() if len(a.get_xdata()) >= 2]
    assert len(lines) >= 1
    xs = lines[0].get_xdata()
    ys = lines[0].get_ydata()
    assert ys[0] < ys[-1]
    assert min(xs) >= 1.0 - 1e-6
    assert max(xs) <= 2.0 + 1e-6

    plt.close(fig)


def test_lane_renderer_continuous_clips_to_window() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from musicue.schemas import CueTrack
    from musicue.visualize.lanes import draw_continuous_lane

    track = CueTrack(
        name="energy",
        type="continuous",
        timescale="macro",
        hop_sec=0.1,
        values=[float(i) for i in range(100)],
    )
    fig, ax = plt.subplots()
    ax.set_xlim(4.0, 6.0)
    ax.set_ylim(0.0, 1.0)

    draw_continuous_lane(ax, track, t_now=5.0, window_sec=2.0)

    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(polys) >= 1

    lines = ax.get_lines()
    assert lines
    xs = lines[0].get_xdata()
    assert min(xs) >= 4.0 - 1e-6
    assert max(xs) <= 6.0 + 1e-6

    plt.close(fig)


def test_format_timecode_decimal() -> None:
    from musicue.visualize.lanes import format_timecode

    assert format_timecode(0.0) == "00:00:00.0"
    assert format_timecode(65.5) == "00:01:05.5"
    assert format_timecode(3661.25) == "01:01:01.2"


def test_bpm_at_returns_closest_tempo() -> None:
    from musicue.visualize.lanes import bpm_at

    tempo_map = [
        {"t": 0.0, "bpm": 120.0},
        {"t": 30.0, "bpm": 140.0},
        {"t": 60.0, "bpm": 100.0},
    ]
    assert bpm_at(tempo_map, 0.0) == 120.0
    assert bpm_at(tempo_map, 15.0) == 120.0
    assert bpm_at(tempo_map, 45.0) == 140.0
    assert bpm_at(tempo_map, 100.0) == 100.0


def test_bpm_at_empty_returns_none() -> None:
    from musicue.visualize.lanes import bpm_at

    assert bpm_at([], 5.0) is None


def test_current_section_label_from_step_track(full_cuesheet) -> None:
    from musicue.visualize.lanes import current_section_label

    assert current_section_label(full_cuesheet, t_now=1.0) == "intro"
    assert current_section_label(full_cuesheet, t_now=6.0) == "chorus"


def test_current_section_label_missing_track() -> None:
    from musicue.schemas import CueSheet
    from musicue.visualize.lanes import current_section_label

    empty = CueSheet(
        source_sha256="x", grammar="g", duration_sec=1.0, tracks=[]
    )
    assert current_section_label(empty, t_now=0.0) is None


def test_compose_frame_writes_png(tmp_path, full_cuesheet) -> None:
    from musicue.visualize.cue_video import compose_frame

    out = tmp_path / "frame.png"
    compose_frame(
        cuesheet=full_cuesheet,
        t_now=2.0,
        out_path=out,
        width=640,
        height=360,
        window_sec=4.0,
    )
    assert out.exists()
    assert out.stat().st_size > 1000


# ---- Integration tests (require ffmpeg/ffprobe on PATH) ----


def _have_ffmpeg() -> bool:
    import shutil as _sh
    return _sh.which("ffmpeg") is not None and _sh.which("ffprobe") is not None


def _ffprobe(path) -> dict:
    import json
    import subprocess
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_format", "-show_streams",
        "-print_format", "json",
        str(path),
    ], text=True)
    return json.loads(out)


@pytest.fixture()
def short_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(str(p), sig, sr)
    return p


@pytest.mark.integration
def test_render_smoke_video(tmp_path, full_cuesheet, short_wav) -> None:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not on PATH")

    from musicue.visualize import render_cue_video

    out = tmp_path / "smoke.mp4"
    full_cuesheet.duration_sec = 2.0
    render_cue_video(
        full_cuesheet, short_wav, out,
        fps=10.0, width=240, height=120, window_sec=2.0,
        workers=1,
    )
    assert out.exists()
    meta = _ffprobe(out)
    streams = {s["codec_type"] for s in meta["streams"]}
    assert "video" in streams and "audio" in streams
    video_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    nb_frames = int(video_stream.get("nb_frames", "0"))
    assert 18 <= nb_frames <= 22


@pytest.mark.integration
def test_render_default_fps_is_2997(tmp_path, full_cuesheet, short_wav) -> None:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not on PATH")

    from musicue.visualize import render_cue_video

    out = tmp_path / "fps.mp4"
    full_cuesheet.duration_sec = 2.0
    render_cue_video(
        full_cuesheet, short_wav, out,
        width=240, height=120, workers=1,
    )
    meta = _ffprobe(out)
    video_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    assert video_stream["r_frame_rate"] == "30000/1001"
    assert video_stream["codec_name"] == "h264"
    assert meta["format"]["format_name"].startswith("mov,mp4")


@pytest.mark.integration
def test_render_uses_audio_duration(tmp_path, full_cuesheet, short_wav, caplog) -> None:
    if not _have_ffmpeg():
        pytest.skip("ffmpeg/ffprobe not on PATH")

    from musicue.visualize import render_cue_video

    out = tmp_path / "dur.mp4"
    full_cuesheet.duration_sec = 5.0
    with caplog.at_level("WARNING"):
        render_cue_video(
            full_cuesheet, short_wav, out,
            fps=10.0, width=240, height=120, workers=1,
        )
    assert "disagrees" in caplog.text
    meta = _ffprobe(out)
    duration = float(meta["format"]["duration"])
    assert 1.8 <= duration <= 2.2


# ---- CLI tests ----


def test_cli_auto_discovers_audio_in_run_dir(
    tmp_path, full_cuesheet, short_wav, monkeypatch
) -> None:
    """When --audio is omitted, the CLI should look for source.* siblings."""
    import shutil as _sh
    from pathlib import Path

    from typer.testing import CliRunner

    from musicue.cli import app

    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    cs_path = run_dir / "cuesheet.json"
    cs_path.write_text(full_cuesheet.model_dump_json())
    source = run_dir / "source.wav"
    _sh.copy(short_wav, source)

    captured = {}

    def fake_render(cuesheet, audio_path, out_path, **kw):
        captured["audio"] = Path(audio_path)
        captured["out"] = Path(out_path)
        Path(out_path).write_bytes(b"FAKE")

    monkeypatch.setattr("musicue.cli.render_cue_video", fake_render)
    runner = CliRunner()
    result = runner.invoke(app, ["cue-video", str(cs_path)])
    assert result.exit_code == 0, result.stdout
    assert captured["audio"] == source
    assert captured["out"] == cs_path.with_suffix(".mp4")


def test_cli_errors_when_audio_not_found(tmp_path, full_cuesheet) -> None:
    from typer.testing import CliRunner

    from musicue.cli import app

    cs_path = tmp_path / "cuesheet.json"
    cs_path.write_text(full_cuesheet.model_dump_json())
    runner = CliRunner()
    result = runner.invoke(app, ["cue-video", str(cs_path)])
    # Missing audio + no sibling source.* should exit non-zero.
    assert result.exit_code != 0


def test_render_command_invokes_cue_video_by_default(
    tmp_path, monkeypatch, short_wav
) -> None:
    """musicue render should call render_cue_video unless --no-cue-video."""
    from pathlib import Path

    from typer.testing import CliRunner

    from musicue.cli import app
    from musicue.schemas import (
        AnalysisConfig,
        AnalysisResult,
        CueSheet,
        CueTrack,
        SourceInfo,
    )

    src = SourceInfo(
        path=str(short_wav), sha256="deadbeef",
        duration_sec=2.0, sample_rate=44100,
    )
    stub_analysis = AnalysisResult(
        source=src, analysis_config=AnalysisConfig(), stems={},
    )

    def fake_run_analysis(*a, **kw):
        return stub_analysis

    def fake_compile(*a, **kw):
        return CueSheet(
            source_sha256="deadbeef", grammar="concert_visuals",
            duration_sec=2.0,
            tracks=[CueTrack(
                name="x", type="impulse", timescale="micro",
                events=[{
                    "t": 1.0, "strength": 1.0,
                    "envelope": {"a": 0.01, "d": 0.1, "s": 0.0, "r": 0.0},
                }],
            )],
        )

    called = {"n": 0, "out_path": None}

    def fake_render(cuesheet, audio_path, out_path, **kw):
        called["n"] += 1
        called["out_path"] = Path(out_path)
        Path(out_path).write_bytes(b"FAKE")

    import musicue.exporters.csv as csv_exporter

    def fake_export(cs, out):
        Path(out).write_text("time,value\n")

    monkeypatch.setattr(csv_exporter, "export", fake_export)
    monkeypatch.setattr("musicue.cli.run_analysis", fake_run_analysis, raising=False)
    monkeypatch.setattr("musicue.cli.compile_analysis", fake_compile, raising=False)
    monkeypatch.setattr("musicue.cli.render_cue_video", fake_render)

    out_path = tmp_path / "out.csv"
    runner = CliRunner()
    result = runner.invoke(app, [
        "render", str(short_wav),
        "--target", "csv",
        "--out", str(out_path),
    ])
    assert result.exit_code == 0, result.stdout
    assert called["n"] == 1
    assert called["out_path"] == out_path.with_name("cue_video.mp4")


def test_render_command_skips_cue_video_when_flag(
    tmp_path, monkeypatch, short_wav
) -> None:
    from pathlib import Path

    from typer.testing import CliRunner

    from musicue.cli import app
    from musicue.schemas import (
        AnalysisConfig,
        AnalysisResult,
        CueSheet,
        SourceInfo,
    )

    src = SourceInfo(
        path=str(short_wav), sha256="x",
        duration_sec=2.0, sample_rate=44100,
    )
    stub = AnalysisResult(source=src, analysis_config=AnalysisConfig(), stems={})

    def fake_run_analysis(*a, **kw):
        return stub

    def fake_compile(*a, **kw):
        return CueSheet(
            source_sha256="x", grammar="g",
            duration_sec=2.0, tracks=[],
        )

    def fake_render(*a, **kw):
        raise AssertionError("render_cue_video should not be called")

    import musicue.exporters.csv as csv_exporter

    monkeypatch.setattr(
        csv_exporter, "export",
        lambda cs, out: Path(out).write_text(""),
    )
    monkeypatch.setattr("musicue.cli.run_analysis", fake_run_analysis, raising=False)
    monkeypatch.setattr("musicue.cli.compile_analysis", fake_compile, raising=False)
    monkeypatch.setattr("musicue.cli.render_cue_video", fake_render)

    runner = CliRunner()
    result = runner.invoke(app, [
        "render", str(short_wav),
        "--target", "csv", "--out", str(tmp_path / "o.csv"),
        "--no-cue-video",
    ])
    assert result.exit_code == 0, result.stdout
