"""Tests for frame/timecode population on analysis and cuesheet events."""
from __future__ import annotations

from musicue.compile.compiler import compile_analysis
from musicue.frame_population import (
    populate_analysis_frames,
    populate_cuesheet_frames,
)
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    BeatEvent,
    CueSheet,
    CueTrack,
    OnsetEvent,
    SectionEvent,
    SourceInfo,
    TimedCurve,
)


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="x.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0"),
        stems={"drums": "stems/drums.wav"},
        beats=[
            BeatEvent(t=0.0, beat_in_bar=1, bar=0, is_downbeat=True, confidence=1.0),
            BeatEvent(t=0.5, beat_in_bar=2, bar=0, is_downbeat=False, confidence=1.0),
            BeatEvent(t=1.0, beat_in_bar=3, bar=0, is_downbeat=False, confidence=1.0),
        ],
        sections=[
            SectionEvent(start=0.0, end=4.0, label="intro", confidence=0.9),
            SectionEvent(start=4.0, end=10.0, label="verse", confidence=0.85),
        ],
        onsets={
            "drums": [
                OnsetEvent(t=0.5, strength=0.8, drum_class="kick"),
                OnsetEvent(t=1.0, strength=0.7, drum_class="snare"),
            ]
        },
        curves={"lufs": TimedCurve(hop_sec=0.04, values=[-20.0] * 250)},
    )


# ---------------------------------------------------------------------------
# Analysis stamping
# ---------------------------------------------------------------------------


def test_populate_analysis_stamps_beats_at_24fps():
    a = populate_analysis_frames(_make_analysis(), fps=24.0)
    # 0.5s at 24 fps = frame 12 → 00:00:00:12
    assert a.beats[1].frame == 12
    assert a.beats[1].timecode == "00:00:00:12"


def test_populate_analysis_stamps_beats_at_30fps():
    a = populate_analysis_frames(_make_analysis(), fps=30.0)
    # 0.5s at 30 fps = frame 15
    assert a.beats[1].frame == 15
    assert a.beats[1].timecode == "00:00:00:15"


def test_populate_analysis_stamps_section_endpoints():
    a = populate_analysis_frames(_make_analysis(), fps=24.0)
    assert a.sections[0].frame_start == 0
    assert a.sections[0].frame_end == 96  # 4s * 24
    assert a.sections[0].timecode_start == "00:00:00:00"
    assert a.sections[0].timecode_end == "00:00:04:00"


def test_populate_analysis_stamps_onsets():
    a = populate_analysis_frames(_make_analysis(), fps=24.0)
    assert a.onsets["drums"][0].frame == 12
    assert a.onsets["drums"][1].timecode == "00:00:01:00"


def test_populate_analysis_writes_config_fps():
    a = populate_analysis_frames(_make_analysis(), fps=30.0)
    assert a.analysis_config.fps == 30.0
    assert a.analysis_config.drop_frame is False


def test_populate_analysis_drop_frame():
    a = populate_analysis_frames(_make_analysis(), fps=29.97, drop_frame=True)
    # DF timecodes use semicolon before final field.
    for b in a.beats:
        assert ";" in b.timecode


# ---------------------------------------------------------------------------
# Cuesheet stamping
# ---------------------------------------------------------------------------


def test_populate_cuesheet_stamps_impulse_events():
    cs = CueSheet(
        source_sha256="abc",
        grammar="concert_visuals",
        duration_sec=10.0,
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.8, "envelope": {}, "tags": []},
                    {"t": 1.0, "strength": 0.7, "envelope": {}, "tags": []},
                ],
            )
        ],
    )
    out = populate_cuesheet_frames(cs, fps=24.0)
    assert out.fps == 24.0
    assert out.tracks[0].events[0]["frame"] == 12
    assert out.tracks[0].events[0]["timecode"] == "00:00:00:12"


def test_populate_cuesheet_stamps_ramp_events():
    cs = CueSheet(
        source_sha256="abc",
        grammar="x",
        duration_sec=10.0,
        tracks=[
            CueTrack(
                name="ramp",
                type="ramp",
                timescale="macro",
                events=[
                    {
                        "t_start": 1.0,
                        "t_end": 2.0,
                        "from": 0.0,
                        "to": 1.0,
                        "shape": "ease_in",
                        "label": "intro->verse",
                    },
                ],
            )
        ],
    )
    out = populate_cuesheet_frames(cs, fps=24.0)
    ev = out.tracks[0].events[0]
    assert ev["frame_start"] == 24
    assert ev["frame_end"] == 48
    assert ev["timecode_start"] == "00:00:01:00"
    assert ev["timecode_end"] == "00:00:02:00"


# ---------------------------------------------------------------------------
# Compile end-to-end with fps override
# ---------------------------------------------------------------------------


def test_compile_inherits_fps_from_analysis_config():
    a = populate_analysis_frames(_make_analysis(), fps=30.0)
    cs = compile_analysis(a, grammar="concert_visuals")
    assert cs.fps == 30.0


def test_compile_fps_override_recomputes():
    a = populate_analysis_frames(_make_analysis(), fps=24.0)
    cs = compile_analysis(a, grammar="concert_visuals", fps=30.0)
    assert cs.fps == 30.0
    # Spot-check: any event in any track should have frames at 30fps now.
    for tr in cs.tracks:
        for ev in tr.events:
            if "t" in ev:
                expected = round(ev["t"] * 30.0)
                assert ev["frame"] == expected
