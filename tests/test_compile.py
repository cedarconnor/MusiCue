import pytest
from musicue.schemas import (
    AnalysisConfig, AnalysisResult, CueSheet, OnsetEvent, SourceInfo, TimedCurve,
)
from musicue.compile.compiler import compile_analysis


def _make_analysis(onsets=None, lufs_values=None) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={"drums": onsets or []},
        curves={"lufs": TimedCurve(hop_sec=0.04, values=lufs_values or ([-20.0] * 250))},
    )


def test_compile_returns_cuesheet():
    cs = compile_analysis(_make_analysis(), grammar="concert_visuals")
    assert isinstance(cs, CueSheet)
    assert cs.grammar == "concert_visuals"
    assert cs.duration_sec == pytest.approx(10.0)


def test_compile_drums_impulse_track():
    onsets = [OnsetEvent(t=0.5, strength=0.9), OnsetEvent(t=1.0, strength=0.8)]
    cs = compile_analysis(_make_analysis(onsets=onsets))
    drum_tracks = [t for t in cs.tracks if t.name == "drums"]
    assert len(drum_tracks) == 1
    assert drum_tracks[0].type == "impulse"
    assert len(drum_tracks[0].events) == 2
    assert drum_tracks[0].events[0]["t"] == pytest.approx(0.5)
    assert drum_tracks[0].events[0]["strength"] == pytest.approx(0.9)


def test_compile_drum_event_has_envelope():
    onsets = [OnsetEvent(t=0.5, strength=0.9)]
    cs = compile_analysis(_make_analysis(onsets=onsets))
    env = cs.tracks[0].events[0]["envelope"]
    assert "a" in env and "d" in env and "s" in env and "r" in env


def test_compile_energy_continuous_track():
    cs = compile_analysis(_make_analysis(lufs_values=[-20.0] * 100))
    energy_tracks = [t for t in cs.tracks if t.name == "energy"]
    assert len(energy_tracks) == 1
    assert energy_tracks[0].type == "continuous"
    assert energy_tracks[0].hop_sec is not None
    assert len(energy_tracks[0].values) == 100


def test_compile_carries_source_sha256():
    cs = compile_analysis(_make_analysis())
    assert cs.source_sha256 == "abc"


def test_compile_empty_onsets_produces_no_drums_track():
    cs = compile_analysis(_make_analysis(onsets=[]))
    assert all(t.name != "drums" for t in cs.tracks)
