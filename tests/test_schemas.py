import pytest
from musicue.schemas import (
    AnalysisResult, SourceInfo, AnalysisConfig, TimedCurve,
    CueSheet, CueTrack, ADSREnvelope, BeatEvent, OnsetEvent,
)


def _minimal_analysis_dict():
    return {
        "schema_version": "1.1",
        "source": {"path": "song.wav", "sha256": "abc123", "duration_sec": 10.0, "sample_rate": 44100},
        "analysis_config": {"demucs_model": "htdemucs_ft", "demucs_version": "4.0.1"},
        "stems": {"drums": "stems/drums.wav"},
    }


def test_analysis_result_validates():
    result = AnalysisResult.model_validate(_minimal_analysis_dict())
    assert result.source.duration_sec == 10.0
    assert result.schema_version == "1.1"
    assert result.beats == []
    assert result.onsets == {}


def test_analysis_result_roundtrip():
    result = AnalysisResult.model_validate(_minimal_analysis_dict())
    dumped = result.model_dump(mode="json")
    result2 = AnalysisResult.model_validate(dumped)
    assert result2.source.sha256 == "abc123"


def test_beat_event_timescale_defaults_to_micro():
    b = BeatEvent(t=0.5, beat_in_bar=1, bar=1, is_downbeat=True, confidence=0.9)
    assert b.timescale == "micro"


def test_onset_event_labels_default_empty():
    o = OnsetEvent(t=0.5, strength=0.8)
    assert o.labels == []
    assert o.drum_class is None


def test_cuesheet_roundtrip():
    cs = CueSheet(
        source_sha256="abc",
        grammar="test",
        duration_sec=10.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[{"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []}],
            )
        ],
    )
    dumped = cs.model_dump(mode="json")
    cs2 = CueSheet.model_validate(dumped)
    assert cs2.tracks[0].name == "kick"
    assert cs2.tracks[0].events[0]["t"] == pytest.approx(0.5)


def test_adsr_fields():
    env = ADSREnvelope(a=0.01, d=0.1, s=0.5, r=0.3)
    assert env.a == 0.01
    assert env.s == 0.5


def test_timed_curve():
    c = TimedCurve(hop_sec=0.04, values=[-20.0, -18.0, -22.0])
    assert len(c.values) == 3
    assert c.hop_sec == pytest.approx(0.04)
