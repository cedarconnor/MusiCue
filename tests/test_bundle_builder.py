import pytest

from musicue.compile.bundle import build_bundle
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    CueSheet,
    SectionEvent,
    SourceInfo,
    TempoInfo,
)


def _analysis(sha: str = "a" * 64, sections=None) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="x.wav", sha256=sha, duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(),
        stems={},
        tempo=TempoInfo(bpm_global=120.0),
        sections=sections or [],
    )


def _cuesheet(sha: str = "a" * 64) -> CueSheet:
    return CueSheet(source_sha256=sha, grammar="concert_visuals", duration_sec=10.0)


def test_sha_cross_check_raises_on_mismatch():
    with pytest.raises(ValueError, match="sha"):
        build_bundle(_analysis(sha="a" * 64), _cuesheet(sha="b" * 64))


def test_empty_analysis_yields_minimal_bundle():
    bundle = build_bundle(_analysis(), _cuesheet())
    assert bundle.schema_version == "1.0"
    assert bundle.duration_sec == 10.0
    assert bundle.sections == []
    assert bundle.drums == {}


from musicue.schemas import OnsetEvent


def test_drums_regrouped_by_drum_class():
    analysis = _analysis()
    analysis.onsets = {
        "drums": [
            OnsetEvent(t=0.5, strength=1.0, drum_class="kick"),
            OnsetEvent(t=0.6, strength=0.5, drum_class="snare"),
            OnsetEvent(t=0.7, strength=0.8, drum_class="kick"),
            OnsetEvent(t=0.8, strength=0.6, drum_class=None),
        ]
    }

    bundle = build_bundle(analysis, _cuesheet())

    assert set(bundle.drums.keys()) == {"kick", "snare"}
    assert len(bundle.drums["kick"]) == 2
    assert len(bundle.drums["snare"]) == 1
    assert bundle.drums["kick"][0].t == 0.5
    assert bundle.drums["kick"][0].strength == 1.0


def test_drums_missing_section_handled():
    analysis = _analysis()
    bundle = build_bundle(analysis, _cuesheet())
    assert bundle.drums == {}


def test_sections_get_normalized_energy_rank():
    sections = [
        SectionEvent(start=0.0, end=4.0, label="intro", confidence=0.9),
        SectionEvent(start=4.0, end=8.0, label="chorus", confidence=0.9),
        SectionEvent(start=8.0, end=10.0, label="outro", confidence=0.9),
    ]
    bundle = build_bundle(_analysis(sections=sections), _cuesheet())

    assert len(bundle.sections) == 3
    for s in bundle.sections:
        assert s.energy_rank == 0.5
        assert s.lufs is None
        assert s.spectral_flux_rise is None
