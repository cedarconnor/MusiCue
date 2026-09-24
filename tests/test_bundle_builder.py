from pathlib import Path

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


def make_analysis_fixture(
    audio_path: Path | None = None,
    sha: str | None = None,
    sections=None,
    duration_sec: float = 10.0,
) -> AnalysisResult:
    """Build a synthetic AnalysisResult.

    If ``audio_path`` is given, sha256 is computed from the file so the
    result can pair with a CedarToy folder export that copies that audio.
    Otherwise, ``sha`` (or the default "a"*64) is used.
    """
    if audio_path is not None:
        from musicue.ui.storage import sha256_of_file
        sha256 = sha256_of_file(audio_path)
        path_str = str(audio_path)
    else:
        sha256 = sha or "a" * 64
        path_str = "x.wav"
    return AnalysisResult(
        source=SourceInfo(
            path=path_str,
            sha256=sha256,
            duration_sec=duration_sec,
            sample_rate=44100,
        ),
        analysis_config=AnalysisConfig(),
        stems={},
        tempo=TempoInfo(bpm_global=120.0),
        sections=sections or [],
    )


def make_cuesheet_fixture(
    source_sha256: str = "a" * 64,
    duration_sec: float = 10.0,
) -> CueSheet:
    return CueSheet(
        source_sha256=source_sha256,
        grammar="concert_visuals",
        duration_sec=duration_sec,
    )


# Backwards-compatible aliases used by tests below.
def _analysis(sha: str = "a" * 64, sections=None) -> AnalysisResult:
    return make_analysis_fixture(sha=sha, sections=sections)


def _cuesheet(sha: str = "a" * 64) -> CueSheet:
    return make_cuesheet_fixture(source_sha256=sha)


def test_sha_cross_check_raises_on_mismatch():
    with pytest.raises(ValueError, match="sha"):
        build_bundle(_analysis(sha="a" * 64), _cuesheet(sha="b" * 64))


def test_empty_analysis_yields_minimal_bundle():
    bundle = build_bundle(_analysis(), _cuesheet())
    assert bundle.schema_version == "1.2"
    assert bundle.duration_sec == 10.0
    assert bundle.sections == []
    assert bundle.drums == {}


from musicue.schemas import MidiNote, OnsetEvent, TimedCurve


def test_global_energy_normalized_from_lufs_curve():
    analysis = _analysis()
    analysis.curves = {"lufs": TimedCurve(hop_sec=0.04, values=[-30.0, -20.0, -10.0, 0.0])}

    bundle = build_bundle(analysis, _cuesheet())

    assert bundle.global_energy.hop_sec == 0.04
    assert bundle.global_energy.values[0] == 0.0
    assert bundle.global_energy.values[-1] == 1.0


def test_global_energy_empty_when_no_lufs_curve():
    bundle = build_bundle(_analysis(), _cuesheet())
    assert bundle.global_energy.values == []


def test_cuesheet_embedded_verbatim():
    cs = _cuesheet()
    cs.grammar = "lighting"
    bundle = build_bundle(_analysis(), cs)

    assert bundle.cuesheet.grammar == "lighting"
    assert bundle.cuesheet.source_sha256 == cs.source_sha256


def test_midi_notes_passed_through():
    analysis = _analysis()
    analysis.midi = {
        "vocals": [
            MidiNote(t=0.0, duration=0.5, pitch=60, velocity=80),
            MidiNote(t=1.0, duration=0.25, pitch=64, velocity=100),
        ]
    }
    bundle = build_bundle(analysis, _cuesheet())

    assert "vocals" in bundle.midi
    assert len(bundle.midi["vocals"]) == 2
    assert bundle.midi["vocals"][0].pitch == 60
    assert bundle.midi["vocals"][1].velocity == 100


def test_midi_energy_curve_derived_per_stem():
    analysis = _analysis()
    analysis.midi = {
        "vocals": [MidiNote(t=0.0, duration=1.0, pitch=60, velocity=127)],
    }
    bundle = build_bundle(analysis, _cuesheet())

    energy = bundle.midi_energy["vocals"]
    assert energy.hop_sec == 0.04
    expected_bins = int(10.0 / 0.04)
    assert len(energy.values) == expected_bins
    assert energy.values[0] > 0.95
    assert energy.values[24] > 0.95
    assert energy.values[30] < 0.05


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


def test_unclassified_drums_emit_warning(caplog):
    """If drums onsets exist but none have drum_class, warn loudly.

    Regression hook for the silent-empty-drums case we hit when the CNN
    checkpoint was missing (drum_classifier_version="not_trained").
    """
    import logging

    analysis = _analysis()
    analysis.onsets = {
        "drums": [
            OnsetEvent(t=0.1, strength=1.0, drum_class=None),
            OnsetEvent(t=0.5, strength=0.8, drum_class=None),
        ]
    }
    with caplog.at_level(logging.WARNING, logger="musicue.compile.bundle"):
        build_bundle(analysis, _cuesheet())

    assert any("ZERO classified" in m for m in caplog.messages)


def test_no_warning_when_drums_classified(caplog):
    import logging

    analysis = _analysis()
    analysis.onsets = {
        "drums": [OnsetEvent(t=0.1, strength=1.0, drum_class="kick")],
    }
    with caplog.at_level(logging.WARNING, logger="musicue.compile.bundle"):
        build_bundle(analysis, _cuesheet())

    assert not any("ZERO classified" in m for m in caplog.messages)


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


# ---- Robust energy normalization ----


def test_global_energy_ignores_silent_intro():
    """Silent intro (-70) + quiet (-30) + loud (-10): the silence floor must
    not define 0, so quiet reads low and loud reads high."""
    analysis = _analysis()
    values = [-70.0] * 100 + [-30.0] * 100 + [-10.0] * 100
    analysis.curves = {"lufs": TimedCurve(hop_sec=0.04, values=values)}

    energy = build_bundle(analysis, _cuesheet()).global_energy.values

    assert max(energy[:100]) == 0.0
    assert max(energy[100:200]) < 0.3
    assert min(energy[200:]) > 0.7
    assert all(0.0 <= v <= 1.0 for v in energy)


def _section_analysis(lufs_by_section: list[float], transitions=None) -> AnalysisResult:
    from musicue.schemas import SectionTransition

    sections = [
        SectionEvent(start=i * 4.0, end=(i + 1) * 4.0, label=f"s{i}", confidence=0.9)
        for i in range(len(lufs_by_section))
    ]
    analysis = _analysis(sections=sections)
    values: list[float] = []
    for lufs in lufs_by_section:
        # Each section: a short silent gap then its level (100 frames = 4 s).
        values += [-70.0] * 10 + [lufs] * 90
    analysis.curves = {"lufs": TimedCurve(hop_sec=0.04, values=values)}
    if transitions:
        analysis.section_transitions = [
            SectionTransition.model_validate({
                "t": i * 4.0, "from": f"s{i - 1}", "to": f"s{i}",
                "ramp": {"t_start": i * 4.0 - 1.2, "t_end": i * 4.0, "shape": "ease_in"},
                "ramp_evidence": {"spectral_flux_rise": rise, "lufs_rise_db": 0.0},
            })
            for i, rise in transitions.items()
        ]
    return analysis


def test_section_energy_rank_is_loudness_only():
    # First section is the loudest; transitions carry high flux rise that
    # must not leak into the rank (it used to be averaged with dB values).
    analysis = _section_analysis([-8.0, -20.0, -14.0], transitions={1: 0.9, 2: 0.1})
    sections = build_bundle(analysis, _cuesheet()).sections

    ranks = [s.energy_rank for s in sections]
    assert ranks[0] == pytest.approx(1.0)
    assert ranks[1] == pytest.approx(0.0)
    assert ranks[2] == pytest.approx(0.5)
    # Silent frames excluded from the section mean.
    assert sections[0].lufs == pytest.approx(-8.0)
    # Passthrough of the transition evidence is kept.
    assert sections[0].spectral_flux_rise is None
    assert sections[1].spectral_flux_rise == pytest.approx(0.9)


# ---- Per-stem energy ----


def test_stems_energy_populated_from_rms_curves():
    analysis = _analysis()
    quiet, loud = 0.01, 0.1  # -40 dBFS, -20 dBFS
    analysis.curves = {
        "rms_drums": TimedCurve(hop_sec=0.04, values=[0.0] * 50 + [quiet] * 50 + [loud] * 50),
        # A much quieter stem still spans 0..1 over its own range.
        "rms_vocals": TimedCurve(hop_sec=0.04, values=[0.002] * 50 + [0.02] * 50),
    }
    stems = build_bundle(analysis, _cuesheet()).stems_energy

    assert set(stems) == {"drums", "vocals"}
    drums = stems["drums"]
    assert drums.hop_sec == 0.04
    assert len(drums.values) == 150
    assert max(drums.values[:50]) == 0.0
    assert max(drums.values[50:100]) < 0.3
    assert min(drums.values[100:]) > 0.7
    assert min(stems["vocals"].values[50:]) > 0.7
    assert all(0.0 <= v <= 1.0 for c in stems.values() for v in c.values)


def test_stems_energy_empty_without_rms_curves():
    assert build_bundle(_analysis(), _cuesheet()).stems_energy == {}


# ---- MIDI energy ----


def test_midi_energy_uses_max_not_sum_over_polyphony():
    analysis = _analysis()
    # A three-note chord at velocity 64 (~0.5): summing saturated at 1.0.
    analysis.midi = {
        "other": [
            MidiNote(t=0.0, duration=1.0, pitch=p, velocity=64) for p in (60, 64, 67)
        ] + [MidiNote(t=0.0, duration=1.0, pitch=72, velocity=32)],
    }
    energy = build_bundle(analysis, _cuesheet()).midi_energy["other"].values
    assert energy[5] == pytest.approx(64 / 127)
    # Partial overlap still weights by coverage of the bin.
    analysis.midi = {"other": [MidiNote(t=0.02, duration=1.0, pitch=60, velocity=127)]}
    energy = build_bundle(analysis, _cuesheet()).midi_energy["other"].values
    assert energy[0] == pytest.approx(0.5)
