import pytest

from musicue.compile.compiler import compile_analysis
from musicue.compile.grammar import Grammar, GrammarTrack
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    CueSheet,
    OnsetEvent,
    SourceInfo,
    TimedCurve,
)


def _make_analysis(onsets=None, lufs_values=None) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={"drums": onsets or []},
        curves={"lufs": TimedCurve(hop_sec=0.04, values=lufs_values or ([-20.0] * 250))},
    )


def _make_grammar_with_kick() -> Grammar:
    return Grammar(
        name="test",
        hierarchy_weights={"macro": 1.5, "meso": 1.2, "micro": 0.8},
        tracks=[
            GrammarTrack(
                name="kick",
                type="impulse",
                source="onsets.drums",
                filter="drum_class == 'kick'",
                score={"base": "strength"},
                envelope={"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Sheet-level invariants (kept from M0 -- still hold under the new compiler).
# ---------------------------------------------------------------------------


def test_compile_returns_cuesheet():
    cs = compile_analysis(_make_analysis(), grammar="concert_visuals")
    assert isinstance(cs, CueSheet)
    assert cs.grammar == "concert_visuals"
    assert cs.duration_sec == pytest.approx(10.0)


def test_compile_carries_source_sha256():
    cs = compile_analysis(_make_analysis(), grammar="concert_visuals")
    assert cs.source_sha256 == "abc"


def test_compile_empty_onsets_produces_no_drums_track():
    cs = compile_analysis(_make_analysis(onsets=[]), grammar="concert_visuals")
    # Concert grammar emits per-class drum tracks (kick/snare); no generic "drums"
    # track exists in the new compiler regardless of input.
    assert all(t.name != "drums" for t in cs.tracks)


def test_compile_energy_continuous_track():
    # concert_visuals declares an "energy" continuous track sourced from
    # curves.lufs with EMA smoothing + percentile normalization.
    cs = compile_analysis(
        _make_analysis(lufs_values=[-20.0] * 100), grammar="concert_visuals"
    )
    energy_tracks = [t for t in cs.tracks if t.name == "energy"]
    assert len(energy_tracks) == 1
    energy = energy_tracks[0]
    assert energy.type == "continuous"
    assert energy.hop_sec is not None
    assert energy.values is not None
    assert len(energy.values) == 100
    # Smoothed + percentile-normalized values must lie within [0, 1].
    assert all(0.0 <= v <= 1.0 for v in energy.values)


# ---------------------------------------------------------------------------
# Replacements for the removed M0 "drums" track tests. The new compiler emits
# kick/snare tracks (gated by `drum_class`) rather than a single "drums" track.
# ---------------------------------------------------------------------------


def test_compile_kick_impulse_track_under_concert_grammar():
    onsets = [
        OnsetEvent(t=0.5, strength=0.9, drum_class="kick"),
        OnsetEvent(t=1.0, strength=0.8, drum_class="kick"),
    ]
    cs = compile_analysis(_make_analysis(onsets=onsets), grammar="concert_visuals")
    kick_tracks = [t for t in cs.tracks if t.name == "kick"]
    assert len(kick_tracks) == 1
    assert kick_tracks[0].type == "impulse"
    assert len(kick_tracks[0].events) == 2
    assert kick_tracks[0].events[0]["t"] == pytest.approx(0.5)


def test_compile_kick_event_has_envelope_under_concert_grammar():
    onsets = [OnsetEvent(t=0.5, strength=0.9, drum_class="kick")]
    cs = compile_analysis(_make_analysis(onsets=onsets), grammar="concert_visuals")
    kick = next(t for t in cs.tracks if t.name == "kick")
    env = kick.events[0]["envelope"]
    assert "a" in env and "d" in env and "s" in env and "r" in env


# ---------------------------------------------------------------------------
# Grammar-driven compiler behavior (new in M2).
# ---------------------------------------------------------------------------


def test_grammar_compiler_filters_by_drum_class():
    onsets = [
        OnsetEvent(t=0.5, strength=0.9, drum_class="kick"),
        OnsetEvent(t=1.0, strength=0.8, drum_class="snare"),
        OnsetEvent(t=1.5, strength=0.7, drum_class="kick"),
    ]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=_make_grammar_with_kick())
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    assert len(kick_track.events) == 2  # only kick onsets pass the filter


def test_grammar_compiler_applies_hierarchy_weight():
    # micro weight 0.8 -> kick scores lower than base
    onsets = [OnsetEvent(t=0.5, strength=1.0, drum_class="kick")]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=_make_grammar_with_kick())
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    # strength=1.0 * micro_weight=0.8 * rarity_bonus~1.0 ~= 0.8
    assert kick_track.events[0]["strength"] == pytest.approx(0.8, abs=0.05)


def test_grammar_compiler_cooldown_suppresses_close_events():
    grammar = Grammar(
        name="test",
        hierarchy_weights={"macro": 1.0, "meso": 1.0, "micro": 1.0},
        tracks=[
            GrammarTrack(
                name="drop",
                type="impulse",
                source="onsets.drums",
                score={"base": 1.0},
                envelope={"a": 0.05, "d": 0.4, "s": 0.6, "r": 1.5},
                cooldown_sec=5.0,
            )
        ],
    )
    onsets = [
        OnsetEvent(t=1.0, strength=0.9),
        OnsetEvent(t=2.0, strength=0.8),  # within 5s cooldown -> suppressed
        OnsetEvent(t=10.0, strength=0.9),  # outside cooldown -> emitted
    ]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=grammar)
    track = cs.tracks[0]
    assert len(track.events) == 2  # t=1.0 and t=10.0
    assert track.events[0]["t"] == pytest.approx(1.0)
    assert track.events[1]["t"] == pytest.approx(10.0)


def test_grammar_compiler_near_downbeat_multiplier_fires():
    # Setup: kick at t=1.0 with downbeat at t=1.0 -- should get the 1.2x boost
    from musicue.schemas import BeatEvent

    grammar = Grammar(
        name="test",
        hierarchy_weights={"macro": 1.0, "meso": 1.0, "micro": 1.0},
        tracks=[
            GrammarTrack(
                name="kick",
                type="impulse",
                source="onsets.drums",
                filter="drum_class == 'kick'",
                score={
                    "base": "strength",
                    "multiplier": [{"when": "near_downbeat(0.05)", "factor": 1.2}],
                },
                envelope={"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
            )
        ],
    )
    onsets = [OnsetEvent(t=1.0, strength=1.0, drum_class="kick")]
    analysis = _make_analysis(onsets=onsets)
    # Attach a downbeat at exactly t=1.0
    analysis.beats = [
        BeatEvent(t=1.0, beat_in_bar=1, bar=1, is_downbeat=True, confidence=0.9, timescale="micro"),
    ]
    cs = compile_analysis(analysis, grammar=grammar)
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    # strength=1.0 * 1.2 multiplier * 1.0 hierarchy * 1.0 rarity = 1.2
    assert kick_track.events[0]["strength"] == pytest.approx(1.2, abs=0.01)


def test_grammar_compiler_near_downbeat_multiplier_skips_when_far():
    from musicue.schemas import BeatEvent

    grammar = Grammar(
        name="test",
        hierarchy_weights={"macro": 1.0, "meso": 1.0, "micro": 1.0},
        tracks=[
            GrammarTrack(
                name="kick",
                type="impulse",
                source="onsets.drums",
                filter="drum_class == 'kick'",
                score={
                    "base": "strength",
                    "multiplier": [{"when": "near_downbeat(0.05)", "factor": 1.2}],
                },
                envelope={"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
            )
        ],
    )
    onsets = [OnsetEvent(t=1.5, strength=1.0, drum_class="kick")]  # 0.5s from downbeat
    analysis = _make_analysis(onsets=onsets)
    analysis.beats = [
        BeatEvent(t=1.0, beat_in_bar=1, bar=1, is_downbeat=True, confidence=0.9, timescale="micro"),
    ]
    cs = compile_analysis(analysis, grammar=grammar)
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    # strength=1.0 * 1.0 (no boost) = 1.0
    assert kick_track.events[0]["strength"] == pytest.approx(1.0, abs=0.01)


def test_compile_loads_default_grammar_from_any_cwd(tmp_path, monkeypatch):
    """Built-in grammar must load regardless of CWD."""
    monkeypatch.chdir(tmp_path)
    cs = compile_analysis(_make_analysis(), grammar="concert_visuals")
    assert cs.grammar == "concert_visuals"


def test_grammar_compiler_loads_from_file(tmp_path):
    import yaml

    grammar_data = {
        "name": "test_file_grammar",
        "hierarchy_weights": {"macro": 1.0, "meso": 1.0, "micro": 1.0},
        "tracks": [
            {
                "name": "drums",
                "type": "impulse",
                "source": "onsets.drums",
                "score": {"base": "strength"},
                "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
            }
        ],
    }
    grammar_file = tmp_path / "test_file_grammar.yaml"
    grammar_file.write_text(yaml.dump(grammar_data))
    analysis = _make_analysis(onsets=[OnsetEvent(t=0.5, strength=0.9)])
    cs = compile_analysis(analysis, grammar="test_file_grammar", grammars_dir=tmp_path)
    assert len(cs.tracks) >= 1
    assert cs.grammar == "test_file_grammar"
