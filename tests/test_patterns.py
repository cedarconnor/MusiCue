"""Tests for beat-pattern detection (phrase blocks, fills, syncopation)."""
from __future__ import annotations

from musicue.analysis.patterns import (
    detect_patterns,
    populate_beat_pattern_fields,
)
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    BeatEvent,
    OnsetEvent,
    SectionEvent,
    SourceInfo,
)


def _bars_of_4_beats(n_bars: int, bar_dur: float = 2.0) -> list[BeatEvent]:
    """A simple 4/4 beat track: 4 beats per bar, downbeat on beat 1."""
    beats: list[BeatEvent] = []
    for bar in range(n_bars):
        for b_in_bar in range(4):
            t = bar * bar_dur + (b_in_bar / 4.0) * bar_dur
            beats.append(
                BeatEvent(
                    t=t,
                    beat_in_bar=b_in_bar + 1,
                    bar=bar,
                    is_downbeat=(b_in_bar == 0),
                    confidence=1.0,
                )
            )
    return beats


def _make_analysis(
    beats=None,
    onsets=None,
    sections=None,
) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="x.wav", sha256="abc", duration_sec=100.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4"),
        stems={"drums": "stems/drums.wav"},
        beats=beats or [],
        sections=sections or [],
        onsets={"drums": onsets or []},
    )


# ---------------------------------------------------------------------------
# Phrase detection
# ---------------------------------------------------------------------------


def test_phrase_detection_finds_4_bar_repeat():
    """Synthesize 16 bars with a clean 4-bar repeating onset pattern."""
    beats = _bars_of_4_beats(16)
    onsets = []
    for bar in range(16):
        # Every bar gets a kick on beat 1 (low onset count). Every 4th bar
        # gets a fill (lots of onsets). Period = 4.
        bar_t = bar * 2.0
        onsets.append(OnsetEvent(t=bar_t, strength=1.0, drum_class="kick"))
        if (bar + 1) % 4 == 0:  # bars 3, 7, 11, 15
            for k in range(8):
                onsets.append(OnsetEvent(t=bar_t + 0.1 + k * 0.2, strength=0.8))
    analysis = _make_analysis(beats=beats, onsets=onsets)

    patterns = detect_patterns(analysis)
    assert patterns.bar_count == 16
    assert len(patterns.phrases) >= 1
    # The dominant period should be 4.
    most_common_length = max(p.length for p in patterns.phrases)
    assert most_common_length == 4


def test_phrase_detection_handles_no_beats():
    analysis = _make_analysis()
    patterns = detect_patterns(analysis)
    assert patterns.bar_count == 0
    assert patterns.phrases == []


def test_phrase_detection_per_section():
    """Two sections — phrase blocks should respect section boundaries."""
    beats = _bars_of_4_beats(16)
    onsets = [OnsetEvent(t=bar * 2.0, strength=1.0, drum_class="kick") for bar in range(16)]
    sections = [
        SectionEvent(start=0.0, end=16.0, label="intro", confidence=1.0),
        SectionEvent(start=16.0, end=32.0, label="verse", confidence=1.0),
    ]
    analysis = _make_analysis(beats=beats, onsets=onsets, sections=sections)

    patterns = detect_patterns(analysis)
    section_labels = {p.section_label for p in patterns.phrases}
    assert section_labels == {"intro", "verse"}


# ---------------------------------------------------------------------------
# Fill detection
# ---------------------------------------------------------------------------


def test_fill_detection_finds_density_spike_at_phrase_end():
    """Bar 3 (end of a 4-bar phrase) has 10 onsets vs ~1 for other bars."""
    beats = _bars_of_4_beats(8)
    onsets = []
    for bar in range(8):
        bar_t = bar * 2.0
        if bar in (3, 7):  # phrase-end bars
            for k in range(10):
                onsets.append(OnsetEvent(t=bar_t + 0.05 + k * 0.18, strength=0.8))
        else:
            onsets.append(OnsetEvent(t=bar_t, strength=1.0))
    analysis = _make_analysis(beats=beats, onsets=onsets)

    patterns = detect_patterns(analysis)
    fill_bars = {f.bar for f in patterns.fills}
    # At least one of the dense bars should be flagged.
    assert fill_bars & {3, 7}


def test_fill_density_zscore_positive():
    beats = _bars_of_4_beats(8)
    onsets = [OnsetEvent(t=bar * 2.0, strength=1.0) for bar in range(8)]
    onsets.extend(OnsetEvent(t=6.0 + k * 0.1, strength=0.8) for k in range(10))
    analysis = _make_analysis(beats=beats, onsets=onsets)

    patterns = detect_patterns(analysis)
    if patterns.fills:
        assert all(f.density_zscore > 0 for f in patterns.fills)


# ---------------------------------------------------------------------------
# Syncopation
# ---------------------------------------------------------------------------


def test_syncopation_zero_for_on_beat_only():
    """All onsets on beats → syncopation = 0."""
    beats = _bars_of_4_beats(4)
    onsets = [OnsetEvent(t=b.t, strength=1.0) for b in beats]
    analysis = _make_analysis(beats=beats, onsets=onsets)

    patterns = detect_patterns(analysis)
    assert all(s == 0.0 for s in patterns.syncopation_per_bar)


def test_syncopation_high_for_off_beat_heavy():
    """Onsets land between beats → high syncopation."""
    beats = _bars_of_4_beats(4)
    onsets = []
    for bar in range(4):
        bar_t = bar * 2.0
        # 8 evenly-spaced offbeats, each landing ~halfway between beats.
        for k in range(8):
            onsets.append(OnsetEvent(t=bar_t + 0.125 + k * 0.25, strength=1.0))
    analysis = _make_analysis(beats=beats, onsets=onsets)

    patterns = detect_patterns(analysis)
    avg_sync = sum(patterns.syncopation_per_bar) / max(1, len(patterns.syncopation_per_bar))
    assert avg_sync > 0.4


# ---------------------------------------------------------------------------
# Beat-field population
# ---------------------------------------------------------------------------


def test_populate_beat_pattern_fields_sets_phrase_position():
    beats = _bars_of_4_beats(8)
    onsets = [OnsetEvent(t=b.t, strength=1.0) for b in beats]
    analysis = _make_analysis(beats=beats, onsets=onsets)

    out = populate_beat_pattern_fields(analysis)
    # Every beat in the same bar should share phrase_id and phrase_position.
    by_bar: dict[int, list[int | None]] = {}
    for b in out.beats:
        by_bar.setdefault(b.bar, []).append(b.phrase_position)
    for bar, positions in by_bar.items():
        # All beats in a bar should agree on phrase_position
        assert len(set(positions)) <= 1


def test_populate_beat_pattern_fields_sets_is_fill():
    beats = _bars_of_4_beats(8)
    onsets = []
    for bar in range(8):
        bar_t = bar * 2.0
        if bar == 3:
            for k in range(10):
                onsets.append(OnsetEvent(t=bar_t + 0.05 + k * 0.18, strength=0.8))
        else:
            onsets.append(OnsetEvent(t=bar_t, strength=1.0))
    analysis = _make_analysis(beats=beats, onsets=onsets)

    out = populate_beat_pattern_fields(analysis)
    fill_bar3 = [b.is_fill for b in out.beats if b.bar == 3]
    other_bars = [b.is_fill for b in out.beats if b.bar != 3]
    assert any(fill_bar3)
    assert not any(other_bars)


def test_populate_beat_pattern_fields_backwards_compat_when_no_data():
    """Empty analysis → empty patterns, no exceptions."""
    analysis = _make_analysis()
    out = populate_beat_pattern_fields(analysis)
    assert out.patterns is not None
    assert out.patterns.bar_count == 0
