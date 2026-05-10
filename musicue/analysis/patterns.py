"""Beat-pattern detection: phrase blocks, drum fills, syncopation.

This module is heuristic — no ML, no new dependencies. It runs after the
upstream beat / onset / section detectors have populated their respective
fields, and produces a `Patterns` block plus enrichment fields on every
beat (phrase_id, phrase_position, is_fill, syncopation).

Phrase detection uses simple bar-window autocorrelation on the per-bar
onset density. Fills are bars whose drum-onset count is >1.5σ above the
local mean *and* sit at the end of a phrase block. Syncopation is the
fraction of off-beat onset strength to total onset strength per bar.
"""
from __future__ import annotations

from collections import defaultdict

from musicue.schemas import (
    AnalysisResult,
    FillEvent,
    OnsetEvent,
    Patterns,
    PhraseBlock,
)

PHRASE_CANDIDATE_LENGTHS: tuple[int, ...] = (16, 8, 4)
FILL_ZSCORE_THRESHOLD = 1.5


def _bar_count(analysis: AnalysisResult) -> int:
    if not analysis.beats:
        return 0
    return max(b.bar for b in analysis.beats) + 1


def _drum_onsets(analysis: AnalysisResult) -> list[OnsetEvent]:
    return analysis.onsets.get("drums", [])


def _bar_t_ranges(analysis: AnalysisResult) -> list[tuple[float, float]]:
    """For each bar index, the [t_start, t_end) of that bar."""
    by_bar: dict[int, list[float]] = defaultdict(list)
    for b in analysis.beats:
        by_bar[b.bar].append(b.t)
    if not by_bar:
        return []
    bar_count = max(by_bar) + 1
    ranges: list[tuple[float, float]] = []
    for i in range(bar_count):
        beats = sorted(by_bar.get(i, []))
        if not beats:
            ranges.append((0.0, 0.0))
            continue
        t_start = beats[0]
        # t_end = first beat of next bar, or last beat + median bar length.
        next_beats = sorted(by_bar.get(i + 1, []))
        if next_beats:
            t_end = next_beats[0]
        elif len(beats) >= 2:
            t_end = beats[-1] + (beats[-1] - beats[0]) / max(1, len(beats) - 1)
        else:
            t_end = beats[-1] + 0.5
        ranges.append((t_start, t_end))
    return ranges


def _section_label_at_bar(analysis: AnalysisResult, bar_t_ranges: list[tuple[float, float]], bar_idx: int) -> str:
    if not analysis.sections or bar_idx >= len(bar_t_ranges):
        return ""
    t_mid = sum(bar_t_ranges[bar_idx]) / 2.0
    for s in analysis.sections:
        if s.start <= t_mid < s.end:
            return s.label
    return ""


def _bar_onset_count(analysis: AnalysisResult, bar_t_ranges: list[tuple[float, float]]) -> list[int]:
    """Drum-onset count per bar."""
    drums = _drum_onsets(analysis)
    counts = [0] * len(bar_t_ranges)
    j = 0  # pointer into drums (assumes drums sorted by t)
    drums_sorted = sorted(drums, key=lambda o: o.t)
    for i, (t_start, t_end) in enumerate(bar_t_ranges):
        while j < len(drums_sorted) and drums_sorted[j].t < t_start:
            j += 1
        k = j
        while k < len(drums_sorted) and drums_sorted[k].t < t_end:
            counts[i] += 1
            k += 1
    return counts


def _autocorrelate_period(values: list[int], candidate_lengths: tuple[int, ...]) -> tuple[int, float]:
    """Return (best_period, confidence) by autocorrelating `values` against
    each candidate length. Confidence is the normalized correlation peak.
    """
    n = len(values)
    if n < min(candidate_lengths):
        return (n, 0.0)

    best_period = candidate_lengths[-1]
    best_score = 0.0
    mean = sum(values) / n
    centered = [v - mean for v in values]
    denom = sum(c * c for c in centered)
    if denom == 0:
        return (best_period, 0.0)

    for period in candidate_lengths:
        if period > n:
            continue
        # Lag-`period` autocorrelation, normalized.
        num = sum(centered[i] * centered[i + period] for i in range(n - period))
        score = num / denom
        if score > best_score:
            best_score = score
            best_period = period

    return (best_period, max(0.0, min(1.0, best_score)))


def _section_phrase_blocks(
    analysis: AnalysisResult,
    bar_t_ranges: list[tuple[float, float]],
    bar_counts: list[int],
) -> list[PhraseBlock]:
    """Slice phrase blocks per section, picking the best period within each."""
    blocks: list[PhraseBlock] = []
    if not analysis.sections:
        # No sections: treat the whole song as one virtual section.
        period, conf = _autocorrelate_period(bar_counts, PHRASE_CANDIDATE_LENGTHS)
        if period <= 0:
            return blocks
        for start in range(0, len(bar_counts), period):
            end = min(start + period, len(bar_counts))
            blocks.append(
                PhraseBlock(
                    bar_start=start, bar_end=end, length=period,
                    section_label="", confidence=conf,
                )
            )
        return blocks

    # Per-section: find which bars belong to this section and autocorrelate within.
    for section in analysis.sections:
        section_bars = [
            i for i, (ts, te) in enumerate(bar_t_ranges)
            if ts >= section.start and ts < section.end
        ]
        if not section_bars:
            continue
        section_counts = [bar_counts[i] for i in section_bars]
        period, conf = _autocorrelate_period(section_counts, PHRASE_CANDIDATE_LENGTHS)
        if period <= 0:
            continue
        for offset in range(0, len(section_bars), period):
            chunk = section_bars[offset:offset + period]
            if not chunk:
                continue
            blocks.append(
                PhraseBlock(
                    bar_start=chunk[0],
                    bar_end=chunk[-1] + 1,
                    length=len(chunk),
                    section_label=section.label,
                    confidence=conf,
                )
            )
    return blocks


def _detect_fills(
    bar_counts: list[int],
    blocks: list[PhraseBlock],
    bar_t_ranges: list[tuple[float, float]],
    analysis: AnalysisResult,
) -> list[FillEvent]:
    """A fill is a bar with high drum-onset z-score AND at the end of a phrase."""
    n = len(bar_counts)
    if n < 2:
        return []
    mean = sum(bar_counts) / n
    var = sum((c - mean) ** 2 for c in bar_counts) / n
    std = var ** 0.5 or 1.0

    end_bars = {b.bar_end - 1 for b in blocks}

    fills: list[FillEvent] = []
    for bar_idx, count in enumerate(bar_counts):
        z = (count - mean) / std
        if z >= FILL_ZSCORE_THRESHOLD and bar_idx in end_bars:
            t_start, t_end = bar_t_ranges[bar_idx]
            # leads_into = label of the next bar's section, if it differs.
            here = _section_label_at_bar(analysis, bar_t_ranges, bar_idx)
            nxt = _section_label_at_bar(analysis, bar_t_ranges, bar_idx + 1)
            leads_into = nxt if nxt and nxt != here else None
            fills.append(
                FillEvent(
                    bar=bar_idx,
                    t_start=t_start,
                    t_end=t_end,
                    density_zscore=z,
                    leads_into=leads_into,
                )
            )
    return fills


def _syncopation_per_bar(
    analysis: AnalysisResult,
    bar_t_ranges: list[tuple[float, float]],
) -> list[float]:
    """Per-bar syncopation: off-beat onset strength / total onset strength.

    "On-beat" = within ~1/8 of a bar of any beat in that bar. Anything else
    is off-beat. Bars with no onsets get 0.0.
    """
    drums = _drum_onsets(analysis)
    drums_sorted = sorted(drums, key=lambda o: o.t)
    by_bar: dict[int, list[float]] = defaultdict(list)
    for b in analysis.beats:
        by_bar[b.bar].append(b.t)

    scores: list[float] = []
    j = 0
    for i, (t_start, t_end) in enumerate(bar_t_ranges):
        bar_beats = sorted(by_bar.get(i, []))
        if not bar_beats:
            scores.append(0.0)
            continue
        bar_dur = max(t_end - t_start, 1e-6)
        # On-beat tolerance ≈ 1/16 of a bar (half a 16th-note in 4/4). Onsets
        # within this window of any beat in the bar are "on-beat"; everything
        # else contributes to the syncopation fraction.
        on_threshold = bar_dur / 16.0

        while j < len(drums_sorted) and drums_sorted[j].t < t_start:
            j += 1
        total_strength = 0.0
        off_strength = 0.0
        k = j
        while k < len(drums_sorted) and drums_sorted[k].t < t_end:
            o = drums_sorted[k]
            s = float(o.strength or 0.5)
            total_strength += s
            on_beat = any(abs(o.t - bt) < on_threshold for bt in bar_beats)
            if not on_beat:
                off_strength += s
            k += 1
        scores.append(off_strength / total_strength if total_strength > 0 else 0.0)
    return scores


def detect_patterns(analysis: AnalysisResult) -> Patterns:
    """Run all pattern detectors against an `AnalysisResult`.

    Pure: returns a `Patterns` object. Does not mutate the input.
    """
    n_bars = _bar_count(analysis)
    if n_bars == 0:
        return Patterns()

    bar_t_ranges = _bar_t_ranges(analysis)
    bar_counts = _bar_onset_count(analysis, bar_t_ranges)
    phrases = _section_phrase_blocks(analysis, bar_t_ranges, bar_counts)
    fills = _detect_fills(bar_counts, phrases, bar_t_ranges, analysis)
    syncopation = _syncopation_per_bar(analysis, bar_t_ranges)

    return Patterns(
        phrases=phrases,
        fills=fills,
        syncopation_per_bar=syncopation,
        bar_count=n_bars,
    )


def populate_beat_pattern_fields(analysis: AnalysisResult) -> AnalysisResult:
    """Populate phrase_id/phrase_position/is_fill/syncopation on every beat.

    Returns a deep copy with the new fields set. Pattern detection runs first
    so this is cheap.
    """
    out = analysis.model_copy(deep=True)
    if out.patterns is None:
        out.patterns = detect_patterns(out)

    fill_bars = {f.bar for f in out.patterns.fills}
    bar_to_phrase: dict[int, tuple[int, int, int]] = {}
    for phrase_idx, p in enumerate(out.patterns.phrases):
        for offset, bar_idx in enumerate(range(p.bar_start, p.bar_end)):
            bar_to_phrase[bar_idx] = (phrase_idx, offset + 1, p.length)

    syncopation = out.patterns.syncopation_per_bar

    for b in out.beats:
        info = bar_to_phrase.get(b.bar)
        if info is not None:
            b.phrase_id, b.phrase_position, b.phrase_length = info
        b.is_fill = b.bar in fill_bars
        if 0 <= b.bar < len(syncopation):
            b.syncopation = syncopation[b.bar]

    return out
