"""Grammar-driven analysis -> cuesheet compiler.

This module implements the M2 compiler that consumes an
:class:`~musicue.schemas.AnalysisResult` plus a YAML grammar (loaded as a
:class:`~musicue.compile.grammar.Grammar`) and emits a
:class:`~musicue.schemas.CueSheet`.

Architecture
------------
For each :class:`~musicue.compile.grammar.GrammarTrack` declared in the
grammar, the compiler:

1. Resolves the track's ``source`` string (e.g. ``onsets.drums``,
   ``beats``, ``sections``, ``section_transitions``, ``phrases.vocals``,
   ``curves.lufs``, or ``onsets.*``) to a list of event dicts via
   :func:`_resolve_source`.
2. Tags each event with ``section_label`` based on the section that
   contains its time, so filters like ``section_label == 'chorus'`` can
   match.
3. Applies the ``filter`` DSL via
   :func:`musicue.compile.scoring.evaluate_filter` to drop non-matching
   events.
4. Honors ``cooldown_sec`` (impulse tracks only) by skipping events that
   fall within the cooldown of the most recently emitted event.
5. Computes a final score via
   :func:`musicue.compile.scoring.compute_score` using
   ``hierarchy_weights[event.timescale]`` and an optional
   :class:`~musicue.compile.scoring.RarityTracker` bonus.
6. Builds a :class:`~musicue.schemas.CueTrack` of the appropriate type
   (impulse / envelope / step / ramp / continuous).

Continuous tracks support optional ``smoothing`` (EMA) and ``normalize``
(percentile clipping to [0, 1]) post-processing declared on the track
config under the ``model_extra`` slot of the GrammarTrack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np

from musicue.compile.grammar import Grammar, GrammarTrack, load_grammar
from musicue.compile.scoring import RarityTracker, compute_score, evaluate_filter
from musicue.schemas import AnalysisResult, CueSheet, CueTrack

_Timescale = Literal["micro", "meso", "macro"]
_VALID_TIMESCALES: frozenset[str] = frozenset({"micro", "meso", "macro"})


def _coerce_timescale(value: str | None, default: _Timescale = "micro") -> _Timescale:
    """Narrow an arbitrary string to the CueTrack timescale literal."""
    if value in _VALID_TIMESCALES:
        return cast(_Timescale, value)
    return default


def _resolve_source(source: str, analysis: AnalysisResult) -> list[dict]:
    """Resolve a grammar source string to a list of event dicts."""
    if source == "beats":
        return [b.model_dump() for b in analysis.beats]
    if source == "sections":
        return [s.model_dump() for s in analysis.sections]
    if source == "section_transitions":
        return [t.model_dump(by_alias=True) for t in analysis.section_transitions]
    if source.startswith("onsets."):
        stem = source[len("onsets."):]
        if stem == "*":
            all_events: list[dict] = []
            for events in analysis.onsets.values():
                all_events.extend(e.model_dump() for e in events)
            return all_events
        return [e.model_dump() for e in analysis.onsets.get(stem, [])]
    if source.startswith("phrases."):
        stem = source[len("phrases."):]
        return [p.model_dump() for p in analysis.phrases.get(stem, [])]
    if source.startswith("curves."):
        curve_name = source[len("curves."):]
        curve = analysis.curves.get(curve_name)
        if curve:
            return [{"_curve": True, "hop_sec": curve.hop_sec, "values": list(curve.values)}]
    return []


def _section_label_at(t: float, analysis: AnalysisResult) -> str:
    """Return the label of the section that contains ``t`` (or empty string)."""
    for s in reversed(analysis.sections):
        if s.start <= t:
            return s.label
    return ""


def _smooth_ema(values: list[float], tau_sec: float, hop_sec: float) -> list[float]:
    """Apply a one-pole EMA with time constant ``tau_sec`` at sample period ``hop_sec``."""
    if not values:
        return []
    alpha = 1.0 - float(np.exp(-hop_sec / max(tau_sec, 1e-6)))
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def _normalize_percentile(values: list[float], low: float, high: float) -> list[float]:
    """Clip values into the [low, high] percentile band and rescale to [0, 1]."""
    if not values:
        return []
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi == lo:
        return [0.0] * len(values)
    return [float(np.clip((float(v) - lo) / (hi - lo), 0.0, 1.0)) for v in values]


def _compile_continuous_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    ev = source_events[0]
    if not ev.get("_curve"):
        return None
    values = list(ev["values"])
    hop_sec = float(ev["hop_sec"])

    extra = track_cfg.model_extra or {}
    smoothing = extra.get("smoothing")
    if isinstance(smoothing, dict) and smoothing.get("kind") == "ema":
        values = _smooth_ema(
            values,
            tau_sec=float(smoothing.get("tau_sec", 0.25)),
            hop_sec=hop_sec,
        )

    normalize = extra.get("normalize")
    if isinstance(normalize, dict) and normalize.get("kind") == "percentile":
        values = _normalize_percentile(
            values,
            low=float(normalize.get("low", 5)),
            high=float(normalize.get("high", 95)),
        )

    return CueTrack(
        name=track_cfg.name,
        type="continuous",
        timescale="macro",
        hop_sec=hop_sec,
        values=values,
    )


def _compile_impulse_track(
    track_cfg: GrammarTrack,
    analysis: AnalysisResult,
    hierarchy_weights: dict[str, float],
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None

    rarity: RarityTracker | None = None
    if track_cfg.rarity:
        rarity = RarityTracker(
            window_sec=float(track_cfg.rarity.get("window_sec", 1.0)),
            decay=float(track_cfg.rarity.get("decay", 4.0)),
        )

    last_emitted_t: float | None = None
    emitted: list[dict] = []
    track_timescale: str | None = None

    sorted_events = sorted(
        source_events,
        key=lambda e: float(e.get("t", e.get("t_start", 0.0))),
    )

    # Precompute downbeat times once and tag each event with its distance to
    # the nearest downbeat. The filter DSL's ``near_downbeat(<seconds>)`` form
    # reads ``downbeat_distance_sec`` and compares against its captured
    # argument so different windows produce different results.
    downbeat_times = [b.t for b in analysis.beats if b.is_downbeat]

    for ev in sorted_events:
        t = float(ev.get("t", ev.get("t_start", 0.0)))
        ev["section_label"] = _section_label_at(t, analysis)
        if downbeat_times:
            ev["downbeat_distance_sec"] = min(abs(t - dt) for dt in downbeat_times)
        else:
            ev["downbeat_distance_sec"] = float("inf")

        if not evaluate_filter(track_cfg.filter, ev):
            continue

        if track_cfg.cooldown_sec and last_emitted_t is not None:
            if t - last_emitted_t < float(track_cfg.cooldown_sec):
                continue

        ev_timescale = ev.get("timescale", "micro")
        tw = float(hierarchy_weights.get(ev_timescale, 1.0))
        rb = rarity.bonus(t) if rarity else 1.0
        score = compute_score(
            track_cfg.score, ev, timescale_weight=tw, rarity_bonus=rb
        )

        if rarity:
            rarity.record(t)

        if track_timescale is None:
            track_timescale = ev_timescale

        emitted.append(
            {
                "t": t,
                "strength": float(score),
                "envelope": dict(track_cfg.envelope),
                "tags": [],
            }
        )
        last_emitted_t = t

    if not emitted:
        return None

    return CueTrack(
        name=track_cfg.name,
        type="impulse",
        timescale=_coerce_timescale(track_timescale),
        events=emitted,
    )


def _compile_step_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult
) -> CueTrack | None:
    sections = analysis.sections
    if not sections:
        return None
    events = [
        {"t": float(s.start), "value": i + 1, "label": s.label}
        for i, s in enumerate(sections)
    ]
    return CueTrack(
        name=track_cfg.name,
        type="step",
        timescale="macro",
        events=events,
    )


def _compile_ramp_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    events: list[dict] = []
    for ev in source_events:
        if not evaluate_filter(track_cfg.filter, ev):
            continue
        ramp = ev.get("ramp", {}) or {}
        t = float(ev.get("t", 0.0))
        events.append(
            {
                "t_start": float(ramp.get("t_start", t - 1.2)),
                "t_end": float(ramp.get("t_end", t)),
                "from": 0.0,
                "to": 1.0,
                "shape": ramp.get("shape", "ease_in_out"),
                "label": f"{ev.get('from', '')}->{ev.get('to', '')}",
            }
        )
    if not events:
        return None
    return CueTrack(
        name=track_cfg.name,
        type="ramp",
        timescale="macro",
        events=events,
    )


def _compile_envelope_track(
    track_cfg: GrammarTrack,
    analysis: AnalysisResult,
    hierarchy_weights: dict[str, float],
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    emitted: list[dict] = []
    for ev in source_events:
        t_start = float(ev.get("t_start", ev.get("t", 0.0)))
        ev["section_label"] = _section_label_at(t_start, analysis)
        if not evaluate_filter(track_cfg.filter, ev):
            continue
        tw = float(hierarchy_weights.get(ev.get("timescale", "meso"), 1.0))
        score = compute_score(
            track_cfg.score, ev, timescale_weight=tw, rarity_bonus=1.0
        )
        t_end = float(ev.get("t_end", t_start + 1.0))
        event_dict: dict = {
            "t_start": t_start,
            "t_end": t_end,
            "strength": float(score),
            "envelope": dict(track_cfg.envelope),
            "tags": [],
        }
        if track_cfg.shape_curve_from and track_cfg.shape_curve_from in ev:
            event_dict["shape_curve"] = ev[track_cfg.shape_curve_from]
        emitted.append(event_dict)
    if not emitted:
        return None
    return CueTrack(
        name=track_cfg.name,
        type="envelope",
        timescale="meso",
        events=emitted,
    )


def compile_analysis(
    analysis: AnalysisResult,
    grammar: str | Grammar = "concert_visuals",
    grammars_dir: Path | None = None,
    fps: float | None = None,
    drop_frame: bool | None = None,
) -> CueSheet:
    """Compile an :class:`AnalysisResult` into a :class:`CueSheet` per ``grammar``.

    Parameters
    ----------
    analysis:
        The analysis result to compile.
    grammar:
        Either a grammar name (resolved against ``grammars_dir``) or an
        already-loaded :class:`Grammar`.
    grammars_dir:
        Directory used to resolve grammar names. When ``None`` (the
        default) the packaged grammars directory is used so built-in
        grammars load regardless of CWD. Ignored when ``grammar`` is a
        :class:`Grammar` instance.
    """
    if isinstance(grammar, str):
        grammar = load_grammar(grammar, grammars_dir=grammars_dir)

    hw = grammar.hierarchy_weights
    tracks: list[CueTrack] = []

    for track_cfg in grammar.tracks:
        track: CueTrack | None = None
        if track_cfg.type == "impulse":
            track = _compile_impulse_track(track_cfg, analysis, hw)
        elif track_cfg.type == "step":
            track = _compile_step_track(track_cfg, analysis)
        elif track_cfg.type == "ramp":
            track = _compile_ramp_track(track_cfg, analysis)
        elif track_cfg.type == "envelope":
            track = _compile_envelope_track(track_cfg, analysis, hw)
        elif track_cfg.type == "continuous":
            track = _compile_continuous_track(track_cfg, analysis)
        if track is not None:
            tracks.append(track)

    # Resolve fps/drop_frame: explicit args win, then analysis_config, then default.
    cs_fps = (
        fps
        if fps is not None
        else analysis.analysis_config.fps
    )
    cs_drop = (
        drop_frame
        if drop_frame is not None
        else analysis.analysis_config.drop_frame
    )

    sheet = CueSheet(
        source_sha256=analysis.source.sha256,
        grammar=grammar.name,
        duration_sec=analysis.source.duration_sec,
        fps=cs_fps,
        drop_frame=cs_drop,
        tempo_map=analysis.tempo.bpm_curve if analysis.tempo else [],
        tracks=tracks,
    )
    # Stamp frame/timecode on every event using the cuesheet's fps.
    from musicue.frame_population import populate_cuesheet_frames

    return populate_cuesheet_frames(sheet, cs_fps, cs_drop)
