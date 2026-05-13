"""Build a MusiCueBundle from an AnalysisResult + its compiled CueSheet."""
from __future__ import annotations

import logging

from musicue.schemas import (
    AnalysisResult,
    CueSheet,
    DrumOnset,
    MidiNoteBundle,
    MusiCueBundle,
    SectionBundleEntry,
    StemEnergyCurve,
    TempoInfo,
)

_logger = logging.getLogger(__name__)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _section_lufs(analysis: AnalysisResult, start: float, end: float) -> float | None:
    curve = analysis.curves.get("lufs")
    if curve is None or curve.hop_sec <= 0 or not curve.values:
        return None
    i0 = max(0, int(start / curve.hop_sec))
    i1 = min(len(curve.values), int(end / curve.hop_sec))
    if i1 <= i0:
        return None
    window = curve.values[i0:i1]
    return sum(window) / len(window)


def _build_sections(analysis: AnalysisResult) -> list[SectionBundleEntry]:
    if not analysis.sections:
        return []

    transitions_by_t = {round(tr.t, 3): tr for tr in analysis.section_transitions}

    raw_scores: list[float] = []
    cached: list[tuple[float | None, float | None]] = []  # (lufs, spectral_rise) per section
    for sec in analysis.sections:
        tr = transitions_by_t.get(round(sec.start, 3))
        spectral_rise = tr.ramp_evidence.spectral_flux_rise if tr else None
        lufs = _section_lufs(analysis, sec.start, sec.end)
        cached.append((lufs, spectral_rise))

        score = 0.0
        components = 0
        if spectral_rise is not None:
            score += spectral_rise
            components += 1
        if lufs is not None:
            score += lufs
            components += 1
        raw_scores.append(score / components if components else 0.0)

    ranks = _normalize(raw_scores)
    out: list[SectionBundleEntry] = []
    for sec, (lufs, spectral_rise), rank in zip(analysis.sections, cached, ranks):
        out.append(SectionBundleEntry(
            start=sec.start,
            end=sec.end,
            label=sec.label,
            confidence=sec.confidence,
            lufs=lufs,
            energy_rank=rank if rank is not None else 0.5,
            spectral_flux_rise=spectral_rise,
        ))
    return out


def _build_global_energy(analysis: AnalysisResult) -> StemEnergyCurve:
    curve = analysis.curves.get("lufs")
    if curve is None or not curve.values:
        return StemEnergyCurve(hop_sec=0.04, values=[])
    return StemEnergyCurve(hop_sec=curve.hop_sec, values=_normalize(curve.values))


def _build_midi(analysis: AnalysisResult) -> dict[str, list[MidiNoteBundle]]:
    out: dict[str, list[MidiNoteBundle]] = {}
    for stem, notes in analysis.midi.items():
        out[stem] = [
            MidiNoteBundle(t=n.t, duration=n.duration, pitch=n.pitch, velocity=n.velocity)
            for n in notes
        ]
    return out


def _build_midi_energy(
    analysis: AnalysisResult, hop_sec: float, duration_sec: float
) -> dict[str, StemEnergyCurve]:
    if hop_sec <= 0:
        return {}
    n_bins = int(duration_sec / hop_sec)
    out: dict[str, StemEnergyCurve] = {}
    for stem, notes in analysis.midi.items():
        values = [0.0] * n_bins
        for note in notes:
            note_end = note.t + note.duration
            bin_start = max(0, int(note.t / hop_sec))
            bin_end = min(n_bins, int(note_end / hop_sec) + 1)
            vel_norm = note.velocity / 127.0
            for b in range(bin_start, bin_end):
                bin_t0 = b * hop_sec
                bin_t1 = bin_t0 + hop_sec
                overlap = max(0.0, min(bin_t1, note_end) - max(bin_t0, note.t))
                values[b] += vel_norm * (overlap / hop_sec)
        values = [max(0.0, min(1.0, v)) for v in values]
        out[stem] = StemEnergyCurve(hop_sec=hop_sec, values=values)
    return out


def _build_drums(analysis: AnalysisResult) -> dict[str, list[DrumOnset]]:
    out: dict[str, list[DrumOnset]] = {}
    for onset in analysis.onsets.get("drums", []):
        if onset.drum_class is None:
            continue
        out.setdefault(onset.drum_class, []).append(
            DrumOnset(t=onset.t, strength=onset.strength, confidence=onset.drum_class_conf)
        )
    return out


def _warn_if_drums_unclassified(analysis: AnalysisResult, drums: dict) -> None:
    """Loud warning when a song has drum onsets but no classifier output.

    Symptom: ``analysis.onsets["drums"]`` has events, but after regrouping by
    ``drum_class`` the bundle's ``drums`` dict is empty. Almost always means
    ``drum_classifier_version="not_trained"`` — the CNN checkpoint
    (``models/drum_cnn.pt``) wasn't found at analysis time, so
    ``classify_onsets_batch`` passed onsets through with ``drum_class=None``
    and the bundle correctly dropped them.

    Without this warning the bundle silently has empty drums, which makes
    CedarToy's iChannel0 low/low_mid/mid_hi bin ranges go dark while
    everything else looks fine — easy to miss in a visual A/B test.
    """
    raw_drum_count = len(analysis.onsets.get("drums", []))
    if raw_drum_count > 0 and not drums:
        version = analysis.analysis_config.drum_classifier_version or "unknown"
        _logger.warning(
            "Bundle has %d drum onsets but ZERO classified — drum_classifier_version=%r. "
            "Train or install the drum CNN checkpoint (models/drum_cnn.pt) to populate "
            "kick/snare/hat/tom/cymbal tracks. Bundle proceeds with empty drums.",
            raw_drum_count, version,
        )


def build_bundle(analysis: AnalysisResult, cuesheet: CueSheet) -> MusiCueBundle:
    if analysis.source.sha256 != cuesheet.source_sha256:
        raise ValueError(
            f"Analysis sha256={analysis.source.sha256} does not match "
            f"cuesheet sha256={cuesheet.source_sha256}"
        )

    drums = _build_drums(analysis)
    _warn_if_drums_unclassified(analysis, drums)

    return MusiCueBundle(
        source_sha256=analysis.source.sha256,
        duration_sec=analysis.source.duration_sec,
        fps=cuesheet.fps,
        tempo=analysis.tempo if analysis.tempo else TempoInfo(bpm_global=120.0),
        beats=analysis.beats,
        sections=_build_sections(analysis),
        drums=drums,
        midi=_build_midi(analysis),
        midi_energy=_build_midi_energy(
            analysis,
            analysis.analysis_config.curve_hop_sec,
            analysis.source.duration_sec,
        ),
        stems_energy={},
        global_energy=_build_global_energy(analysis),
        cuesheet=cuesheet,
    )
