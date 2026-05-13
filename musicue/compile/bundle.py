"""Build a MusiCueBundle from an AnalysisResult + its compiled CueSheet."""
from __future__ import annotations

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


def build_bundle(analysis: AnalysisResult, cuesheet: CueSheet) -> MusiCueBundle:
    if analysis.source.sha256 != cuesheet.source_sha256:
        raise ValueError(
            f"Analysis sha256={analysis.source.sha256} does not match "
            f"cuesheet sha256={cuesheet.source_sha256}"
        )

    return MusiCueBundle(
        source_sha256=analysis.source.sha256,
        duration_sec=analysis.source.duration_sec,
        fps=cuesheet.fps,
        tempo=analysis.tempo if analysis.tempo else TempoInfo(bpm_global=120.0),
        beats=analysis.beats,
        sections=_build_sections(analysis),
        drums={},
        midi={},
        midi_energy={},
        stems_energy={},
        global_energy=StemEnergyCurve(hop_sec=0.04, values=[]),
        cuesheet=cuesheet,
    )
