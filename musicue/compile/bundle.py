"""Build a MusiCueBundle from an AnalysisResult + its compiled CueSheet."""
from __future__ import annotations

import logging
import math

from musicue.compile.normalize import percentile_normalize
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


# LUFS frames at or below this are treated as silence (the LUFS curve
# floors digital silence at -70).
_LUFS_SILENCE = -60.0
# Per-stem RMS (dBFS) at or below this is treated as silence / bleed.
_RMS_DB_SILENCE = -60.0
_STEMS = ("drums", "bass", "vocals", "other")


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _section_lufs(analysis: AnalysisResult, start: float, end: float) -> float | None:
    """Mean LUFS over the section's non-silent frames.

    Silent (-70 floor) frames are excluded so a section with a pause in it
    isn't dragged toward -70; a fully silent section reports the floor.
    """
    curve = analysis.curves.get("lufs")
    if curve is None or curve.hop_sec <= 0 or not curve.values:
        return None
    i0 = max(0, int(start / curve.hop_sec))
    i1 = min(len(curve.values), int(end / curve.hop_sec))
    if i1 <= i0:
        return None
    window = [v for v in curve.values[i0:i1] if v > _LUFS_SILENCE]
    if not window:
        return min(curve.values[i0:i1])
    return sum(window) / len(window)


def _build_sections(analysis: AnalysisResult) -> list[SectionBundleEntry]:
    if not analysis.sections:
        return []

    transitions_by_t = {round(tr.t, 3): tr for tr in analysis.section_transitions}

    cached: list[tuple[float | None, float | None]] = []  # (lufs, spectral_rise) per section
    for sec in analysis.sections:
        tr = transitions_by_t.get(round(sec.start, 3))
        spectral_rise = tr.ramp_evidence.spectral_flux_rise if tr else None
        lufs = _section_lufs(analysis, sec.start, sec.end)
        cached.append((lufs, spectral_rise))

    # energy_rank is loudness only: LUFS (dB) and spectral_flux_rise (a
    # 0..1 boundary statistic) are different units and can't be averaged.
    # Sections without a LUFS value get the neutral 0.5.
    known = [lufs for lufs, _ in cached if lufs is not None]
    normalized = iter(_normalize(known))
    ranks = [next(normalized) if lufs is not None else 0.5 for lufs, _ in cached]

    out: list[SectionBundleEntry] = []
    for sec, (lufs, spectral_rise), rank in zip(analysis.sections, cached, ranks):
        out.append(SectionBundleEntry(
            start=sec.start,
            end=sec.end,
            label=sec.label,
            confidence=sec.confidence,
            lufs=lufs,
            energy_rank=rank,
            spectral_flux_rise=spectral_rise,
        ))
    return out


def _build_global_energy(analysis: AnalysisResult) -> StemEnergyCurve:
    """Mix loudness in 0..1: 5th→0, 95th→1 percentile of non-silent LUFS.

    Min-max over raw LUFS let the -70 silence floor define 0, squashing the
    whole song into the top of the range.
    """
    curve = analysis.curves.get("lufs")
    if curve is None or not curve.values:
        return StemEnergyCurve(hop_sec=0.04, values=[])
    return StemEnergyCurve(
        hop_sec=curve.hop_sec,
        values=percentile_normalize(
            curve.values, 5.0, 95.0, floor=_LUFS_SILENCE, flat_value=0.5
        ),
    )


def _build_stems_energy(analysis: AnalysisResult) -> dict[str, StemEnergyCurve]:
    """Per-stem loudness in 0..1 from the ``rms_<stem>`` analysis curves.

    RMS is converted to dBFS and percentile-normalized per stem (5th→0,
    95th→1 of that stem's non-silent frames), so each stem uses its own
    dynamic range: a quiet vocal still reaches 1.0 at its loudest.
    """
    out: dict[str, StemEnergyCurve] = {}
    for stem in _STEMS:
        curve = analysis.curves.get(f"rms_{stem}")
        if curve is None or not curve.values:
            continue
        db = [20.0 * math.log10(max(float(v), 1e-10)) for v in curve.values]
        out[stem] = StemEnergyCurve(
            hop_sec=curve.hop_sec,
            values=percentile_normalize(
                db, 5.0, 95.0, floor=_RMS_DB_SILENCE, flat_value=0.5
            ),
        )
    return out


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
        # Per bin: the loudest sounding note (velocity weighted by how much of
        # the bin it covers). Summing over polyphony saturated at 1.0 for
        # any chord.
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
                values[b] = max(values[b], vel_norm * (overlap / hop_sec))
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
        stems_energy=_build_stems_energy(analysis),
        global_energy=_build_global_energy(analysis),
        cuesheet=cuesheet,
    )
