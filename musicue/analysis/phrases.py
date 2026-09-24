"""Group transcribed MIDI notes into phrases separated by silence gaps.

A phrase is a contiguous run of notes whose inter-note gaps stay below a
threshold (default 0.6 s). The output is a list of phrase dicts that matches
the M1 ``phrases`` schema (timescale ``"meso"``), used downstream for cue
generation and structural reasoning. This module is pure Python -- no audio
or ML dependencies -- and operates on the note dicts produced by
``musicue.analysis.transcription.transcribe_stem``.
"""
from __future__ import annotations

import math

_DEFAULT_HOP_SEC = 0.04


def _rms_reference(values: list[float]) -> float:
    """99th-percentile level of a stem RMS curve (robust "full scale")."""
    ordered = sorted(v for v in values if v > 0.0)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, int(math.ceil(0.99 * len(ordered))) - 1)
    return float(ordered[max(0, idx)])


def _energy_from_rms(
    rms_curve: dict, ref: float, t_start: float, t_end: float
) -> dict | None:
    hop = float(rms_curve.get("hop_sec") or 0.0)
    values = rms_curve.get("values") or []
    if hop <= 0 or not values or ref <= 0:
        return None
    i0 = max(0, int(t_start / hop))
    i1 = min(len(values), max(i0 + 1, int(math.ceil(t_end / hop))))
    window = values[i0:i1]
    if not window:
        return None
    return {
        "hop_sec": hop,
        "values": [min(1.0, max(0.0, float(v) / ref)) for v in window],
    }


def _energy_from_velocity(group: list[dict], t_start: float, t_end: float) -> dict:
    """Per-hop max velocity (0..1) of the notes sounding in each hop."""
    hop = _DEFAULT_HOP_SEC
    n_bins = max(1, int(math.ceil((t_end - t_start) / hop)))
    values = [0.0] * n_bins
    for note in group:
        n_start = note["t"]
        n_end = n_start + note.get("duration", 0.3)
        vel = min(1.0, max(0.0, note.get("velocity", 64) / 127.0))
        b0 = max(0, int((n_start - t_start) / hop))
        b1 = min(n_bins, max(b0 + 1, int(math.ceil((n_end - t_start) / hop))))
        for b in range(b0, b1):
            values[b] = max(values[b], vel)
    return {"hop_sec": hop, "values": values}


def group_into_phrases(
    notes: list[dict],
    gap_sec: float = 0.6,
    rms_curve: dict | None = None,
) -> list[dict]:
    """Group sorted MIDI note dicts into phrases separated by silence gaps.

    Parameters
    ----------
    notes:
        Iterable of note dicts shaped like
        ``{"t": float, "duration": float, "pitch": int, "velocity": int}``.
        Need not be pre-sorted; we sort defensively by ``t``.
    gap_sec:
        Maximum silence (seconds) between the end of one note and the start
        of the next within the same phrase. A gap strictly larger than this
        starts a new phrase.
    rms_curve:
        Optional ``{"hop_sec", "values"}`` RMS curve of the stem the notes
        came from. When given, each phrase's ``energy_curve`` is that curve
        over the phrase span, scaled by the stem's 99th-percentile RMS into
        [0, 1]. Otherwise it falls back to per-hop max note velocity / 127.

    Returns
    -------
    list[dict]
        One phrase dict per detected phrase, in time order.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n["t"])
    groups: list[list[dict]] = []
    current: list[dict] = [sorted_notes[0]]
    for note in sorted_notes[1:]:
        prev = current[-1]
        prev_end = prev["t"] + prev.get("duration", 0.3)
        if note["t"] - prev_end > gap_sec:
            groups.append(current)
            current = [note]
        else:
            current.append(note)
    groups.append(current)

    rms_ref = _rms_reference(list(rms_curve.get("values") or [])) if rms_curve else 0.0

    phrases: list[dict] = []
    for group in groups:
        t_start = group[0]["t"]
        last = group[-1]
        t_end = last["t"] + last.get("duration", 0.3)
        pitches = [n["pitch"] for n in group]
        stride = max(1, len(pitches) // 10)
        energy = None
        if rms_curve is not None:
            energy = _energy_from_rms(rms_curve, rms_ref, t_start, t_end)
        if energy is None:
            energy = _energy_from_velocity(group, t_start, t_end)
        phrases.append({
            "t_start": float(t_start),
            "t_end": float(t_end),
            "timescale": "meso",
            "note_count": len(group),
            "pitch_peak": int(max(pitches)),
            "pitch_low": int(min(pitches)),
            "pitch_contour": [int(p) for p in pitches[::stride]],
            "energy_curve": energy,
            "labels": [],
        })
    return phrases
