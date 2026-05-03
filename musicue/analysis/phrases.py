"""Group transcribed MIDI notes into phrases separated by silence gaps.

A phrase is a contiguous run of notes whose inter-note gaps stay below a
threshold (default 0.6 s). The output is a list of phrase dicts that matches
the M1 ``phrases`` schema (timescale ``"meso"``), used downstream for cue
generation and structural reasoning. This module is pure Python -- no audio
or ML dependencies -- and operates on the note dicts produced by
``musicue.analysis.transcription.transcribe_stem``.
"""
from __future__ import annotations


def group_into_phrases(notes: list[dict], gap_sec: float = 0.6) -> list[dict]:
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

    phrases: list[dict] = []
    for group in groups:
        t_start = group[0]["t"]
        last = group[-1]
        t_end = last["t"] + last.get("duration", 0.3)
        pitches = [n["pitch"] for n in group]
        stride = max(1, len(pitches) // 10)
        phrases.append({
            "t_start": float(t_start),
            "t_end": float(t_end),
            "timescale": "meso",
            "note_count": len(group),
            "pitch_peak": int(max(pitches)),
            "pitch_low": int(min(pitches)),
            "pitch_contour": [int(p) for p in pitches[::stride]],
            "energy_curve": {"hop_sec": 0.04, "values": []},
            "labels": [],
        })
    return phrases
