"""Polyphonic note transcription via Spotify's Basic Pitch.

Basic Pitch (https://github.com/spotify/basic-pitch) is a lightweight CNN that
produces frame-wise pitch activations and decodes them into MIDI note events.
We wrap its ``predict`` API and normalize the output into a list of dicts that
matches the M1 ``notes`` schema (``{t, duration, pitch, velocity}``).

The dependency is optional and lazy-imported inside ``transcribe_stem`` so the
package can be inspected (and tests can stub it via ``sys.modules``) without
``basic-pitch`` actually being installed.
"""
from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

log = logging.getLogger(__name__)


def basic_pitch_version() -> str:
    """Return the installed basic-pitch package version, or 'unknown' if not installed."""
    try:
        return _pkg_version("basic-pitch")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def transcribe_stem(audio_path: Path) -> list[dict]:
    """Transcribe a single stem (or full mix) into a list of note events.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file (any format basic-pitch / librosa can decode).

    Returns
    -------
    list[dict]
        Notes sorted by start time. Each dict has::

            {
                "t": float,          # start time in seconds
                "duration": float,   # seconds (end - start)
                "pitch": int,        # MIDI note number
                "velocity": int,     # 1..127, derived from amplitude
            }
    """
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, note_events = predict(str(audio_path), ICASSP_2022_MODEL_PATH)
    notes = [
        {
            "t": float(start),
            "duration": float(end - start),
            "pitch": int(pitch),
            "velocity": max(1, min(127, int(amplitude * 127))),
        }
        for start, end, pitch, amplitude, _ in note_events
    ]
    notes.sort(key=lambda n: n["t"])
    return notes
