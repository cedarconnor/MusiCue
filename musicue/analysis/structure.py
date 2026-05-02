"""Beat / downbeat / section detection.

Primary backend: All-In-One (https://github.com/mir-aidj/all-in-one), which jointly
predicts tempo, beats, downbeats, and functional segments. Falls back to librosa's
beat tracker when All-In-One is unavailable or fails (e.g. on Windows where the
package can be hard to install).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def allin1_version() -> str:
    """Return the installed allin1 package version, or 'unknown' if not installed."""
    try:
        return _pkg_version("allin1")
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _librosa_fallback(audio_path: Path) -> dict:
    """Beat detection via librosa.beat.beat_track. No section detection."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # librosa.beat.beat_track may return tempo as np.ndarray or np.float64.
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo.flat[0]) if tempo.size > 0 else 0.0
    else:
        tempo_val = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beats = [
        {
            "t": float(t),
            "beat_in_bar": (i % 4) + 1,
            "bar": i // 4 + 1,
            "is_downbeat": i % 4 == 0,
            "confidence": 0.7,
            "timescale": "micro",
        }
        for i, t in enumerate(beat_times)
    ]
    return {
        "tempo": {
            "bpm_global": tempo_val,
            "bpm_curve": [{"t": 0.0, "bpm": tempo_val}],
            "time_signature": [4, 4],
        },
        "beats": beats,
        "sections": [],
    }


def detect_structure(audio_path: Path, backend: str = "allin1") -> dict:
    """Detect tempo, beats/downbeats, and functional sections.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file (any format librosa/soundfile can load).
    backend : str
        "allin1" (default) tries All-In-One first and falls back to librosa on
        error. Any other value goes straight to the librosa fallback.

    Returns
    -------
    dict with keys ``tempo``, ``beats``, ``sections`` matching the M1 schema.
    """
    if backend == "allin1":
        try:
            import allin1  # type: ignore[import-not-found]

            result = allin1.analyze(str(audio_path))
            downbeat_set = set(result.downbeats)
            beats: list[dict] = []
            bar = 0
            beat_in_bar = 0
            for t in result.beats:
                if t in downbeat_set:
                    bar += 1
                    beat_in_bar = 1
                else:
                    beat_in_bar += 1
                beats.append(
                    {
                        "t": float(t),
                        "beat_in_bar": beat_in_bar,
                        "bar": bar,
                        "is_downbeat": t in downbeat_set,
                        "confidence": 0.9,
                        "timescale": "micro",
                    }
                )
            sections = [
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "label": str(seg.label),
                    "confidence": 0.9,
                    "timescale": "macro",
                }
                for seg in result.segments
            ]
            return {
                "tempo": {
                    "bpm_global": float(result.bpm),
                    "bpm_curve": [{"t": 0.0, "bpm": float(result.bpm)}],
                    "time_signature": [4, 4],
                },
                "beats": beats,
                "sections": sections,
            }
        except Exception:
            # Fall through to librosa fallback on any error (import, model load,
            # inference, attribute access on result, etc.).
            pass
    return _librosa_fallback(audio_path)
