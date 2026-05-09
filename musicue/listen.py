from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from musicue.schemas import CueSheet

SR = 44100

# Stereo pan positions per track type (-1=left, 0=center, 1=right)
_PANS = {
    "kick": 0.0,
    "snare": 0.2,
    "hat": 1.0,
    "hihat": 1.0,
    "downbeat": 0.0,
    "downbeat_pulse": 0.0,
    "vocal_phrase": -0.8,
    "accent": -0.5,
    "drop": 0.0,
}
_DEFAULT_PAN = 0.0

# Click tone frequencies per track type (Hz). MIDI-metronome style:
# very high (3-6.5 kHz) pure sine beeps that sit ABOVE the entire
# musical spectrum and read as obvious synthetic ticks against any
# source material.
_FREQS = {
    "kick": 4000,
    "snare": 5000,
    "hat": 6500,
    "hihat": 6500,
    "downbeat": 3500,
    "downbeat_pulse": 3500,
    "vocal_phrase": 5500,
    "drop": 4500,
    "accent": 4500,
}
_DEFAULT_FREQ = 5000

# Source mix is dropped to 0.15 so the clicks (peaking near full scale)
# clearly dominate. Bumped from 0.25 because the prior pass was still
# subtle against energetic source material.
_SOURCE_GAIN = 0.15
_CLICK_GAIN = 1.0


def _click(strength: float, freq: int, sr: int, decay_ms: float = 30.0) -> np.ndarray:
    """High-pitched MIDI-metronome beep: pure sine with hard onset and
    exponential decay. No noise component (sounded like a cymbal crash);
    the high pitch alone carries the percussive feel."""
    n = int(decay_ms * sr / 1000)
    t = np.arange(n) / sr
    # Sharp exponential decay -- ~75% of energy in the first 8 ms so the
    # click reads as a tick rather than a tone.
    env = np.exp(-t / (decay_ms / 1000 / 4))
    sine = env * np.sin(2 * np.pi * freq * t)
    # tanh(1.5x) gives a perceived ~3 dB loudness boost without harsh
    # hard-clip artifacts. Clicks now peak ~0.9 vs 0.6 before.
    out = np.tanh(1.5 * sine * strength * _CLICK_GAIN)
    return out.astype(np.float32)


def _pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Convert mono to stereo with pan in [-1, 1]."""
    left = mono * np.sqrt(0.5 * (1 - pan))
    right = mono * np.sqrt(0.5 * (1 + pan))
    return np.stack([left, right], axis=1)


def render_click_track(
    cuesheet: CueSheet,
    source_audio: Path | None,
    out_path: Path,
    sr: int = SR,
) -> None:
    n_samples = int(np.ceil(cuesheet.duration_sec * sr))
    mix = np.zeros((n_samples, 2), dtype=np.float32)

    if source_audio and source_audio.exists():
        try:
            data, file_sr = sf.read(str(source_audio))
        except sf.LibsndfileError:
            # Compressed input (m4a/mp3/aac); libsndfile can't decode -- use
            # librosa which proxies through audioread + ffmpeg.
            import librosa

            y, file_sr = librosa.load(str(source_audio), sr=None, mono=False)
            data = (y.T if y.ndim > 1 else y).astype(np.float32)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if file_sr != sr:
            import librosa

            data = librosa.resample(data.T, orig_sr=file_sr, target_sr=sr).T
        end = min(len(data), n_samples)
        mix[:end] += data[:end].astype(np.float32) * _SOURCE_GAIN

    for track in cuesheet.tracks:
        if track.type not in ("impulse", "envelope"):
            continue
        pan = _PANS.get(track.name, _DEFAULT_PAN)
        freq = _FREQS.get(track.name, _DEFAULT_FREQ)
        for event in track.events:
            t = float(event.get("t") if event.get("t") is not None else event.get("t_start", 0.0))
            strength = float(event.get("strength", 0.5))
            click_mono = _click(strength, freq, sr)
            click_stereo = _pan_stereo(click_mono, pan)
            start = int(t * sr)
            end = min(n_samples, start + len(click_stereo))
            length = end - start
            mix[start:end] += click_stereo[:length]

    mix = np.clip(mix, -1.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mix, sr)
