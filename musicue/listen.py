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

# Click tone frequencies per track type (Hz). Pushed up into the
# 1.2-4 kHz range so the clicks sit ABOVE most musical content (which
# generally lives below ~1 kHz fundamental); makes them clearly audible
# over the source mix.
_FREQS = {
    "kick": 1500,
    "snare": 2000,
    "hat": 3500,
    "hihat": 3500,
    "downbeat": 1200,
    "downbeat_pulse": 1200,
    "vocal_phrase": 2500,
    "drop": 1800,
    "accent": 2200,
}
_DEFAULT_FREQ = 2000

# Source mix and click levels. Source is dropped to 0.25 so clicks at
# unity dominate the mix; clicks themselves get a sharp noise transient
# layered on top of the sine for a "tick" attack that survives over
# music with strong transients of its own.
_SOURCE_GAIN = 0.25
_CLICK_GAIN = 1.0


def _click(strength: float, freq: int, sr: int, decay_ms: float = 18.0) -> np.ndarray:
    """Sharp metronome-style click: noise burst (first 1.5 ms) layered with
    an exponentially decaying sine. The noise burst gives a percussive
    transient that punches through the music; the sine carries the
    pitch that disambiguates kick/snare/hihat etc."""
    n = int(decay_ms * sr / 1000)
    t = np.arange(n) / sr

    # Sine carrier with quick decay envelope.
    env = np.exp(-t / (decay_ms / 1000 / 3))
    sine = env * np.sin(2 * np.pi * freq * t)

    # Noise transient on the first ~1.5 ms only -- a sharp tick attack.
    noise_len = max(1, int(0.0015 * sr))
    noise_env = np.exp(-np.arange(noise_len) / (noise_len / 4))
    noise = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed=int(freq))
    noise[:noise_len] = rng.standard_normal(noise_len).astype(np.float32) * noise_env

    wave = 0.7 * sine + 0.3 * noise
    out = wave * strength * _CLICK_GAIN
    # Soft-clip to [-1, 1] without harsh hard clipping artifacts.
    out = np.tanh(out)
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
