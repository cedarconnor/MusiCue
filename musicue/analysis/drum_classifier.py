"""Drum onset classification.

Two execution paths:

1. **CNN (preferred when a pretrained checkpoint is available)**.
   `models/drum_cnn.pt` is loaded and inference is run per onset.

2. **Band-split heuristic (default, no training required)**. Instead of
   detecting one onset stream and then giving each onset a single class,
   :func:`detect_drum_onsets_by_band` computes a separate spectral-flux
   onset envelope for each drum-relevant band (kick / snare / hat) and
   peak-picks each band independently. A kick and a hat that land on the
   same beat therefore produce two events, and a decaying kick tail (which
   has *negative* flux) no longer swallows the off-beat hats. No model
   download, no training step. Used automatically when no checkpoint is
   present.

   Known limitation: a band's envelope is normalized against its own
   loudest peaks, so on a kit with no snare at all the kick's beater click
   spilling into the mid band can surface as low-confidence "snare" events.

``torch`` is imported lazily: only the CNN path needs it.

Inference contract (CNN):
    Input:  audio window (np.ndarray, mono, float32) at the given sample rate.
    Output: (drum_class: str in DRUM_CLASSES, confidence: float in [0, 1]).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np

DRUM_CLASSES = ["kick", "snare", "hat", "tom", "cymbal", "other"]
WINDOW_MS = 50
N_MELS = 64
HOP_LENGTH = 512

HEURISTIC_VERSION = "heuristic-bandsplit-v1"


@lru_cache(maxsize=1)
def _cnn_classes() -> tuple[type, type]:
    """Define the torch model classes on first use (keeps torch import lazy)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(F.relu(self.bn(self.conv(x))))

    class DrumClassifierCNN(nn.Module):
        def __init__(self, n_classes: int = 6) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                _ConvBlock(1, 32),
                _ConvBlock(32, 64),
                _ConvBlock(64, 128),
                _ConvBlock(128, 256),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(256, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.blocks(x)
            x = self.pool(x).flatten(1)
            return self.head(x)

    return _ConvBlock, DrumClassifierCNN


def __getattr__(name: str) -> Any:
    # PEP 562: resolve the torch-backed classes only when someone asks.
    if name == "DrumClassifierCNN":
        return _cnn_classes()[1]
    if name == "_ConvBlock":
        return _cnn_classes()[0]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _onset_to_mel(audio_window: np.ndarray, sr: int = 44100) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio_window, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def _extract_window(
    audio: np.ndarray, t: float, sr: int, window_ms: int = WINDOW_MS
) -> np.ndarray:
    n = int(window_ms * sr / 1000)
    center = int(t * sr)
    start = max(0, center - n // 2)
    end = min(len(audio), start + n)
    chunk = audio[start:end]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    return chunk


def classify_onset(
    audio_window: np.ndarray,
    model: Any,
    sr: int = 44100,
    device: str = "cpu",
) -> tuple[str, float]:
    import torch
    import torch.nn.functional as F

    mel = _onset_to_mel(audio_window, sr=sr)
    target_frames = 44
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        mel = mel[:, :target_frames]
    tensor = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)  # pyright: ignore[reportPrivateImportUsage]
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return DRUM_CLASSES[idx], float(probs[idx])


# ---- Heuristic (no-model) paths ----

# Frequency bands used by the heuristic. Tuned for a typical drum kit:
#   kick:  fundamental + first harmonic mostly under ~180 Hz
#   snare: shell + early decay sits in 150-2000 Hz
#   hat:   shimmer / sizzle dominates above ~5 kHz
_HEURISTIC_BANDS: dict[str, tuple[float, float]] = {
    "kick": (20.0, 180.0),
    "snare": (180.0, 2000.0),
    "hat": (5000.0, 16000.0),
}


def _band_energy(audio_window: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """RMS energy in the [lo, hi] Hz band of `audio_window` via rFFT."""
    if len(audio_window) == 0:
        return 0.0
    spec = np.fft.rfft(audio_window)
    freqs = np.fft.rfftfreq(len(audio_window), 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(np.abs(spec[mask]) ** 2)))


def classify_heuristic(
    onsets: list[dict],
    audio: np.ndarray,
    sr: int = 44100,
    window_ms: int = WINDOW_MS,
) -> list[dict]:
    """Spectral-band fallback classifier. Mutates and returns `onsets`.

    For each onset, computes RMS energy in three drum-relevant bands
    and picks the dominant band as the drum class. Confidence is the
    dominant band's share of total band energy (≈0.33 = no signal,
    ≈1.0 = entirely in one band).
    """
    band_names = list(_HEURISTIC_BANDS.keys())
    for event in onsets:
        t = float(event["t"])
        window = _extract_window(audio, t, sr, window_ms=window_ms)
        energies = np.array([
            _band_energy(window, sr, *_HEURISTIC_BANDS[name])
            for name in band_names
        ])
        total = float(energies.sum())
        if total <= 1e-9:
            event["drum_class"] = "other"
            event["drum_class_conf"] = 0.0
            continue
        idx = int(energies.argmax())
        event["drum_class"] = band_names[idx]
        event["drum_class_conf"] = float(energies[idx] / total)
    return onsets


# Bands for the band-split onset detector. Slightly narrower low band than
# the per-onset heuristic so a kick's upper harmonics don't dominate.
_ONSET_BANDS: dict[str, tuple[float, float]] = {
    "kick": (20.0, 150.0),
    "snare": (150.0, 2500.0),
    "hat": (5000.0, 16000.0),
}
_BAND_N_FFT = 2048
# log1p(gamma * power / band_max): dynamic-range compression before flux.
_BAND_LOG_GAMMA = 1000.0
# Peak height (relative to the band's 99th-percentile peak) needed to emit.
_BAND_MIN_STRENGTH = 0.4


def detect_drum_onsets_by_band(
    audio: np.ndarray,
    sr: int = 44100,
    hop_length: int = HOP_LENGTH,
) -> list[dict]:
    """Band-split drum onset detection on a (mono) drums stem.

    For each of kick / snare / hat, compute a spectral-flux onset envelope
    restricted to that band's STFT bins and peak-pick it independently.
    Emits one OnsetEvent-shaped dict per (band, onset), sorted by time.

    * ``strength``: peak height / band's 99th-percentile peak, clipped [0, 1].
    * ``drum_class_conf``: this band's share of the (normalized) onset
      salience summed across all bands at that frame — ≈1.0 for an isolated
      hit, ≈0.33–0.5 when several bands fire together.
    """
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < _BAND_N_FFT or not np.any(y):
        return []

    power = np.abs(librosa.stft(y, n_fft=_BAND_N_FFT, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_BAND_N_FFT)
    wait = max(1, int(0.05 * sr / hop_length))

    norm_envs: dict[str, np.ndarray] = {}
    for name, (lo, hi) in _ONSET_BANDS.items():
        mask = (freqs >= lo) & (freqs < min(hi, sr / 2.0))
        band = power[mask]
        band_max = float(band.max()) if band.size else 0.0
        if band_max <= 0.0:
            continue
        compressed = np.log1p(_BAND_LOG_GAMMA * band / band_max)
        # center=False: the STFT above is already centered, so librosa's
        # extra centering pad would push peaks ~2 frames late.
        env = librosa.onset.onset_strength(
            S=compressed, sr=sr, hop_length=hop_length, center=False
        )
        ref = float(np.percentile(env, 99)) if env.size else 0.0
        if ref <= 0.0:
            continue
        norm_envs[name] = env / ref

    if not norm_envs:
        return []
    n_frames = min(len(e) for e in norm_envs.values())
    stacked = np.vstack([e[:n_frames] for e in norm_envs.values()])
    # Salience at each frame, max-pooled over ±1 frame so near-coincident
    # hits in different bands share the same denominator.
    pooled = np.maximum.reduce([
        np.roll(stacked, -1, axis=1), stacked, np.roll(stacked, 1, axis=1)
    ])
    total = pooled.sum(axis=0)

    events: list[dict] = []
    for row, (name, env) in enumerate(norm_envs.items()):
        env = env[:n_frames]
        peaks = librosa.util.peak_pick(
            env, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
            delta=0.1, wait=wait,
        )
        peaks = peaks[env[peaks] >= _BAND_MIN_STRENGTH]
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        for f, t in zip(peaks, times):
            denom = float(total[f])
            share = float(pooled[row, f]) / denom if denom > 0 else 0.0
            events.append({
                "t": float(t),
                "strength": float(np.clip(env[f], 0.0, 1.0)),
                "timescale": "micro",
                "drum_class": name,
                "drum_class_conf": float(np.clip(share, 0.0, 1.0)),
                "labels": [],
            })
    events.sort(key=lambda e: (e["t"], e["drum_class"]))
    return events


def classify_onsets_batch(
    onsets: list[dict],
    audio: np.ndarray,
    sr: int = 44100,
    model: Any | None = None,
    model_path: Path | None = None,
    device: str | None = None,
) -> list[dict]:
    """Classify drum onsets in-place.

    Preference order:
      1. Explicit `model` argument — used for tests.
      2. CNN loaded from `model_path` if it exists.
      3. Heuristic spectral-band fallback (no training/download needed).
    """
    if model is None and (model_path is None or not model_path.exists()):
        # No checkpoint available — heuristic fallback so the cuesheet
        # still gets kick/snare/hat lanes populated. (The pipeline prefers
        # detect_drum_onsets_by_band in this case; this keeps the
        # per-onset API working for callers that already have onsets.)
        return classify_heuristic(onsets, audio, sr=sr)

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        assert model_path is not None
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        model = _cnn_classes()[1](n_classes=len(DRUM_CLASSES))
        model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    for event in onsets:
        t = float(event["t"])
        window = _extract_window(audio, t, sr)
        drum_class, conf = classify_onset(window, model, sr=sr, device=device)
        event["drum_class"] = drum_class
        event["drum_class_conf"] = conf
    return onsets


def drum_classifier_version(model_path: Path | None = None) -> str:
    if model_path and model_path.exists():
        import hashlib

        h = hashlib.sha256(model_path.read_bytes()).hexdigest()[:8]
        return f"cnn-{h}"
    return HEURISTIC_VERSION
