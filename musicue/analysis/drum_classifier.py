"""Drum onset classification.

Two execution paths:

1. **CNN (preferred when a pretrained checkpoint is available)**.
   `models/drum_cnn.pt` is loaded and inference is run per onset.

2. **Heuristic (default, no training required)**. A pure-DSP spectral
   classifier that buckets each onset's energy into low/mid/high bands
   (kick / snare / hat). Imperfect but instantly available — no model
   download, no training step. Used automatically when no checkpoint
   is present.

Inference contract:
    Input:  audio window (np.ndarray, mono, float32) at the given sample rate.
    Output: (drum_class: str in DRUM_CLASSES, confidence: float in [0, 1]).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DRUM_CLASSES = ["kick", "snare", "hat", "tom", "cymbal", "other"]
WINDOW_MS = 50
N_MELS = 64
HOP_LENGTH = 512


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
    model: DrumClassifierCNN,
    sr: int = 44100,
    device: str = "cpu",
) -> tuple[str, float]:
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


# ---- Heuristic (no-model) classifier ----

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


def classify_onsets_batch(
    onsets: list[dict],
    audio: np.ndarray,
    sr: int = 44100,
    model: DrumClassifierCNN | None = None,
    model_path: Path | None = None,
    device: str | None = None,
) -> list[dict]:
    """Classify drum onsets in-place.

    Preference order:
      1. Explicit `model` argument — used for tests.
      2. CNN loaded from `model_path` if it exists.
      3. Heuristic spectral-band fallback (no training/download needed).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        if model_path is None or not model_path.exists():
            # No checkpoint available — heuristic fallback so the cuesheet
            # still gets kick/snare/hat lanes populated.
            return classify_heuristic(onsets, audio, sr=sr)
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        model = DrumClassifierCNN(n_classes=len(DRUM_CLASSES))
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
    return "heuristic"
