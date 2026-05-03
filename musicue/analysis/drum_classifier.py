"""Drum onset classifier CNN.

A small 4-block convolutional network that classifies drum onsets into one of
six categories (kick, snare, hat, tom, cymbal, other) from a 50ms log-mel
patch around the onset time.

Architecture:
    Conv(1->32) -> BN -> ReLU -> MaxPool(2)
    Conv(32->64) -> BN -> ReLU -> MaxPool(2)
    Conv(64->128) -> BN -> ReLU -> MaxPool(2)
    Conv(128->256) -> BN -> ReLU -> MaxPool(2)
    AdaptiveAvgPool(1, 1) -> Linear(256 -> n_classes)

Inference contract:
    Input:  audio window (np.ndarray, mono, float32) at the given sample rate.
    Output: (drum_class: str in DRUM_CLASSES, confidence: float in [0, 1]).

The classifier checkpoint (`models/drum_cnn.pt`) is optional. When the file
is missing, `classify_onsets_batch` returns the input event list unchanged so
callers can use this module as a no-op pass-through during development.
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


def classify_onsets_batch(
    onsets: list[dict],
    audio: np.ndarray,
    sr: int = 44100,
    model: DrumClassifierCNN | None = None,
    model_path: Path | None = None,
    device: str | None = None,
) -> list[dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        if model_path is None or not model_path.exists():
            return onsets  # no model checkpoint -- pass through unchanged
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
    return "not_trained"
