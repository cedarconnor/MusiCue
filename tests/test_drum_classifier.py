import numpy as np
import torch

from musicue.analysis.drum_classifier import DRUM_CLASSES, DrumClassifierCNN, classify_onset


def test_drum_classes_list():
    assert "kick" in DRUM_CLASSES
    assert "snare" in DRUM_CLASSES
    assert "hat" in DRUM_CLASSES
    assert len(DRUM_CLASSES) == 6  # kick, snare, hat, tom, cymbal, other


def test_model_forward_shape():
    model = DrumClassifierCNN(n_classes=6)
    model.eval()
    # (batch, channels, mel_bins, time_frames)
    batch = torch.from_numpy(np.zeros((4, 1, 64, 44), dtype=np.float32))  # pyright: ignore[reportPrivateImportUsage]
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (4, 6)


def test_classify_onset_returns_class_and_conf():
    model = DrumClassifierCNN(n_classes=6)
    model.eval()
    audio = np.zeros(2205, dtype=np.float32)  # 50ms at 44100
    drum_class, conf = classify_onset(audio, model, sr=44100)
    assert drum_class in DRUM_CLASSES
    assert 0.0 <= conf <= 1.0


def test_classify_onset_batch():
    from musicue.analysis.drum_classifier import classify_onsets_batch

    model = DrumClassifierCNN(n_classes=6)
    model.eval()
    base = {
        "timescale": "micro",
        "drum_class": None,
        "drum_class_conf": None,
        "labels": [],
    }
    onsets = [
        {"t": 0.5, "strength": 0.9, **base},
        {"t": 1.0, "strength": 0.8, **base},
    ]
    audio = np.zeros(44100, dtype=np.float32)
    result = classify_onsets_batch(onsets, audio, sr=44100, model=model)
    for ev in result:
        assert ev["drum_class"] in DRUM_CLASSES
        assert ev["drum_class_conf"] is not None
