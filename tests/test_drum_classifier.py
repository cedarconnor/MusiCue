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


# ---- Heuristic fallback (no CNN checkpoint) ----


def _synthesize_drum_hits(sr: int = 44100) -> tuple[np.ndarray, list[dict]]:
    """Build a 2-second audio buffer with three distinct synthetic hits at
    known times: a low-freq kick at 0.25s, a noise-burst snare at 0.75s,
    and a high-freq hat at 1.25s. Returns (audio, onset_dicts)."""
    duration = 2.0
    n = int(duration * sr)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    win = int(0.05 * sr)

    # Kick: 60 Hz sine with exponential decay
    t_window = np.arange(win) / sr
    kick = 0.8 * np.sin(2 * np.pi * 60 * t_window) * np.exp(-t_window / 0.02)
    start = int(0.25 * sr)
    audio[start:start + win] += kick.astype(np.float32)

    # Snare: filtered noise centered ~600 Hz (just use white noise — its
    # spectrum has plenty of mid energy and the relative comparison still
    # works out via the bandpass argmax).
    snare = 0.6 * rng.standard_normal(win).astype(np.float32) * np.exp(-t_window / 0.05)
    start = int(0.75 * sr)
    audio[start:start + win] += snare

    # Hat: 7kHz noisy ping with very fast decay
    hat_carrier = 0.4 * np.sin(2 * np.pi * 7000 * t_window)
    hat_noise = 0.3 * rng.standard_normal(win)
    hat = (hat_carrier + hat_noise).astype(np.float32) * np.exp(-t_window / 0.008)
    start = int(1.25 * sr)
    audio[start:start + win] += hat

    onsets = [
        {"t": 0.25, "strength": 0.9, "drum_class": None, "drum_class_conf": None,
         "labels": [], "timescale": "micro"},
        {"t": 0.75, "strength": 0.8, "drum_class": None, "drum_class_conf": None,
         "labels": [], "timescale": "micro"},
        {"t": 1.25, "strength": 0.7, "drum_class": None, "drum_class_conf": None,
         "labels": [], "timescale": "micro"},
    ]
    return audio, onsets


def test_classify_heuristic_routes_synthetic_hits():
    from musicue.analysis.drum_classifier import classify_heuristic

    audio, onsets = _synthesize_drum_hits()
    result = classify_heuristic(onsets, audio, sr=44100)
    classes = [ev["drum_class"] for ev in result]
    assert classes == ["kick", "snare", "hat"], classes
    for ev in result:
        assert 0.0 <= ev["drum_class_conf"] <= 1.0


def test_classify_onsets_batch_falls_back_to_heuristic_when_model_missing(tmp_path):
    """When model_path doesn't exist, classify_onsets_batch should still
    populate drum_class via the heuristic instead of returning untouched."""
    from musicue.analysis.drum_classifier import classify_onsets_batch

    audio, onsets = _synthesize_drum_hits()
    result = classify_onsets_batch(
        onsets, audio, sr=44100,
        model=None, model_path=tmp_path / "nonexistent.pt",
    )
    classes = [ev["drum_class"] for ev in result]
    # Without a CNN we can't guarantee exact label order, but they must
    # not be None anymore.
    assert all(c is not None for c in classes), classes
    # And the obvious-frequency hits should classify reasonably:
    assert result[0]["drum_class"] == "kick"
    assert result[2]["drum_class"] == "hat"


def test_drum_classifier_version_reports_heuristic_when_no_checkpoint(tmp_path):
    from musicue.analysis.drum_classifier import drum_classifier_version

    missing = tmp_path / "missing.pt"
    v = drum_classifier_version(missing)
    assert "heuristic" in v.lower() or v == "not_trained"


# ---- Band-split onset detection ----


def _synth_loop(sr: int = 44100, bars: int = 4, offset: float = 0.5):
    """120 BPM loop: kick on quarters, snare on 2 & 4, closed hat on 8ths.

    Returns (audio, quarter_times, eighth_times, backbeat_times).
    """
    import scipy.signal as ss

    beat = 0.5
    n_quarters = bars * 4
    n = int((offset + n_quarters * beat + 1.0) * sr)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)

    def add(t: float, x: np.ndarray) -> None:
        i = int(round(t * sr))
        audio[i:i + len(x)] += x[: n - i].astype(np.float32)

    # Kick: pitch-swept sine (130 -> 50 Hz) with a long decaying tail.
    tk = np.arange(int(0.45 * sr)) / sr
    freq = 50.0 + 80.0 * np.exp(-tk / 0.03)
    kick = 0.9 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-tk / 0.08)
    kick[:44] *= np.linspace(0.0, 1.0, 44)
    # Snare: band-passed noise + body tone.
    tn = np.arange(int(0.3 * sr)) / sr
    b, a = ss.butter(4, [200, 2000], btype="band", fs=sr)
    snare = (1.5 * ss.lfilter(b, a, rng.standard_normal(len(tn)))
             + 0.3 * np.sin(2 * np.pi * 190 * tn)) * np.exp(-tn / 0.04)
    # Hat: high-passed noise, very short.
    th = np.arange(int(0.1 * sr)) / sr
    bh, ah = ss.butter(4, 7000, btype="high", fs=sr)
    hat = 0.4 * ss.lfilter(bh, ah, rng.standard_normal(len(th))) * np.exp(-th / 0.012)

    quarters = [offset + q * beat for q in range(n_quarters)]
    eighths = [offset + e * beat / 2 for e in range(n_quarters * 2)]
    backbeats = [t for q, t in enumerate(quarters) if q % 4 in (1, 3)]
    for t in quarters:
        add(t, kick)
    for t in backbeats:
        add(t, snare)
    for t in eighths:
        add(t, hat)
    return audio, quarters, eighths, backbeats


def _times(events: list[dict], cls: str) -> list[float]:
    return [e["t"] for e in events if e["drum_class"] == cls]


def _matches(detected: list[float], expected: list[float], tol: float) -> bool:
    """Every expected time has a detection within tol, and vice versa."""
    return (
        all(any(abs(d - e) <= tol for d in detected) for e in expected)
        and all(any(abs(d - e) <= tol for e in expected) for d in detected)
    )


def test_band_split_detects_kick_snare_hat_on_synthetic_loop():
    from musicue.analysis.drum_classifier import detect_drum_onsets_by_band

    sr = 44100
    audio, quarters, eighths, backbeats = _synth_loop(sr)
    events = detect_drum_onsets_by_band(audio, sr=sr)
    tol = 0.03

    kicks = _times(events, "kick")
    snares = _times(events, "snare")
    hats = _times(events, "hat")

    assert _matches(kicks, quarters, tol), kicks
    assert _matches(snares, backbeats, tol), snares
    # Hats: every 8th — the off-beats (not swallowed by the kick tail) and
    # the on-beats (not merged into the simultaneous kick).
    assert _matches(hats, eighths, tol), hats
    off_beats = [t for t in eighths if t not in quarters]
    assert all(any(abs(h - t) <= tol for h in hats) for t in off_beats)

    for e in events:
        assert 0.0 < e["strength"] <= 1.0
        assert 0.0 < e["drum_class_conf"] <= 1.0
        assert e["timescale"] == "micro"
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)


def test_band_split_silence_returns_empty():
    from musicue.analysis.drum_classifier import detect_drum_onsets_by_band

    assert detect_drum_onsets_by_band(np.zeros(44100, dtype=np.float32), sr=44100) == []


def test_heuristic_version_is_bandsplit():
    from musicue.analysis.drum_classifier import drum_classifier_version

    assert drum_classifier_version(None) == "heuristic-bandsplit-v1"


def test_heuristic_path_does_not_import_torch():
    import subprocess
    import sys

    code = (
        "import sys, numpy as np\n"
        "from musicue.analysis.drum_classifier import detect_drum_onsets_by_band\n"
        "detect_drum_onsets_by_band(np.zeros(4096, dtype=np.float32), sr=44100)\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
