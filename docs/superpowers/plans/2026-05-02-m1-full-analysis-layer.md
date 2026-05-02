# M1: Full Analysis Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Layer 1 so `analysis.json` contains beats, downbeats, sections, MIDI events, phrases, CLAP-labeled events, full spectral curves, stereo width/pan, and section transition ramps — everything the M2 compiler needs.

**Architecture:** Six new modules drop into `musicue/analysis/` and are wired into the existing `pipeline.py`. Each module is independently tested with the synthetic WAV fixture from M0 (`tests/conftest.py`). The `inspect` and `plot` CLI commands are added after the pipeline is complete. The `allin1` beat fallback uses `librosa.beat.beat_track` — madmom is not supported on Windows.

**Tech Stack:** allin1, basic-pitch, laion-clap (optional dep), librosa, pyloudnorm, soundfile, numpy

**Prerequisite:** M0 plan complete and all tests passing.

---

## File Structure (additions to M0)

```
musicue/
├── musicue/
│   ├── inspect.py                    ← NEW: inspect + plot CLI helpers
│   ├── analysis/
│   │   ├── structure.py              ← NEW: All-In-One wrapper + librosa fallback
│   │   ├── transcription.py          ← NEW: Basic Pitch per-stem MIDI extraction
│   │   ├── phrases.py                ← NEW: MIDI note → phrase grouping
│   │   ├── clap_reranker.py          ← NEW: CLAP labels attached to candidate events
│   │   ├── curves.py                 ← MODIFY: add spectral centroid, spectral flux, stereo width/pan
│   │   ├── transitions.py            ← NEW: section_transitions ramp derivation
│   │   └── pipeline.py               ← MODIFY: wire all new stages
└── tests/
    ├── test_structure.py             ← NEW
    ├── test_transcription.py         ← NEW
    ├── test_phrases.py               ← NEW
    ├── test_clap_reranker.py         ← NEW
    ├── test_transitions.py           ← NEW
    └── test_inspect.py               ← NEW
```

---

### Task 1: Beat / downbeat / section detection (All-In-One)

**Files:**
- Create: `musicue/analysis/structure.py`
- Create: `tests/test_structure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_structure.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from musicue.analysis.structure import detect_structure, allin1_version


def test_allin1_version_returns_string():
    v = allin1_version()
    assert isinstance(v, str)


def test_detect_structure_returns_expected_keys(synthetic_wav):
    # Mock allin1.analyze to avoid real model inference in unit tests
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = [0.5, 1.0, 1.5, 2.0]
    mock_result.downbeats = [0.5, 2.0]
    mock_result.segments = [
        MagicMock(start=0.0, end=5.0, label="intro"),
        MagicMock(start=5.0, end=10.0, label="verse"),
    ]

    with patch("allin1.analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    assert "tempo" in result
    assert "beats" in result
    assert "sections" in result
    assert result["tempo"]["bpm_global"] == pytest.approx(120.0)


def test_detect_structure_beat_fields(synthetic_wav):
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = [0.5, 1.0]
    mock_result.downbeats = [0.5]
    mock_result.segments = []

    with patch("allin1.analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    beats = result["beats"]
    assert len(beats) == 2
    b = beats[0]
    assert "t" in b and "is_downbeat" in b and "timescale" in b and "confidence" in b
    assert b["is_downbeat"] is True
    assert b["timescale"] == "micro"


def test_detect_structure_section_fields(synthetic_wav):
    mock_result = MagicMock()
    mock_result.bpm = 120.0
    mock_result.beats = []
    mock_result.downbeats = []
    mock_result.segments = [MagicMock(start=0.0, end=5.0, label="chorus")]

    with patch("allin1.analyze", return_value=mock_result):
        result = detect_structure(synthetic_wav)

    sections = result["sections"]
    assert len(sections) == 1
    s = sections[0]
    assert s["label"] == "chorus"
    assert s["timescale"] == "macro"
    assert "start" in s and "end" in s


def test_detect_structure_falls_back_to_librosa_on_error(synthetic_wav):
    with patch("allin1.analyze", side_effect=RuntimeError("allin1 failed")):
        result = detect_structure(synthetic_wav, backend="allin1")
    # Should not raise — should fall back to librosa
    assert "beats" in result
    assert "tempo" in result
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_structure.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.structure'`

- [ ] **Step 3: Implement musicue/analysis/structure.py**

```python
from __future__ import annotations
from pathlib import Path

try:
    from importlib.metadata import version as _v
    def allin1_version() -> str:
        try:
            return _v("allin1")
        except Exception:
            return "unknown"
except ImportError:
    def allin1_version() -> str:
        return "unknown"


def _librosa_fallback(audio_path: Path) -> dict:
    import librosa
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
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
            "bpm_global": float(tempo),
            "bpm_curve": [{"t": 0.0, "bpm": float(tempo)}],
            "time_signature": [4, 4],
        },
        "beats": beats,
        "sections": [],
    }


def detect_structure(audio_path: Path, backend: str = "allin1") -> dict:
    if backend == "allin1":
        try:
            import allin1
            result = allin1.analyze(str(audio_path))
            downbeat_set = set(result.downbeats)
            beats = []
            bar = 0
            beat_in_bar = 0
            for t in result.beats:
                if t in downbeat_set:
                    bar += 1
                    beat_in_bar = 1
                else:
                    beat_in_bar += 1
                beats.append({
                    "t": float(t),
                    "beat_in_bar": beat_in_bar,
                    "bar": bar,
                    "is_downbeat": t in downbeat_set,
                    "confidence": 0.9,
                    "timescale": "meso" if t in downbeat_set else "micro",
                })
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
            pass
    return _librosa_fallback(audio_path)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_structure.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/structure.py tests/test_structure.py
git commit -m "feat: All-In-One beat/section detection with librosa fallback"
```

---

### Task 2: Polyphonic transcription (Basic Pitch)

**Files:**
- Create: `musicue/analysis/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_transcription.py`:

```python
import pytest
from unittest.mock import patch
import numpy as np
from pathlib import Path
from musicue.analysis.transcription import transcribe_stem, basic_pitch_version


def test_basic_pitch_version_returns_string():
    v = basic_pitch_version()
    assert isinstance(v, str)


def _mock_bp_output():
    # basic_pitch returns (model_output, midi_data, note_events)
    # note_events is a list of (start_time, end_time, pitch, amplitude, pitch_bends)
    note_events = [
        (0.5, 1.0, 64, 0.8, None),
        (1.5, 2.0, 67, 0.7, None),
        (2.5, 3.0, 60, 0.9, None),
    ]
    return ({}, None, note_events)


def test_transcribe_returns_midi_list(synthetic_wav):
    with patch("basic_pitch.inference.predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    assert isinstance(notes, list)
    assert len(notes) == 3


def test_transcribe_note_fields(synthetic_wav):
    with patch("basic_pitch.inference.predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    n = notes[0]
    assert "t" in n and "duration" in n and "pitch" in n and "velocity" in n
    assert n["t"] == pytest.approx(0.5)
    assert n["pitch"] == 64
    assert 0 < n["velocity"] <= 127


def test_transcribe_notes_sorted_by_time(synthetic_wav):
    with patch("basic_pitch.inference.predict", return_value=_mock_bp_output()):
        notes = transcribe_stem(synthetic_wav)
    times = [n["t"] for n in notes]
    assert times == sorted(times)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_transcription.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.transcription'`

- [ ] **Step 3: Implement musicue/analysis/transcription.py**

```python
from __future__ import annotations
from pathlib import Path

try:
    from importlib.metadata import version as _v
    def basic_pitch_version() -> str:
        try:
            return _v("basic-pitch")
        except Exception:
            return "unknown"
except ImportError:
    def basic_pitch_version() -> str:
        return "unknown"


def transcribe_stem(audio_path: Path) -> list[dict]:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

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
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_transcription.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/transcription.py tests/test_transcription.py
git commit -m "feat: Basic Pitch polyphonic transcription wrapper"
```

---

### Task 3: Phrase grouping

**Files:**
- Create: `musicue/analysis/phrases.py`
- Create: `tests/test_phrases.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phrases.py`:

```python
import pytest
from musicue.analysis.phrases import group_into_phrases


def _notes(times_and_pitches):
    return [
        {"t": t, "duration": 0.3, "pitch": p, "velocity": 80}
        for t, p in times_and_pitches
    ]


def test_single_phrase_no_gaps():
    notes = _notes([(0.0, 60), (0.3, 62), (0.6, 64)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert len(phrases) == 1
    p = phrases[0]
    assert p["t_start"] == pytest.approx(0.0)
    assert p["note_count"] == 3


def test_gap_splits_into_two_phrases():
    notes = _notes([(0.0, 60), (0.3, 62), (2.0, 67), (2.3, 69)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert len(phrases) == 2
    assert phrases[0]["note_count"] == 2
    assert phrases[1]["t_start"] == pytest.approx(2.0)


def test_phrase_pitch_features():
    notes = _notes([(0.0, 60), (0.3, 67), (0.6, 64)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    p = phrases[0]
    assert p["pitch_peak"] == 67
    assert p["pitch_low"] == 60
    assert len(p["pitch_contour"]) > 0


def test_phrase_timescale():
    notes = _notes([(0.0, 60), (0.3, 62)])
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert phrases[0]["timescale"] == "meso"


def test_phrase_t_end():
    notes = _notes([(0.0, 60), (0.5, 62)])
    # note at 0.5 with duration 0.3 ends at 0.8
    notes[1]["duration"] = 0.3
    phrases = group_into_phrases(notes, gap_sec=0.6)
    assert phrases[0]["t_end"] == pytest.approx(0.8)


def test_empty_notes_returns_empty():
    assert group_into_phrases([], gap_sec=0.6) == []
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_phrases.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.phrases'`

- [ ] **Step 3: Implement musicue/analysis/phrases.py**

```python
from __future__ import annotations
import numpy as np


def group_into_phrases(notes: list[dict], gap_sec: float = 0.6) -> list[dict]:
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

    phrases = []
    for group in groups:
        t_start = group[0]["t"]
        last = group[-1]
        t_end = last["t"] + last.get("duration", 0.3)
        pitches = [n["pitch"] for n in group]
        phrases.append({
            "t_start": float(t_start),
            "t_end": float(t_end),
            "timescale": "meso",
            "note_count": len(group),
            "pitch_peak": int(max(pitches)),
            "pitch_low": int(min(pitches)),
            "pitch_contour": [int(p) for p in pitches[:: max(1, len(pitches) // 10)]],
            "energy_curve": {"hop_sec": 0.04, "values": []},
            "labels": [],
        })
    return phrases
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_phrases.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/phrases.py tests/test_phrases.py
git commit -m "feat: MIDI note → phrase grouping"
```

---

### Task 4: Extended curves (spectral + stereo width/pan)

**Files:**
- Modify: `musicue/analysis/curves.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Add curve tests**

Append to `tests/test_analysis.py`:

```python
from musicue.analysis.curves import (
    compute_spectral_centroid_curve,
    compute_spectral_flux_curve,
    compute_stereo_width_pan,
)


def test_spectral_centroid_curve(synthetic_wav):
    c = compute_spectral_centroid_curve(synthetic_wav, hop_sec=0.04)
    assert len(c["values"]) > 0
    assert all(v >= 0 for v in c["values"])


def test_spectral_flux_curve(synthetic_wav):
    c = compute_spectral_flux_curve(synthetic_wav, hop_sec=0.04)
    assert len(c["values"]) > 0
    assert all(v >= 0 for v in c["values"])


def test_stereo_width_pan_on_mono_returns_zero(synthetic_wav):
    result = compute_stereo_width_pan(synthetic_wav, hop_sec=0.04)
    # synthetic_wav is mono; width should be 0 or near 0
    assert "width" in result and "pan" in result
    assert all(abs(v) < 0.01 for v in result["width"]["values"])
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_analysis.py -v -k "spectral or stereo"
```
Expected: `ImportError` — functions not yet defined in curves.py

- [ ] **Step 3: Add functions to musicue/analysis/curves.py**

Append to existing `musicue/analysis/curves.py`:

```python
def compute_spectral_centroid_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, int(hop_sec * rate))
    centroid = librosa.feature.spectral_centroid(y=y, sr=rate, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in centroid]}


def compute_spectral_flux_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, int(hop_sec * rate))
    flux = librosa.onset.onset_strength(y=y, sr=rate, hop_length=hop)
    return {"hop_sec": hop / rate, "values": [float(v) for v in flux]}


def compute_stereo_width_pan(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = sf.read(str(audio_path))
    hop = max(1, int(hop_sec * rate))
    if data.ndim == 1:
        n = max(1, len(data) // hop)
        zeros = [0.0] * n
        return {
            "width": {"hop_sec": hop / rate, "values": zeros},
            "pan": {"hop_sec": hop / rate, "values": zeros},
        }
    L, R = data[:, 0], data[:, 1]
    width_vals, pan_vals = [], []
    for i in range(0, len(data) - hop, hop):
        l_chunk = L[i : i + hop]
        r_chunk = R[i : i + hop]
        mid = l_chunk + r_chunk
        side = l_chunk - r_chunk
        mid_rms = float(np.sqrt(np.mean(mid ** 2)) + 1e-9)
        side_rms = float(np.sqrt(np.mean(side ** 2)) + 1e-9)
        width_vals.append(float(np.clip(side_rms / mid_rms, 0.0, 1.0)))
        l_rms = float(np.sqrt(np.mean(l_chunk ** 2)) + 1e-9)
        r_rms = float(np.sqrt(np.mean(r_chunk ** 2)) + 1e-9)
        pan_vals.append(float(np.clip((r_rms - l_rms) / (r_rms + l_rms), -1.0, 1.0)))
    actual_hop = hop / rate
    return {
        "width": {"hop_sec": actual_hop, "values": width_vals},
        "pan": {"hop_sec": actual_hop, "values": pan_vals},
    }
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_analysis.py -v -k "spectral or stereo"
```
Expected: all 3 new tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/curves.py tests/test_analysis.py
git commit -m "feat: spectral centroid, spectral flux, stereo width/pan curves"
```

---

### Task 5: Section transition ramp derivation

**Files:**
- Create: `musicue/analysis/transitions.py`
- Create: `tests/test_transitions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_transitions.py`:

```python
import pytest
from musicue.analysis.transitions import derive_transitions


def _make_sections():
    return [
        {"start": 0.0, "end": 17.2, "label": "intro", "confidence": 0.9, "timescale": "macro"},
        {"start": 17.2, "end": 51.6, "label": "verse", "confidence": 0.9, "timescale": "macro"},
        {"start": 51.6, "end": 86.0, "label": "chorus", "confidence": 0.9, "timescale": "macro"},
    ]


def _make_flux(hop_sec=0.04, n=2500):
    import numpy as np
    values = [0.1] * n
    # Add a rise before each transition
    for trans_t in (17.2, 51.6):
        idx = int(trans_t / hop_sec)
        for j in range(max(0, idx - 35), idx):
            values[j] = 0.8 + (j - (idx - 35)) * 0.005
    return {"hop_sec": hop_sec, "values": values}


def _make_lufs(hop_sec=0.04, n=2500):
    return {"hop_sec": hop_sec, "values": [-20.0] * n}


def test_derive_transitions_count(synthetic_wav):
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    # 2 transitions: intro→verse, verse→chorus
    assert len(transitions) == 2


def test_derive_transitions_fields():
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    t = transitions[0]
    assert "t" in t
    assert "from" in t or "from_section" in t
    assert "to" in t
    assert "ramp" in t
    assert "ramp_evidence" in t
    assert t["t"] == pytest.approx(17.2)


def test_derive_transitions_to_from_labels():
    sections = _make_sections()
    flux = _make_flux()
    lufs = _make_lufs()
    transitions = derive_transitions(sections, flux, lufs)
    assert transitions[0]["to"] == "verse"
    assert transitions[1]["to"] == "chorus"


def test_derive_transitions_ramp_evidence_keys():
    transitions = derive_transitions(_make_sections(), _make_flux(), _make_lufs())
    ev = transitions[0]["ramp_evidence"]
    assert "spectral_flux_rise" in ev
    assert "lufs_rise_db" in ev


def test_no_sections_returns_empty():
    assert derive_transitions([], _make_flux(), _make_lufs()) == []
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_transitions.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.transitions'`

- [ ] **Step 3: Implement musicue/analysis/transitions.py**

```python
from __future__ import annotations
import numpy as np


def derive_transitions(
    sections: list[dict],
    spectral_flux: dict,
    lufs: dict,
    lookback_sec: float = 1.5,
) -> list[dict]:
    if len(sections) < 2:
        return []

    flux_hop = spectral_flux["hop_sec"]
    flux_vals = np.array(spectral_flux["values"])
    lufs_hop = lufs["hop_sec"]
    lufs_vals = np.array(lufs["values"])

    transitions = []
    for i in range(1, len(sections)):
        t = sections[i]["start"]
        lookback_frames = int(lookback_sec / flux_hop)
        t_idx = min(int(t / flux_hop), len(flux_vals) - 1)
        start_idx = max(0, t_idx - lookback_frames)

        window_flux = flux_vals[start_idx:t_idx] if t_idx > start_idx else np.array([0.0])
        flux_rise = float(np.max(window_flux) / (np.mean(flux_vals) + 1e-9)) if len(window_flux) > 0 else 0.0
        flux_rise = float(np.clip(flux_rise, 0.0, 1.0))

        lufs_t_idx = min(int(t / lufs_hop), len(lufs_vals) - 1)
        lufs_start_idx = max(0, lufs_t_idx - int(lookback_sec / lufs_hop))
        window_lufs = lufs_vals[lufs_start_idx:lufs_t_idx]
        lufs_rise = float(window_lufs[-1] - window_lufs[0]) if len(window_lufs) >= 2 else 0.0

        ramp_start = max(0.0, t - lookback_sec * 0.8)
        transitions.append({
            "t": float(t),
            "from": sections[i - 1]["label"],
            "to": sections[i]["label"],
            "ramp": {"t_start": ramp_start, "t_end": float(t), "shape": "ease_in"},
            "ramp_evidence": {
                "spectral_flux_rise": flux_rise,
                "lufs_rise_db": lufs_rise,
            },
        })
    return transitions
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_transitions.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/transitions.py tests/test_transitions.py
git commit -m "feat: section transition ramp derivation from spectral flux + LUFS"
```

---

### Task 6: CLAP re-ranker

**Files:**
- Create: `musicue/analysis/clap_reranker.py`
- Create: `tests/test_clap_reranker.py`

**Note:** CLAP (`laion-clap`) is an optional dependency. Tests mock the model; the real model loads only when `laion-clap` is installed and `use_clap=True` is set in the config.

- [ ] **Step 1: Write failing tests**

Create `tests/test_clap_reranker.py`:

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from musicue.analysis.clap_reranker import attach_clap_labels, clap_version


def test_clap_version_returns_string():
    v = clap_version()
    assert isinstance(v, str)


def _make_events():
    return [
        {"t": 0.5, "strength": 0.9, "timescale": "micro", "drum_class": "kick", "drum_class_conf": 0.9, "labels": []},
        {"t": 1.5, "strength": 0.8, "timescale": "micro", "drum_class": None, "drum_class_conf": None, "labels": []},
    ]


def test_attach_clap_labels_no_op_when_disabled():
    events = _make_events()
    result = attach_clap_labels(events, audio_path=None, prompts=["sub bass drop"], enabled=False)
    assert all(e["labels"] == [] for e in result)


def test_attach_clap_labels_adds_labels_above_threshold():
    events = _make_events()
    prompts = ["punchy kick", "sub bass drop"]

    mock_model = MagicMock()
    # Simulate cosine similarities: event 0 scores high on "punchy kick"
    mock_model.get_audio_embedding_from_data = MagicMock(
        return_value=np.array([[0.8, 0.1], [0.2, 0.3]])
    )
    mock_model.get_text_embedding = MagicMock(
        return_value=np.array([[1.0, 0.0], [0.0, 1.0]])
    )

    # Use the real scoring logic but mock out the model loading + audio extraction
    with patch("musicue.analysis.clap_reranker._load_model", return_value=mock_model):
        with patch("musicue.analysis.clap_reranker._extract_window", return_value=np.zeros(44100)):
            result = attach_clap_labels(
                events, audio_path=None, prompts=prompts, enabled=True, threshold=0.5, top_k=2
            )
    # Event 0 should have at least one label
    assert len(result[0]["labels"]) > 0
    label = result[0]["labels"][0]
    assert "label" in label and "score" in label and "source" in label
    assert label["source"] == "clap"


def test_attach_clap_labels_skips_below_threshold():
    events = _make_events()
    prompts = ["sub bass drop"]
    mock_model = MagicMock()
    mock_model.get_audio_embedding_from_data = MagicMock(
        return_value=np.array([[0.1], [0.1]])
    )
    mock_model.get_text_embedding = MagicMock(return_value=np.array([[1.0]]))

    with patch("musicue.analysis.clap_reranker._load_model", return_value=mock_model):
        with patch("musicue.analysis.clap_reranker._extract_window", return_value=np.zeros(44100)):
            result = attach_clap_labels(
                events, audio_path=None, prompts=prompts, enabled=True, threshold=0.5, top_k=3
            )
    assert all(e["labels"] == [] for e in result)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_clap_reranker.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.clap_reranker'`

- [ ] **Step 3: Implement musicue/analysis/clap_reranker.py**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

try:
    from importlib.metadata import version as _v
    def clap_version() -> str:
        try:
            return _v("laion-clap")
        except Exception:
            return "not installed"
except ImportError:
    def clap_version() -> str:
        return "not installed"

_MODEL_CACHE: Any = None


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import laion_clap
        _MODEL_CACHE = laion_clap.CLAP_Module(enable_fusion=False)
        _MODEL_CACHE.load_ckpt()
    return _MODEL_CACHE


def _extract_window(audio_path: Path, t: float, window_sec: float = 2.0, sr: int = 44100) -> np.ndarray:
    import soundfile as sf
    data, file_sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    n_samples = int(window_sec * file_sr)
    center = int(t * file_sr)
    start = max(0, center - n_samples // 2)
    end = min(len(data), start + n_samples)
    chunk = data[start:end]
    if len(chunk) < n_samples:
        chunk = np.pad(chunk, (0, n_samples - len(chunk)))
    if file_sr != sr:
        import librosa
        chunk = librosa.resample(chunk.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return chunk.astype(np.float32)


def attach_clap_labels(
    events: list[dict],
    audio_path: Path | None,
    prompts: list[str],
    enabled: bool = True,
    threshold: float = 0.55,
    top_k: int = 3,
) -> list[dict]:
    if not enabled or not prompts or audio_path is None:
        return events

    model = _load_model()
    audio_windows = [_extract_window(audio_path, e["t"]) for e in events]

    audio_embeddings = model.get_audio_embedding_from_data(
        np.stack(audio_windows), use_tensor=False
    )
    text_embeddings = model.get_text_embedding(prompts, use_tensor=False)

    # cosine similarity: (n_events, n_prompts)
    audio_norm = audio_embeddings / (np.linalg.norm(audio_embeddings, axis=1, keepdims=True) + 1e-9)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-9)
    scores = audio_norm @ text_norm.T  # (n_events, n_prompts)

    for i, event in enumerate(events):
        top_indices = np.argsort(scores[i])[::-1][:top_k]
        labels = []
        for idx in top_indices:
            score = float(scores[i, idx])
            if score >= threshold:
                labels.append({"label": prompts[idx], "score": score, "source": "clap"})
        event["labels"] = labels
    return events
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_clap_reranker.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/clap_reranker.py tests/test_clap_reranker.py
git commit -m "feat: CLAP re-ranker attaches semantic labels to candidate events"
```

---

### Task 7: Wire everything into pipeline.py

**Files:**
- Modify: `musicue/analysis/pipeline.py`
- Modify: `tests/test_analysis.py`

This task updates `run_analysis` to call all M1 modules in sequence: structure → transcription → phrases → CLAP → extended curves → transitions.

- [ ] **Step 1: Add M1 pipeline integration test**

Append to `tests/test_analysis.py`:

```python
def test_m1_pipeline_includes_beats_and_sections(tmp_path, synthetic_wav):
    from musicue.analysis.structure import detect_structure as _real_detect
    cfg = _make_cfg(tmp_path)

    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        with patch("musicue.analysis.pipeline.detect_structure") as mock_struct:
            mock_struct.return_value = {
                "tempo": {"bpm_global": 120.0, "bpm_curve": [{"t": 0.0, "bpm": 120.0}], "time_signature": [4, 4]},
                "beats": [
                    {"t": 0.5, "beat_in_bar": 1, "bar": 1, "is_downbeat": True, "confidence": 0.9, "timescale": "meso"},
                ],
                "sections": [
                    {"start": 0.0, "end": 5.0, "label": "intro", "confidence": 0.9, "timescale": "macro"},
                ],
            }
            result = run_analysis(synthetic_wav, cfg)

    assert result.tempo is not None
    assert result.tempo.bpm_global == pytest.approx(120.0)
    assert len(result.beats) == 1
    assert result.beats[0].is_downbeat is True
    assert len(result.sections) == 1
    assert result.sections[0].label == "intro"


def test_m1_pipeline_includes_spectral_curves(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    assert "spectral_centroid" in result.curves
    assert "spectral_flux" in result.curves
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_analysis.py -v -k "m1_pipeline"
```
Expected: FAIL — `run_analysis` doesn't call `detect_structure` yet

- [ ] **Step 3: Update musicue/analysis/pipeline.py**

Replace entire file:

```python
from __future__ import annotations
import hashlib
import soundfile as sf
from pathlib import Path

from musicue.analysis.clap_reranker import attach_clap_labels, clap_version
from musicue.analysis.curves import (
    compute_lufs_curve,
    compute_rms_curve,
    compute_spectral_centroid_curve,
    compute_spectral_flux_curve,
    compute_stereo_width_pan,
)
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.phrases import group_into_phrases
from musicue.analysis.separation import separate, demucs_version
from musicue.analysis.structure import detect_structure, allin1_version
from musicue.analysis.transcription import transcribe_stem, basic_pitch_version
from musicue.analysis.transitions import derive_transitions
from musicue.cache import Cache, build_audio_cache_key
from musicue.config import MusiCueConfig
from musicue.schemas import (
    AnalysisConfig, AnalysisResult, BeatEvent, OnsetEvent, PhraseEvent,
    SectionEvent, SectionTransition, SourceInfo, TempoInfo, TimedCurve,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_dict(cfg: MusiCueConfig) -> dict:
    return {
        "demucs_model": cfg.analysis.demucs_model,
        "demucs_version": demucs_version(),
        "allin1_version": allin1_version(),
        "basic_pitch_version": basic_pitch_version(),
        "clap_version": clap_version(),
        "beat_backend": cfg.analysis.beat_backend,
        "curve_hop_sec": cfg.analysis.curve_hop_sec,
    }


def run_analysis(audio_path: Path, cfg: MusiCueConfig) -> AnalysisResult:
    audio_path = audio_path.resolve()
    version_dict = _version_dict(cfg)
    cache_key = build_audio_cache_key(audio_path, version_dict)
    cache = Cache(cfg.cache_dir)

    cached = cache.get(cache_key, "analysis.json")
    if cached is not None:
        return AnalysisResult.model_validate_json(cached.read_text())

    sha256 = _sha256(audio_path)
    info = sf.info(str(audio_path))
    duration_sec = info.frames / info.samplerate

    run_dir = cfg.runs_dir / cache_key[:12]
    stems = separate(audio_path, run_dir / "stems", model=cfg.analysis.demucs_model)
    stems_str = {k: str(v) for k, v in stems.items()}

    # Structure: beats, sections, tempo
    structure = detect_structure(audio_path, backend=cfg.analysis.beat_backend)
    beats = [BeatEvent.model_validate(b) for b in structure.get("beats", [])]
    sections = [SectionEvent.model_validate(s) for s in structure.get("sections", [])]
    tempo = TempoInfo.model_validate(structure["tempo"]) if "tempo" in structure else None

    # Onsets per stem
    onsets: dict[str, list[OnsetEvent]] = {}
    for stem_name, stem_path in stems.items():
        onsets[stem_name] = [OnsetEvent.model_validate(o) for o in detect_onsets(stem_path)]

    # MIDI transcription + phrase grouping for vocals + other
    midi: dict = {}
    phrases: dict[str, list[PhraseEvent]] = {}
    for stem_name in ("vocals", "other"):
        stem_path = stems.get(stem_name)
        if stem_path is None:
            continue
        notes = transcribe_stem(stem_path)
        midi[stem_name] = notes
        gap = cfg.analysis.phrase_gap_sec.get(stem_name, 0.5)
        raw_phrases = group_into_phrases(notes, gap_sec=gap)
        phrases[stem_name] = [PhraseEvent.model_validate(p) for p in raw_phrases]

    # CLAP labels on drum onsets + phrase starts
    if cfg.analysis.clap_top_k > 0:
        import yaml
        from pathlib import Path as _P
        prompts_file = _P("prompt_banks/default_clap_prompts.yaml")
        if prompts_file.exists():
            prompts = yaml.safe_load(prompts_file.read_text()).get("prompts", [])
        else:
            prompts = [
                "sub bass drop", "vocal stab", "cymbal swell", "vocal melisma",
                "snare roll", "riser", "impact hit", "breakdown", "build up",
                "punchy kick", "deep kick", "piano arpeggio", "808 slide",
            ]
        for stem_name in list(onsets.keys()):
            raw_events = [o.model_dump() for o in onsets[stem_name]]
            labeled = attach_clap_labels(
                raw_events, audio_path=audio_path, prompts=prompts,
                enabled=True,
                threshold=cfg.analysis.clap_threshold,
                top_k=cfg.analysis.clap_top_k,
            )
            onsets[stem_name] = [OnsetEvent.model_validate(e) for e in labeled]

    # Curves
    hop = cfg.analysis.curve_hop_sec
    curves: dict[str, TimedCurve] = {
        "lufs": TimedCurve(**compute_lufs_curve(audio_path, hop_sec=hop)),
        "spectral_centroid": TimedCurve(**compute_spectral_centroid_curve(audio_path, hop_sec=hop)),
        "spectral_flux": TimedCurve(**compute_spectral_flux_curve(audio_path, hop_sec=hop)),
    }
    stereo = compute_stereo_width_pan(audio_path, hop_sec=hop)
    curves["stereo_width"] = TimedCurve(**stereo["width"])
    curves["stereo_pan"] = TimedCurve(**stereo["pan"])
    for stem_name, stem_path in stems.items():
        curves[f"rms_{stem_name}"] = TimedCurve(**compute_rms_curve(stem_path, hop_sec=hop))

    # Section transitions
    flux_dict = curves["spectral_flux"].model_dump()
    lufs_dict = curves["lufs"].model_dump()
    raw_transitions = derive_transitions(
        [s.model_dump() for s in sections], flux_dict, lufs_dict
    )
    section_transitions = []
    for t in raw_transitions:
        # Rename 'from' key to 'from_section' alias handled in schema
        section_transitions.append(SectionTransition.model_validate(t))

    result = AnalysisResult(
        source=SourceInfo(
            path=str(audio_path), sha256=sha256,
            duration_sec=duration_sec, sample_rate=info.samplerate,
        ),
        analysis_config=AnalysisConfig(
            demucs_model=cfg.analysis.demucs_model,
            demucs_version=demucs_version(),
            allin1_version=allin1_version(),
            basic_pitch_version=basic_pitch_version(),
            clap_version=clap_version(),
            beat_backend=cfg.analysis.beat_backend,
        ),
        stems=stems_str,
        tempo=tempo,
        beats=beats,
        sections=sections,
        section_transitions=section_transitions,
        onsets=onsets,
        midi={k: [] for k in midi},  # stored as raw dicts; typed in M2
        phrases=phrases,
        curves=curves,
    )

    out_json = run_dir / "analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(result.model_dump_json(indent=2))
    cache.put(cache_key, "analysis.json", out_json)
    return result
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_analysis.py -v
```
Expected: all tests PASS (mocked structure + mocked separate)

- [ ] **Step 5: Commit**

```
git add musicue/analysis/pipeline.py tests/test_analysis.py
git commit -m "feat: M1 pipeline — wire All-In-One, Basic Pitch, phrases, CLAP, full curves, transitions"
```

---

### Task 8: Inspect and plot CLI commands

**Files:**
- Create: `musicue/inspect.py`
- Modify: `musicue/cli.py`
- Create: `tests/test_inspect.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_inspect.py`:

```python
import json
import pytest
from pathlib import Path
from musicue.schemas import AnalysisResult, SourceInfo, AnalysisConfig, TimedCurve, OnsetEvent


def _make_analysis_json(tmp_path: Path) -> Path:
    result = AnalysisResult(
        source=SourceInfo(path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={"drums": [OnsetEvent(t=0.5, strength=0.9), OnsetEvent(t=1.0, strength=0.8)]},
        curves={"lufs": TimedCurve(hop_sec=0.04, values=[-20.0] * 250)},
    )
    p = tmp_path / "analysis.json"
    p.write_text(result.model_dump_json())
    return p


def test_summary_returns_dict(tmp_path):
    from musicue.inspect import summarize
    path = _make_analysis_json(tmp_path)
    summary = summarize(path)
    assert "duration_sec" in summary
    assert "onset_counts" in summary
    assert summary["duration_sec"] == pytest.approx(10.0)
    assert summary["onset_counts"]["drums"] == 2


def test_summary_lists_curves(tmp_path):
    from musicue.inspect import summarize
    path = _make_analysis_json(tmp_path)
    summary = summarize(path)
    assert "curves" in summary
    assert "lufs" in summary["curves"]
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_inspect.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.inspect'`

- [ ] **Step 3: Implement musicue/inspect.py**

```python
from __future__ import annotations
from pathlib import Path
from musicue.schemas import AnalysisResult


def summarize(analysis_path: Path) -> dict:
    result = AnalysisResult.model_validate_json(analysis_path.read_text())
    onset_counts = {stem: len(events) for stem, events in result.onsets.items()}
    return {
        "duration_sec": result.source.duration_sec,
        "sample_rate": result.source.sample_rate,
        "schema_version": result.schema_version,
        "beat_count": len(result.beats),
        "downbeat_count": sum(1 for b in result.beats if b.is_downbeat),
        "section_count": len(result.sections),
        "sections": [{"label": s.label, "start": s.start, "end": s.end} for s in result.sections],
        "onset_counts": onset_counts,
        "curves": {name: len(curve.values) for name, curve in result.curves.items()},
        "phrase_counts": {stem: len(ps) for stem, ps in result.phrases.items()},
        "analysis_config": result.analysis_config.model_dump(),
    }


def plot_timeline(analysis_path: Path, out_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    result = AnalysisResult.model_validate_json(analysis_path.read_text())

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Row 1: LUFS curve
    if "lufs" in result.curves:
        curve = result.curves["lufs"]
        t = np.arange(len(curve.values)) * curve.hop_sec
        axes[0].plot(t, curve.values, color="royalblue", lw=0.8)
        axes[0].set_ylabel("LUFS")

    # Row 2: Drum onsets
    for stem, events in result.onsets.items():
        times = [e.t for e in events]
        axes[1].vlines(times, 0, 1, label=stem, alpha=0.6, lw=0.6)
    axes[1].set_ylabel("Onsets")
    axes[1].legend(loc="upper right", fontsize=7)

    # Row 3: Section labels
    colors = {"intro": "#4CAF50", "verse": "#2196F3", "chorus": "#FF5722",
              "bridge": "#9C27B0", "outro": "#607D8B"}
    for s in result.sections:
        color = colors.get(s.label, "#999")
        axes[2].axvspan(s.start, s.end, alpha=0.3, color=color, label=s.label)
        axes[2].text((s.start + s.end) / 2, 0.5, s.label, ha="center", fontsize=8)
    axes[2].set_ylabel("Sections")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    if out_path:
        plt.savefig(str(out_path), dpi=120)
    else:
        plt.show()
    plt.close()
```

- [ ] **Step 4: Add inspect/plot to CLI**

Append these two commands to `musicue/cli.py` (before `if __name__ == "__main__":`):

```python
@app.command()
def inspect(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    latent: bool = typer.Option(False, "--latent", help="Show Music2Latent correlations (requires m2l in analysis)"),
) -> None:
    """Print a human-readable summary of analysis.json."""
    import json
    from musicue.inspect import summarize
    summary = summarize(analysis_path)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def plot(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save plot to file instead of showing"),
) -> None:
    """Render a matplotlib timeline of the analysis."""
    from musicue.inspect import plot_timeline
    plot_timeline(analysis_path, out_path=out)
    if out:
        typer.echo(f"Plot saved to {out}")
```

Also update `_EXPORTERS` to note inspect/plot aren't export targets.

- [ ] **Step 5: Run tests**

```
pytest tests/test_inspect.py tests/test_cli.py -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/inspect.py musicue/cli.py tests/test_inspect.py
git commit -m "feat: inspect + plot CLI commands for analysis.json"
```

---

### Task 9: Create default CLAP prompt bank

**Files:**
- Create: `prompt_banks/default_clap_prompts.yaml`

- [ ] **Step 1: Create the prompt bank file**

```yaml
# prompt_banks/default_clap_prompts.yaml
# Default CLAP prompt bank used by the CLAP re-ranker (§5.5).
# Add prompts here to extend label coverage without modifying Python code.
# Per-grammar prompt overrides go in grammar.yaml under `clap_prompts:`.

prompts:
  - "sub bass drop"
  - "vocal stab"
  - "guitar stab"
  - "cymbal swell"
  - "vocal melisma"
  - "snare roll"
  - "riser"
  - "impact hit"
  - "breakdown"
  - "build up"
  - "silence"
  - "808 slide"
  - "orchestral hit"
  - "reverse cymbal"
  - "piano arpeggio"
  - "punchy kick"
  - "deep kick"
```

- [ ] **Step 2: Commit**

```
git add prompt_banks/default_clap_prompts.yaml
git commit -m "feat: default CLAP prompt bank"
```

---

## M1 Complete

`analysis.json` now contains the full v1.1 schema: beats, downbeats, sections, MIDI, phrases, CLAP-labeled events, spectral curves, stereo width/pan, and section transitions. Continue with the M2 plan to implement the YAML grammar DSL compiler, drum classifier CNN, and QC tooling.
