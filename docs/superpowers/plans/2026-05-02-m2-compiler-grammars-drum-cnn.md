# M2: Compiler, Grammars, and Drum CNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the M0 hardcoded compiler with the full YAML grammar DSL, ship all four built-in grammars, promote the drum classifier CNN from spectral heuristic to a real trained model, and add the `listen` QC click-track renderer and `diff` comparison command.

**Architecture:** `musicue/compile/` gains three modules: `grammar.py` (YAML loading + validation), `scoring.py` (expression evaluator + rarity + hierarchy weights), `envelopes.py` (ADSR + ramp utilities). `compiler.py` is rewritten to use these. The drum CNN lives in `musicue/analysis/drum_classifier.py`. Training script goes in `scripts/train_drum_classifier.py`. `listen.py` and `diff.py` are standalone modules wired into the CLI.

**Tech Stack:** PyYAML, torch (CNN inference), pydantic v2, soundfile, numpy, scipy

**Prerequisite:** M1 plan complete and all tests passing.

---

## File Structure (additions/replacements)

```
musicue/
├── musicue/
│   ├── listen.py                        ← NEW: QC click-track renderer
│   ├── diff.py                          ← NEW: cuesheet diff
│   ├── analysis/
│   │   └── drum_classifier.py           ← NEW: CNN kick/snare/hat/tom/cymbal
│   └── compile/
│       ├── compiler.py                  ← REWRITE: now uses grammar DSL
│       ├── grammar.py                   ← NEW: YAML DSL loader + validation
│       ├── scoring.py                   ← NEW: expression evaluator + scoring
│       └── envelopes.py                 ← NEW: ADSR + ramp utilities
├── grammars/
│   ├── concert_visuals.yaml             ← NEW: full grammar
│   ├── character_animation.yaml         ← NEW
│   ├── lighting.yaml                    ← NEW
│   └── camera_edit.yaml                 ← NEW
├── scripts/
│   └── train_drum_classifier.py         ← NEW: training script stub
└── tests/
    ├── test_grammar.py                  ← NEW
    ├── test_scoring.py                  ← NEW
    ├── test_drum_classifier.py          ← NEW
    ├── test_listen.py                   ← NEW
    └── test_diff.py                     ← NEW
```

---

### Task 1: Grammar YAML loader and validation

**Files:**
- Create: `musicue/compile/grammar.py`
- Create: `tests/test_grammar.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grammar.py`:

```python
import pytest
import yaml
from pathlib import Path
from musicue.compile.grammar import Grammar, GrammarTrack, load_grammar


def _write_grammar(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(data))
    return p


MINIMAL_GRAMMAR = {
    "name": "test_grammar",
    "hierarchy_weights": {"macro": 1.5, "meso": 1.2, "micro": 0.8},
    "tracks": [
        {
            "name": "kick",
            "type": "impulse",
            "source": "onsets.drums",
            "filter": "drum_class == 'kick'",
            "score": {"base": "strength"},
            "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
        }
    ],
}


def test_load_grammar_from_file(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.name == "test_grammar"
    assert len(grammar.tracks) == 1
    assert grammar.tracks[0].name == "kick"


def test_grammar_track_type(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.tracks[0].type == "impulse"
    assert grammar.tracks[0].source == "onsets.drums"


def test_grammar_hierarchy_weights(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.hierarchy_weights["macro"] == pytest.approx(1.5)
    assert grammar.hierarchy_weights["micro"] == pytest.approx(0.8)


def test_grammar_track_envelope(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    env = grammar.tracks[0].envelope
    assert env["a"] == pytest.approx(0.005)
    assert env["s"] == pytest.approx(0.0)


def test_load_grammar_missing_file():
    with pytest.raises(FileNotFoundError):
        load_grammar(Path("nonexistent.yaml"))


def test_load_grammar_by_name(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p.stem, grammars_dir=tmp_path)
    assert grammar.name == "test_grammar"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_grammar.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.compile.grammar'`

- [ ] **Step 3: Implement musicue/compile/grammar.py**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class GrammarTrack(BaseModel):
    name: str
    type: str
    source: str
    filter: str | None = None
    score: dict[str, Any] = Field(default_factory=lambda: {"base": 1.0})
    envelope: dict[str, float] = Field(default_factory=lambda: {"a": 0.01, "d": 0.1, "s": 0.0, "r": 0.0})
    rarity: dict[str, float] | None = None
    cooldown_sec: float | None = None
    shape_curve_from: str | None = None
    emit: str | None = None


class Grammar(BaseModel):
    name: str
    hierarchy_weights: dict[str, float] = Field(
        default_factory=lambda: {"macro": 1.0, "meso": 1.0, "micro": 1.0}
    )
    tracks: list[GrammarTrack] = Field(default_factory=list)
    clap_prompts: list[str] | None = None


def load_grammar(name_or_path: str | Path, grammars_dir: Path = Path("grammars")) -> Grammar:
    path = Path(name_or_path)
    if not path.suffix:
        path = grammars_dir / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Grammar file not found: {path}")
    data = yaml.safe_load(path.read_text())
    return Grammar.model_validate(data)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_grammar.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/compile/grammar.py tests/test_grammar.py
git commit -m "feat: YAML grammar loader and pydantic validation"
```

---

### Task 2: ADSR envelope and ramp utilities

**Files:**
- Create: `musicue/compile/envelopes.py`
- (No separate test file — tested via compiler tests in Task 5)

- [ ] **Step 1: Create musicue/compile/envelopes.py**

```python
from __future__ import annotations
import math
import numpy as np


RAMP_SHAPES = {
    "linear":      lambda x: x,
    "ease_in":     lambda x: x * x,
    "ease_out":    lambda x: 1 - (1 - x) ** 2,
    "ease_in_out": lambda x: x * x * (3 - 2 * x),
    "s_curve":     lambda x: x * x * x * (x * (x * 6 - 15) + 10),
    "exp_in":      lambda x: (math.exp(x * 3) - 1) / (math.exp(3) - 1),
    "exp_out":     lambda x: 1 - (math.exp((1 - x) * 3) - 1) / (math.exp(3) - 1),
}


def adsr_to_dict(a: float, d: float, s: float, r: float) -> dict:
    return {"a": a, "d": d, "s": s, "r": r}


def render_adsr(a: float, d: float, s: float, r: float, sr: float = 100.0) -> np.ndarray:
    """Render ADSR envelope to a numpy array at `sr` samples/sec."""
    attack_n = max(1, int(a * sr))
    decay_n = max(1, int(d * sr))
    sustain_hold = max(0, int(0.1 * sr))  # brief sustain window
    release_n = max(1, int(r * sr))
    total = attack_n + decay_n + sustain_hold + release_n
    env = np.zeros(total)
    env[:attack_n] = np.linspace(0, 1, attack_n)
    env[attack_n : attack_n + decay_n] = np.linspace(1, s, decay_n)
    env[attack_n + decay_n : attack_n + decay_n + sustain_hold] = s
    env[attack_n + decay_n + sustain_hold :] = np.linspace(s, 0, release_n)
    return env


def render_ramp(shape: str, n: int) -> np.ndarray:
    fn = RAMP_SHAPES.get(shape, RAMP_SHAPES["linear"])
    return np.array([fn(x) for x in np.linspace(0.0, 1.0, n)])
```

- [ ] **Step 2: Commit**

```
git add musicue/compile/envelopes.py
git commit -m "feat: ADSR envelope + ramp rendering utilities"
```

---

### Task 3: Event scoring engine

**Files:**
- Create: `musicue/compile/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scoring.py`:

```python
import math
import pytest
from musicue.compile.scoring import (
    evaluate_filter,
    compute_score,
    RarityTracker,
)


def _event(drum_class=None, strength=0.8, labels=None, timescale="micro", is_downbeat=False):
    return {
        "t": 1.0,
        "strength": strength,
        "drum_class": drum_class,
        "timescale": timescale,
        "is_downbeat": is_downbeat,
        "labels": labels or [],
    }


# --- filter evaluation ---

def test_filter_drum_class_eq_true():
    assert evaluate_filter("drum_class == 'kick'", _event(drum_class="kick")) is True


def test_filter_drum_class_eq_false():
    assert evaluate_filter("drum_class == 'kick'", _event(drum_class="snare")) is False


def test_filter_is_downbeat_true():
    assert evaluate_filter("is_downbeat == true", _event(is_downbeat=True)) is True


def test_filter_is_downbeat_false():
    assert evaluate_filter("is_downbeat == true", _event(is_downbeat=False)) is False


def test_filter_none_matches_all():
    assert evaluate_filter(None, _event()) is True


def test_filter_any_label_match():
    event = _event(labels=[{"label": "sub bass drop", "score": 0.75, "source": "clap"}])
    assert evaluate_filter("any_label('sub bass drop', min_score=0.6)", event) is True


def test_filter_any_label_below_threshold():
    event = _event(labels=[{"label": "sub bass drop", "score": 0.4, "source": "clap"}])
    assert evaluate_filter("any_label('sub bass drop', min_score=0.6)", event) is False


# --- scoring ---

def test_compute_score_base_literal():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(strength=0.8), timescale_weight=1.0, rarity_bonus=1.0)
    assert s == pytest.approx(1.0)


def test_compute_score_base_field():
    score_cfg = {"base": "strength"}
    s = compute_score(score_cfg, _event(strength=0.75), timescale_weight=1.0, rarity_bonus=1.0)
    assert s == pytest.approx(0.75)


def test_compute_score_with_timescale_weight():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(), timescale_weight=1.5, rarity_bonus=1.0)
    assert s == pytest.approx(1.5)


def test_compute_score_with_rarity_bonus():
    score_cfg = {"base": 1.0}
    s = compute_score(score_cfg, _event(), timescale_weight=1.0, rarity_bonus=0.5)
    assert s == pytest.approx(0.5)


# --- rarity tracker ---

def test_rarity_bonus_no_recent_events():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    bonus = tracker.bonus(t=5.0)
    assert bonus == pytest.approx(1.0)  # exp(0) = 1


def test_rarity_bonus_one_recent_event():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    tracker.record(t=4.5)
    bonus = tracker.bonus(t=5.0)
    # one event in window → exp(-1/4) ≈ 0.778
    assert bonus == pytest.approx(math.exp(-1 / 4.0), abs=0.01)


def test_rarity_bonus_outside_window():
    tracker = RarityTracker(window_sec=1.0, decay=4.0)
    tracker.record(t=3.0)  # 2s ago — outside 1s window
    bonus = tracker.bonus(t=5.0)
    assert bonus == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_scoring.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.compile.scoring'`

- [ ] **Step 3: Implement musicue/compile/scoring.py**

```python
from __future__ import annotations
import math
import re
from typing import Any


def evaluate_filter(expr: str | None, event: dict) -> bool:
    if expr is None:
        return True

    # any_label('label', min_score=X)
    m = re.match(r"any_label\('([^']+)',\s*min_score=([\d.]+)\)", expr)
    if m:
        target_label, min_score = m.group(1), float(m.group(2))
        return any(
            lbl["label"] == target_label and lbl["score"] >= min_score
            for lbl in event.get("labels", [])
        )

    # near_downbeat(seconds)
    m = re.match(r"near_downbeat\(([\d.]+)\)", expr)
    if m:
        return bool(event.get("near_downbeat", False))

    # field == 'value'
    m = re.match(r"(\w+)\s*==\s*'([^']*)'", expr)
    if m:
        field, value = m.group(1), m.group(2)
        return str(event.get(field, "")) == value

    # field == true/false
    m = re.match(r"(\w+)\s*==\s*(true|false)", expr)
    if m:
        field, value = m.group(1), m.group(2) == "true"
        return bool(event.get(field, False)) == value

    # field > value
    m = re.match(r"(\w+)\s*>\s*([\d.]+)", expr)
    if m:
        field, value = m.group(1), float(m.group(2))
        return float(event.get(field, 0)) > value

    return True  # unknown expr — pass through


def _resolve_base(base: Any, event: dict) -> float:
    if isinstance(base, (int, float)):
        return float(base)
    if isinstance(base, str):
        if base in event:
            return float(event[base])
        m = re.match(r"label_score\('([^']+)'\)", base)
        if m:
            target = m.group(1)
            for lbl in event.get("labels", []):
                if lbl["label"] == target:
                    return float(lbl["score"])
            return 0.0
        if base == "max(energy_curve)":
            curve = event.get("energy_curve", {})
            vals = curve.get("values", [0.0])
            return float(max(vals)) if vals else 0.0
    return 1.0


def compute_score(
    score_cfg: dict,
    event: dict,
    timescale_weight: float = 1.0,
    rarity_bonus: float = 1.0,
) -> float:
    base = _resolve_base(score_cfg.get("base", 1.0), event)
    multiplier = 1.0
    for rule in score_cfg.get("multiplier", []):
        if evaluate_filter(rule.get("when"), event):
            multiplier *= float(rule.get("factor", 1.0))
    return base * multiplier * timescale_weight * rarity_bonus


class RarityTracker:
    def __init__(self, window_sec: float = 1.0, decay: float = 4.0) -> None:
        self.window_sec = window_sec
        self.decay = decay
        self._history: list[float] = []

    def bonus(self, t: float) -> float:
        count = sum(1 for et in self._history if t - et <= self.window_sec)
        return math.exp(-count / self.decay)

    def record(self, t: float) -> None:
        self._history.append(t)
        # prune old entries
        cutoff = t - self.window_sec * 2
        self._history = [et for et in self._history if et >= cutoff]
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_scoring.py -v
```
Expected: all 14 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/compile/scoring.py tests/test_scoring.py
git commit -m "feat: event scoring engine — filter DSL, base/multiplier, rarity tracker"
```

---

### Task 4: Four built-in grammars

**Files:**
- Create: `grammars/concert_visuals.yaml`
- Create: `grammars/character_animation.yaml`
- Create: `grammars/lighting.yaml`
- Create: `grammars/camera_edit.yaml`

- [ ] **Step 1: Create grammars/concert_visuals.yaml**

```yaml
name: concert_visuals

hierarchy_weights:
  macro: 1.5
  meso:  1.2
  micro: 0.8

tracks:
  - name: downbeat_pulse
    type: impulse
    source: beats
    filter: "is_downbeat == true"
    score:
      base: 1.0
      multiplier:
        - {when: "section_label == 'chorus'", factor: 1.3}
        - {when: "section_label == 'drop'",   factor: 1.5}
    envelope: {a: 0.02, d: 0.20, s: 0.0, r: 0.0}

  - name: kick
    type: impulse
    source: onsets.drums
    filter: "drum_class == 'kick'"
    score:
      base: "strength"
      multiplier:
        - {when: "near_downbeat(0.05)", factor: 1.2}
    rarity:
      window_sec: 1.0
      decay: 4.0
    envelope: {a: 0.005, d: 0.12, s: 0.0, r: 0.0}

  - name: snare
    type: impulse
    source: onsets.drums
    filter: "drum_class == 'snare'"
    score:
      base: "strength"
    rarity:
      window_sec: 0.5
      decay: 3.0
    envelope: {a: 0.005, d: 0.15, s: 0.0, r: 0.0}

  - name: drop
    type: impulse
    source: "onsets.*"
    filter: "any_label('sub bass drop', min_score=0.6)"
    score:
      base: "label_score('sub bass drop')"
    envelope: {a: 0.05, d: 0.4, s: 0.6, r: 1.5}
    cooldown_sec: 8.0

  - name: vocal_phrase
    type: envelope
    source: phrases.vocals
    score:
      base: "max(energy_curve)"
    envelope: {a: 0.30, d: 0.20, s: 0.7, r: 0.50}
    shape_curve_from: "energy_curve"

  - name: section_change
    type: step
    source: sections
    emit: section_boundary

  - name: section_ramp
    type: ramp
    source: section_transitions
    filter: "ramp_evidence.spectral_flux_rise > 0.4"

  - name: energy
    type: continuous
    source: curves.lufs
    smoothing: {kind: ema, tau_sec: 0.25}
    normalize: {kind: percentile, low: 5, high: 95}
```

- [ ] **Step 2: Create grammars/character_animation.yaml**

```yaml
name: character_animation

hierarchy_weights:
  macro: 1.2
  meso:  1.5
  micro: 0.6

tracks:
  - name: vocal_phrase
    type: envelope
    source: phrases.vocals
    score:
      base: "max(energy_curve)"
    envelope: {a: 0.40, d: 0.30, s: 0.8, r: 0.80}
    shape_curve_from: "energy_curve"

  - name: melody_phrase
    type: envelope
    source: phrases.other
    score:
      base: "max(energy_curve)"
    envelope: {a: 0.35, d: 0.25, s: 0.6, r: 0.60}

  - name: downbeat
    type: impulse
    source: beats
    filter: "is_downbeat == true"
    score:
      base: 0.7
    envelope: {a: 0.05, d: 0.30, s: 0.0, r: 0.0}

  - name: accent
    type: impulse
    source: onsets.vocals
    score:
      base: "strength"
    rarity:
      window_sec: 2.0
      decay: 6.0
    envelope: {a: 0.02, d: 0.25, s: 0.2, r: 0.40}

  - name: section_change
    type: step
    source: sections
    emit: section_boundary

  - name: energy
    type: continuous
    source: curves.rms_vocals
    smoothing: {kind: ema, tau_sec: 0.5}
    normalize: {kind: percentile, low: 5, high: 95}
```

- [ ] **Step 3: Create grammars/lighting.yaml**

```yaml
name: lighting

hierarchy_weights:
  macro: 1.0
  meso:  1.0
  micro: 1.3

tracks:
  - name: kick
    type: impulse
    source: onsets.drums
    filter: "drum_class == 'kick'"
    score:
      base: "strength"
    envelope: {a: 0.002, d: 0.10, s: 0.0, r: 0.0}

  - name: snare
    type: impulse
    source: onsets.drums
    filter: "drum_class == 'snare'"
    score:
      base: "strength"
    envelope: {a: 0.002, d: 0.12, s: 0.0, r: 0.0}

  - name: hihat
    type: impulse
    source: onsets.drums
    filter: "drum_class == 'hat'"
    score:
      base: "strength"
    rarity:
      window_sec: 0.2
      decay: 2.0
    envelope: {a: 0.001, d: 0.06, s: 0.0, r: 0.0}

  - name: downbeat
    type: impulse
    source: beats
    filter: "is_downbeat == true"
    score:
      base: 1.2
    envelope: {a: 0.01, d: 0.25, s: 0.0, r: 0.0}

  - name: build
    type: impulse
    source: "onsets.*"
    filter: "any_label('build up', min_score=0.55)"
    score:
      base: "label_score('build up')"
    envelope: {a: 0.1, d: 0.5, s: 0.7, r: 2.0}
    cooldown_sec: 16.0

  - name: section_brightness
    type: step
    source: sections
    emit: section_boundary

  - name: section_ramp
    type: ramp
    source: section_transitions

  - name: intensity
    type: continuous
    source: curves.lufs
    smoothing: {kind: ema, tau_sec: 0.1}
    normalize: {kind: percentile, low: 5, high: 95}
```

- [ ] **Step 4: Create grammars/camera_edit.yaml**

```yaml
name: camera_edit

hierarchy_weights:
  macro: 2.0
  meso:  1.0
  micro: 0.4

tracks:
  - name: cut_point
    type: step
    source: sections
    emit: section_boundary

  - name: bar_start
    type: impulse
    source: beats
    filter: "is_downbeat == true"
    score:
      base: 1.0
      multiplier:
        - {when: "section_label == 'chorus'", factor: 1.4}
    envelope: {a: 0.01, d: 0.5, s: 0.0, r: 0.0}
    cooldown_sec: 2.0

  - name: impact
    type: impulse
    source: "onsets.*"
    filter: "any_label('impact hit', min_score=0.6)"
    score:
      base: "label_score('impact hit')"
    envelope: {a: 0.02, d: 0.6, s: 0.4, r: 2.0}
    cooldown_sec: 4.0

  - name: section_ramp
    type: ramp
    source: section_transitions
    filter: "ramp_evidence.spectral_flux_rise > 0.3"

  - name: energy
    type: continuous
    source: curves.lufs
    smoothing: {kind: ema, tau_sec: 0.5}
    normalize: {kind: percentile, low: 10, high: 90}
```

- [ ] **Step 5: Commit**

```
git add grammars/
git commit -m "feat: four built-in grammars (concert_visuals, character_animation, lighting, camera_edit)"
```

---

### Task 5: Full compiler (replaces M0 hardcoded)

**Files:**
- Rewrite: `musicue/compile/compiler.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Add grammar-driven compiler tests**

Append to `tests/test_compile.py`:

```python
from musicue.compile.grammar import Grammar, GrammarTrack
from musicue.compile.compiler import compile_analysis


def _make_grammar_with_kick() -> Grammar:
    return Grammar(
        name="test",
        hierarchy_weights={"macro": 1.5, "meso": 1.2, "micro": 0.8},
        tracks=[
            GrammarTrack(
                name="kick",
                type="impulse",
                source="onsets.drums",
                filter="drum_class == 'kick'",
                score={"base": "strength"},
                envelope={"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
            )
        ],
    )


def test_grammar_compiler_filters_by_drum_class():
    onsets = [
        OnsetEvent(t=0.5, strength=0.9, drum_class="kick"),
        OnsetEvent(t=1.0, strength=0.8, drum_class="snare"),
        OnsetEvent(t=1.5, strength=0.7, drum_class="kick"),
    ]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=_make_grammar_with_kick())
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    assert len(kick_track.events) == 2  # only kick onsets pass the filter


def test_grammar_compiler_applies_hierarchy_weight():
    # micro weight 0.8 → kick scores lower than base
    onsets = [OnsetEvent(t=0.5, strength=1.0, drum_class="kick")]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=_make_grammar_with_kick())
    kick_track = next(t for t in cs.tracks if t.name == "kick")
    # strength=1.0 * micro_weight=0.8 * rarity_bonus≈1.0 ≈ 0.8
    assert kick_track.events[0]["strength"] == pytest.approx(0.8, abs=0.05)


def test_grammar_compiler_cooldown_suppresses_close_events():
    grammar = Grammar(
        name="test",
        hierarchy_weights={"macro": 1.0, "meso": 1.0, "micro": 1.0},
        tracks=[
            GrammarTrack(
                name="drop",
                type="impulse",
                source="onsets.drums",
                score={"base": 1.0},
                envelope={"a": 0.05, "d": 0.4, "s": 0.6, "r": 1.5},
                cooldown_sec=5.0,
            )
        ],
    )
    onsets = [
        OnsetEvent(t=1.0, strength=0.9),
        OnsetEvent(t=2.0, strength=0.8),  # within 5s cooldown → suppressed
        OnsetEvent(t=10.0, strength=0.9),  # outside cooldown → emitted
    ]
    analysis = _make_analysis(onsets=onsets)
    cs = compile_analysis(analysis, grammar=grammar)
    track = cs.tracks[0]
    assert len(track.events) == 2  # t=1.0 and t=10.0


def test_grammar_compiler_loads_from_file(tmp_path):
    import yaml
    grammar_data = {
        "name": "test_file_grammar",
        "hierarchy_weights": {"macro": 1.0, "meso": 1.0, "micro": 1.0},
        "tracks": [
            {"name": "drums", "type": "impulse", "source": "onsets.drums",
             "score": {"base": "strength"}, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}}
        ],
    }
    grammar_file = tmp_path / "test_file_grammar.yaml"
    grammar_file.write_text(yaml.dump(grammar_data))
    analysis = _make_analysis(onsets=[OnsetEvent(t=0.5, strength=0.9)])
    cs = compile_analysis(analysis, grammar="test_file_grammar", grammars_dir=tmp_path)
    assert len(cs.tracks) >= 1
```

- [ ] **Step 2: Run to verify existing tests still pass**

```
pytest tests/test_compile.py -v
```
Expected: old tests PASS, new tests FAIL (compile_analysis doesn't accept Grammar object yet)

- [ ] **Step 3: Rewrite musicue/compile/compiler.py**

```python
from __future__ import annotations
from pathlib import Path
from musicue.compile.envelopes import render_ramp
from musicue.compile.grammar import Grammar, GrammarTrack, load_grammar
from musicue.compile.scoring import RarityTracker, compute_score, evaluate_filter
from musicue.schemas import AnalysisResult, CueSheet, CueTrack, TimedCurve
import numpy as np


def _resolve_source(source: str, analysis: AnalysisResult) -> list[dict]:
    """Resolve a grammar source string to a list of event dicts."""
    if source == "beats":
        return [b.model_dump() for b in analysis.beats]
    if source == "sections":
        return [s.model_dump() for s in analysis.sections]
    if source == "section_transitions":
        return [t.model_dump(by_alias=True) for t in analysis.section_transitions]
    if source.startswith("onsets."):
        stem = source[len("onsets."):]
        if stem == "*":
            all_events = []
            for events in analysis.onsets.values():
                all_events.extend(e.model_dump() for e in events)
            return all_events
        return [e.model_dump() for e in analysis.onsets.get(stem, [])]
    if source.startswith("phrases."):
        stem = source[len("phrases."):]
        return [p.model_dump() for p in analysis.phrases.get(stem, [])]
    if source.startswith("curves."):
        curve_name = source[len("curves."):]
        curve = analysis.curves.get(curve_name)
        if curve:
            return [{"_curve": True, "hop_sec": curve.hop_sec, "values": curve.values}]
    return []


def _section_label_at(t: float, analysis: AnalysisResult) -> str:
    for s in reversed(analysis.sections):
        if s.start <= t:
            return s.label
    return ""


def _smooth_ema(values: list[float], tau_sec: float, hop_sec: float) -> list[float]:
    if not values:
        return []
    alpha = 1.0 - np.exp(-hop_sec / max(tau_sec, 1e-6))
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _normalize_percentile(values: list[float], low: float, high: float) -> list[float]:
    if not values:
        return []
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi == lo:
        return [0.0] * len(values)
    return [float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) for v in values]


def _compile_continuous_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    ev = source_events[0]
    if not ev.get("_curve"):
        return None
    values = list(ev["values"])
    hop_sec = float(ev["hop_sec"])

    if "smoothing" in (track_cfg.model_extra or {}):
        smoothing = track_cfg.model_extra["smoothing"]
        if smoothing.get("kind") == "ema":
            values = _smooth_ema(values, tau_sec=smoothing.get("tau_sec", 0.25), hop_sec=hop_sec)

    if "normalize" in (track_cfg.model_extra or {}):
        norm = track_cfg.model_extra["normalize"]
        if norm.get("kind") == "percentile":
            values = _normalize_percentile(values, norm.get("low", 5), norm.get("high", 95))

    return CueTrack(name=track_cfg.name, type="continuous", timescale="macro",
                    hop_sec=hop_sec, values=values)


def _compile_impulse_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult, hierarchy_weights: dict
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None

    timescale = "micro"
    rarity = RarityTracker(
        window_sec=track_cfg.rarity["window_sec"] if track_cfg.rarity else 1.0,
        decay=track_cfg.rarity["decay"] if track_cfg.rarity else 4.0,
    ) if track_cfg.rarity else None

    last_emitted_t: float | None = None
    emitted = []

    for ev in sorted(source_events, key=lambda e: float(e.get("t", e.get("t_start", 0)))):
        t = float(ev.get("t", ev.get("t_start", 0)))
        ev["section_label"] = _section_label_at(t, analysis)

        if not evaluate_filter(track_cfg.filter, ev):
            continue

        if track_cfg.cooldown_sec and last_emitted_t is not None:
            if t - last_emitted_t < track_cfg.cooldown_sec:
                continue

        tw = hierarchy_weights.get(ev.get("timescale", timescale), 1.0)
        rb = rarity.bonus(t) if rarity else 1.0
        score = compute_score(track_cfg.score, ev, timescale_weight=tw, rarity_bonus=rb)

        if rarity:
            rarity.record(t)

        emitted.append({
            "t": t,
            "strength": float(score),
            "envelope": track_cfg.envelope,
            "tags": [],
        })
        last_emitted_t = t

    if not emitted:
        return None

    # Infer timescale from first event source
    first_ev = source_events[0]
    ts = first_ev.get("timescale", "micro")
    return CueTrack(name=track_cfg.name, type="impulse", timescale=ts, events=emitted)


def _compile_step_track(track_cfg: GrammarTrack, analysis: AnalysisResult) -> CueTrack | None:
    sections = analysis.sections
    if not sections:
        return None
    events = [
        {"t": float(s.start), "value": i + 1, "label": s.label}
        for i, s in enumerate(sections)
    ]
    return CueTrack(name=track_cfg.name, type="step", timescale="macro", events=events)


def _compile_ramp_track(track_cfg: GrammarTrack, analysis: AnalysisResult) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    events = []
    for ev in source_events:
        if not evaluate_filter(track_cfg.filter, ev):
            continue
        ramp = ev.get("ramp", {})
        events.append({
            "t_start": float(ramp.get("t_start", ev.get("t", 0) - 1.2)),
            "t_end": float(ev.get("t", ramp.get("t_end", 0))),
            "from": 0.0,
            "to": 1.0,
            "shape": ramp.get("shape", "ease_in_out"),
            "label": f"{ev.get('from', '')}→{ev.get('to', '')}",
        })
    if not events:
        return None
    return CueTrack(name=track_cfg.name, type="ramp", timescale="macro", events=events)


def _compile_envelope_track(
    track_cfg: GrammarTrack, analysis: AnalysisResult, hierarchy_weights: dict
) -> CueTrack | None:
    source_events = _resolve_source(track_cfg.source, analysis)
    if not source_events:
        return None
    emitted = []
    for ev in source_events:
        if not evaluate_filter(track_cfg.filter, ev):
            continue
        tw = hierarchy_weights.get(ev.get("timescale", "meso"), 1.0)
        score = compute_score(track_cfg.score, ev, timescale_weight=tw, rarity_bonus=1.0)
        t_start = float(ev.get("t_start", ev.get("t", 0)))
        t_end = float(ev.get("t_end", t_start + 1.0))
        event_dict = {
            "t_start": t_start,
            "t_end": t_end,
            "strength": score,
            "envelope": track_cfg.envelope,
            "tags": [],
        }
        if track_cfg.shape_curve_from and track_cfg.shape_curve_from in ev:
            event_dict["shape_curve"] = ev[track_cfg.shape_curve_from]
        emitted.append(event_dict)
    if not emitted:
        return None
    return CueTrack(name=track_cfg.name, type="envelope", timescale="meso", events=emitted)


def compile_analysis(
    analysis: AnalysisResult,
    grammar: str | Grammar = "concert_visuals",
    grammars_dir: Path = Path("grammars"),
) -> CueSheet:
    if isinstance(grammar, str):
        grammar = load_grammar(grammar, grammars_dir=grammars_dir)

    hw = grammar.hierarchy_weights
    tracks: list[CueTrack] = []

    for track_cfg in grammar.tracks:
        track: CueTrack | None = None
        if track_cfg.type == "impulse":
            track = _compile_impulse_track(track_cfg, analysis, hw)
        elif track_cfg.type == "step":
            track = _compile_step_track(track_cfg, analysis)
        elif track_cfg.type == "ramp":
            track = _compile_ramp_track(track_cfg, analysis)
        elif track_cfg.type == "envelope":
            track = _compile_envelope_track(track_cfg, analysis, hw)
        elif track_cfg.type == "continuous":
            track = _compile_continuous_track(track_cfg, analysis)
        if track is not None:
            tracks.append(track)

    return CueSheet(
        source_sha256=analysis.source.sha256,
        grammar=grammar.name,
        duration_sec=analysis.source.duration_sec,
        tempo_map=analysis.tempo.bpm_curve if analysis.tempo else [],
        tracks=tracks,
    )
```

Note: `GrammarTrack` needs `model_extra` support. Add `model_config = {"extra": "allow"}` to `GrammarTrack` in `grammar.py`:

```python
class GrammarTrack(BaseModel):
    model_config = {"extra": "allow"}
    # ... existing fields unchanged
```

- [ ] **Step 4: Run all compile tests**

```
pytest tests/test_compile.py -v
```
Expected: all tests PASS (including new grammar tests)

- [ ] **Step 5: Commit**

```
git add musicue/compile/compiler.py musicue/compile/grammar.py tests/test_compile.py
git commit -m "feat: full YAML grammar DSL compiler — filter, scoring, rarity, cooldown, all track types"
```

---

### Task 6: Drum classifier CNN

**Files:**
- Create: `musicue/analysis/drum_classifier.py`
- Create: `scripts/train_drum_classifier.py`
- Create: `tests/test_drum_classifier.py`

The CNN is a 4-conv-block network over 64-band log-mel patches. In M2 it runs inference. The training script is a stub with the full architecture and data loading — actual training requires the ENST-Drums + MDB Drums datasets.

- [ ] **Step 1: Write failing tests**

Create `tests/test_drum_classifier.py`:

```python
import pytest
import numpy as np
import torch
from musicue.analysis.drum_classifier import DrumClassifierCNN, classify_onset, DRUM_CLASSES


def test_drum_classes_list():
    assert "kick" in DRUM_CLASSES
    assert "snare" in DRUM_CLASSES
    assert "hat" in DRUM_CLASSES
    assert len(DRUM_CLASSES) == 6  # kick, snare, hat, tom, cymbal, other


def test_model_forward_shape():
    model = DrumClassifierCNN(n_classes=6)
    model.eval()
    batch = torch.zeros(4, 1, 64, 44)  # (batch, channels, mel_bins, time_frames)
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
    onsets = [
        {"t": 0.5, "strength": 0.9, "timescale": "micro", "drum_class": None, "drum_class_conf": None, "labels": []},
        {"t": 1.0, "strength": 0.8, "timescale": "micro", "drum_class": None, "drum_class_conf": None, "labels": []},
    ]
    audio = np.zeros(44100, dtype=np.float32)
    result = classify_onsets_batch(onsets, audio, sr=44100, model=model)
    for ev in result:
        assert ev["drum_class"] in DRUM_CLASSES
        assert ev["drum_class_conf"] is not None
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_drum_classifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.drum_classifier'`

- [ ] **Step 3: Implement musicue/analysis/drum_classifier.py**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

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
    mel = librosa.feature.melspectrogram(y=audio_window, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def _extract_window(audio: np.ndarray, t: float, sr: int, window_ms: int = WINDOW_MS) -> np.ndarray:
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
    # Pad/crop to fixed width (44 frames ≈ 50ms window at 512 hop)
    target_frames = 44
    if mel.shape[1] < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
    else:
        mel = mel[:, :target_frames]
    tensor = torch.tensor(mel[np.newaxis, np.newaxis]).to(device)
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
            return onsets  # no model checkpoint — pass through unchanged
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
```

- [ ] **Step 4: Create scripts/train_drum_classifier.py**

```python
"""
Training script for the MusiCue drum classifier CNN.

Datasets required (download separately):
  - ENST-Drums: https://perso.telecom-paristech.fr/grichard/ENST-drums/
  - MDB Drums: https://github.com/MDanalysis/MDB-Drums
  - Slakh2100: https://zenodo.org/record/4599666

Usage:
  python scripts/train_drum_classifier.py --data-dir D:/drum_data --out models/drum_cnn.pt

The script expects pre-processed onset windows in HDF5 format:
  data-dir/
    onsets.h5  — dataset with keys: 'audio' (N, 2205), 'labels' (N,) int in [0,5]
                 Label map: 0=kick, 1=snare, 2=hat, 3=tom, 4=cymbal, 5=other
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from musicue.analysis.drum_classifier import DrumClassifierCNN, _onset_to_mel, DRUM_CLASSES

SR = 44100
N_CLASSES = len(DRUM_CLASSES)


class DrumOnsetDataset(Dataset):
    def __init__(self, h5_path: Path) -> None:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            self.audio = f["audio"][:]   # (N, 2205)
            self.labels = f["labels"][:].astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        mel = _onset_to_mel(self.audio[idx], sr=SR)
        target_frames = 44
        if mel.shape[1] < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
        else:
            mel = mel[:, :target_frames]
        return torch.tensor(mel[np.newaxis]), int(self.labels[idx])


def train(data_dir: Path, out_path: Path, epochs: int = 30, batch_size: int = 128, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DrumOnsetDataset(data_dir / "onsets.h5")
    n_val = int(len(dataset) * 0.15)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DrumClassifierCNN(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch:3d} | train_loss={train_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/drum_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data_dir, args.out, args.epochs, args.batch_size, args.lr)
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_drum_classifier.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/analysis/drum_classifier.py scripts/train_drum_classifier.py tests/test_drum_classifier.py
git commit -m "feat: drum classifier CNN (kick/snare/hat/tom/cymbal/other) + training script"
```

---

### Task 7: QC click-track renderer (`listen`)

**Files:**
- Create: `musicue/listen.py`
- Modify: `musicue/cli.py`
- Create: `tests/test_listen.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_listen.py`:

```python
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from musicue.schemas import CueSheet, CueTrack


def _make_cuesheet(duration=5.0) -> CueSheet:
    return CueSheet(
        source_sha256="abc",
        grammar="test",
        duration_sec=duration,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
                    {"t": 1.0, "strength": 0.8, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
                ],
            ),
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=1.0,
                values=[-20.0, -18.0, -22.0, -19.0, -21.0],
            ),
        ],
    )


def test_render_click_track_creates_wav(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet()
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out)
    assert out.exists()


def test_render_click_track_duration(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet(duration=5.0)
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out, sr=44100)
    data, sr = sf.read(str(out))
    assert abs(len(data) / sr - 5.0) < 0.1


def test_render_click_track_has_transients(tmp_path):
    from musicue.listen import render_click_track
    cs = _make_cuesheet()
    out = tmp_path / "clicks.wav"
    render_click_track(cs, None, out, sr=44100)
    data, _ = sf.read(str(out))
    if data.ndim > 1:
        data = data[:, 0]
    # There should be spikes near t=0.5 and t=1.0
    for burst_t in (0.5, 1.0):
        idx = int(burst_t * 44100)
        window = data[max(0, idx - 2205): idx + 2205]
        assert np.max(np.abs(window)) > 0.1, f"No click near t={burst_t}s"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_listen.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.listen'`

- [ ] **Step 3: Implement musicue/listen.py**

```python
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

# Click tone frequencies per track type (Hz)
_FREQS = {
    "kick": 80,
    "snare": 200,
    "hat": 800,
    "hihat": 800,
    "downbeat": 440,
    "downbeat_pulse": 440,
    "vocal_phrase": 600,
    "drop": 150,
}
_DEFAULT_FREQ = 330


def _click(strength: float, freq: int, sr: int, decay_ms: float = 15.0) -> np.ndarray:
    n = int(decay_ms * sr / 1000)
    t = np.arange(n) / sr
    env = np.exp(-t / (decay_ms / 1000 / 3))
    wave = env * np.sin(2 * np.pi * freq * t)
    return (wave * strength * 0.8).astype(np.float32)


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
        data, file_sr = sf.read(str(source_audio))
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if file_sr != sr:
            import librosa
            data = librosa.resample(data.T, orig_sr=file_sr, target_sr=sr).T
        end = min(len(data), n_samples)
        mix[:end] += data[:end].astype(np.float32) * 0.4

    for track in cuesheet.tracks:
        if track.type not in ("impulse", "envelope"):
            continue
        pan = _PANS.get(track.name, _DEFAULT_PAN)
        freq = _FREQS.get(track.name, _DEFAULT_FREQ)
        for event in track.events:
            t = float(event.get("t") or event.get("t_start", 0.0))
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
```

- [ ] **Step 4: Add `listen` command to cli.py**

Append to `musicue/cli.py`:

```python
@app.command()
def listen(
    cuesheet_path: Path = typer.Argument(..., help="Path to cuesheet.json"),
    audio: Optional[Path] = typer.Option(None, "--audio", "-a", help="Original audio to mix under clicks"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output WAV path"),
) -> None:
    """Render a QC click-track: stereo-placed clicks on every event, mixed over optional source audio."""
    from musicue.listen import render_click_track
    from musicue.schemas import CueSheet

    cs = CueSheet.model_validate_json(cuesheet_path.read_text())
    out_path = out or cuesheet_path.parent / "clicks.wav"
    render_click_track(cs, audio, out_path)
    typer.echo(f"Click track written to {out_path}")
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_listen.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/listen.py musicue/cli.py tests/test_listen.py
git commit -m "feat: QC click-track renderer with stereo-placed per-track clicks"
```

---

### Task 8: Cuesheet diff command

**Files:**
- Create: `musicue/diff.py`
- Modify: `musicue/cli.py`
- Create: `tests/test_diff.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_diff.py`:

```python
import pytest
from musicue.schemas import CueSheet, CueTrack
from musicue.diff import diff_cuesheets


def _cs(tracks) -> CueSheet:
    return CueSheet(source_sha256="x", grammar="g", duration_sec=10.0, tempo_map=[], tracks=tracks)


def _impulse_track(name, times) -> CueTrack:
    return CueTrack(
        name=name, type="impulse", timescale="micro",
        events=[{"t": t, "strength": 0.8, "envelope": {"a":0.005,"d":0.12,"s":0,"r":0}, "tags":[]} for t in times],
    )


def test_diff_identical_cuesheets():
    cs = _cs([_impulse_track("kick", [0.5, 1.0, 1.5])])
    result = diff_cuesheets(cs, cs)
    assert result["kick"]["added"] == 0
    assert result["kick"]["removed"] == 0
    assert result["kick"]["count_a"] == 3
    assert result["kick"]["count_b"] == 3


def test_diff_added_events():
    cs_a = _cs([_impulse_track("kick", [0.5, 1.0])])
    cs_b = _cs([_impulse_track("kick", [0.5, 1.0, 1.5, 2.0])])
    result = diff_cuesheets(cs_a, cs_b)
    assert result["kick"]["added"] == 2
    assert result["kick"]["removed"] == 0


def test_diff_removed_events():
    cs_a = _cs([_impulse_track("kick", [0.5, 1.0, 1.5])])
    cs_b = _cs([_impulse_track("kick", [0.5])])
    result = diff_cuesheets(cs_a, cs_b)
    assert result["kick"]["removed"] == 2
    assert result["kick"]["added"] == 0


def test_diff_new_track_in_b():
    cs_a = _cs([_impulse_track("kick", [0.5])])
    cs_b = _cs([_impulse_track("kick", [0.5]), _impulse_track("snare", [1.0])])
    result = diff_cuesheets(cs_a, cs_b)
    assert "snare" in result
    assert result["snare"]["added"] == 1
    assert result["snare"]["count_a"] == 0


def test_diff_missing_track_in_b():
    cs_a = _cs([_impulse_track("kick", [0.5]), _impulse_track("snare", [1.0])])
    cs_b = _cs([_impulse_track("kick", [0.5])])
    result = diff_cuesheets(cs_a, cs_b)
    assert "snare" in result
    assert result["snare"]["removed"] == 1
    assert result["snare"]["count_b"] == 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_diff.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.diff'`

- [ ] **Step 3: Implement musicue/diff.py**

```python
from __future__ import annotations
import numpy as np
from musicue.schemas import CueSheet


def _event_times(track) -> list[float]:
    return [float(e.get("t") or e.get("t_start", 0)) for e in track.events]


def _match_events(times_a: list[float], times_b: list[float], tol: float = 0.05) -> dict:
    matched_b = set()
    matched_a = set()
    for i, ta in enumerate(times_a):
        for j, tb in enumerate(times_b):
            if j not in matched_b and abs(ta - tb) <= tol:
                matched_a.add(i)
                matched_b.add(j)
                break
    return {
        "matched": len(matched_a),
        "removed": len(times_a) - len(matched_a),
        "added": len(times_b) - len(matched_b),
    }


def diff_cuesheets(cs_a: CueSheet, cs_b: CueSheet, tol: float = 0.05) -> dict:
    tracks_a = {t.name: t for t in cs_a.tracks}
    tracks_b = {t.name: t for t in cs_b.tracks}
    all_names = set(tracks_a) | set(tracks_b)

    result = {}
    for name in sorted(all_names):
        times_a = _event_times(tracks_a[name]) if name in tracks_a else []
        times_b = _event_times(tracks_b[name]) if name in tracks_b else []
        m = _match_events(times_a, times_b, tol=tol)
        result[name] = {
            "count_a": len(times_a),
            "count_b": len(times_b),
            "matched": m["matched"],
            "added": m["added"],
            "removed": m["removed"],
        }
    return result
```

- [ ] **Step 4: Add `diff` command to cli.py**

Append to `musicue/cli.py`:

```python
@app.command()
def diff(
    cuesheet_a: Path = typer.Argument(..., help="First cuesheet.json"),
    cuesheet_b: Path = typer.Argument(..., help="Second cuesheet.json"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Save JSON diff report"),
) -> None:
    """Compare two cuesheets: per-track event count deltas and timing matches."""
    import json
    from musicue.diff import diff_cuesheets
    from musicue.schemas import CueSheet

    cs_a = CueSheet.model_validate_json(cuesheet_a.read_text())
    cs_b = CueSheet.model_validate_json(cuesheet_b.read_text())
    report = diff_cuesheets(cs_a, cs_b)

    typer.echo(f"{'Track':<20} {'A':>6} {'B':>6} {'Added':>7} {'Removed':>9} {'Matched':>9}")
    typer.echo("-" * 60)
    for name, stats in report.items():
        typer.echo(
            f"{name:<20} {stats['count_a']:>6} {stats['count_b']:>6} "
            f"{stats['added']:>7} {stats['removed']:>9} {stats['matched']:>9}"
        )

    if out:
        out.write_text(json.dumps(report, indent=2))
        typer.echo(f"\nDiff report saved to {out}")
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_diff.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/diff.py musicue/cli.py tests/test_diff.py
git commit -m "feat: cuesheet diff command for A/B grammar comparison"
```

---

### Task 9: Wire drum classifier into analysis pipeline

**Files:**
- Modify: `musicue/analysis/pipeline.py`

- [ ] **Step 1: Update pipeline to run drum classifier on drum onsets**

In `musicue/analysis/pipeline.py`, after the onset detection loop, add:

```python
from musicue.analysis.drum_classifier import classify_onsets_batch, drum_classifier_version

# After onsets dict is built, classify drum onsets
drum_model_path = Path("models/drum_cnn.pt")
if drum_model_path.exists():
    import soundfile as sf as _sf2
    drum_audio, drum_sr = _sf2.read(str(stems["drums"]))
    if drum_audio.ndim > 1:
        drum_audio = drum_audio.mean(axis=1)
    drum_onset_dicts = [o.model_dump() for o in onsets.get("drums", [])]
    classified = classify_onsets_batch(
        drum_onset_dicts, drum_audio.astype(np.float32), sr=drum_sr,
        model_path=drum_model_path,
    )
    onsets["drums"] = [OnsetEvent.model_validate(e) for e in classified]
```

Also update `_version_dict` to include `drum_classifier_version`:

```python
from musicue.analysis.drum_classifier import drum_classifier_version as _dcv

def _version_dict(cfg: MusiCueConfig) -> dict:
    drum_model_path = Path("models/drum_cnn.pt")
    return {
        # ... existing keys ...
        "drum_classifier_version": _dcv(drum_model_path),
    }
```

- [ ] **Step 2: Run full test suite**

```
pytest tests/ -v -m "not integration"
```
Expected: all unit tests PASS

- [ ] **Step 3: Commit**

```
git add musicue/analysis/pipeline.py
git commit -m "feat: wire drum CNN into analysis pipeline (skips gracefully if no model checkpoint)"
```

---

## M2 Complete

The full YAML grammar DSL compiler is live. All four built-in grammars work. The drum classifier CNN architecture is implemented (training requires separate dataset download). `listen` lets you QC click placement by ear. `diff` enables A/B grammar iteration. Continue with M3 to add MIDI, After Effects, TouchDesigner, and OSC exporters.
