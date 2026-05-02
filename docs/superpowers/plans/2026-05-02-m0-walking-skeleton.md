# M0: Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire every layer end-to-end so `musicue render song.wav --target csv` produces a CSV cue file without errors.

**Architecture:** Three strict layers — Analysis (Demucs stem separation + librosa onsets + basic LUFS curve), Compiler (hardcoded concert_visuals grammar), Exporter (CSV + JSON). Each layer reads/writes typed pydantic v2 models serialized to JSON. An sha256 + model-version-keyed disk cache prevents re-running Layer 1 on unchanged input.

**Tech Stack:** Python 3.11, pydantic v2, Typer, Demucs 4.x, librosa ≥ 0.10, pyloudnorm, soundfile, numpy, scipy, uv

---

## File Structure

```
musicue/
├── pyproject.toml
├── scripts/
│   └── setup_env.ps1
├── musicue/
│   ├── __init__.py               ← public analyze/compile/export API
│   ├── schemas.py                ← pydantic models for analysis.json + cuesheet.json
│   ├── config.py                 ← MusiCueConfig + sub-configs with YAML loading
│   ├── cache.py                  ← sha256-keyed disk cache
│   ├── cli.py                    ← Typer CLI: analyze/compile/export/render
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pipeline.py           ← Layer 1 orchestration
│   │   ├── separation.py         ← Demucs subprocess wrapper
│   │   ├── onsets.py             ← librosa onset detection
│   │   └── curves.py             ← LUFS + per-stem RMS curves
│   ├── compile/
│   │   ├── __init__.py
│   │   └── compiler.py           ← hardcoded M0 grammar
│   └── exporters/
│       ├── __init__.py
│       ├── csv.py                ← wide CSV (one column per track)
│       └── json_export.py        ← pretty-printed cuesheet.json
└── tests/
    ├── conftest.py               ← synthetic WAV fixture
    ├── test_schemas.py
    ├── test_cache.py
    ├── test_analysis.py
    ├── test_compile.py
    ├── test_exporters.py
    └── test_cli.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `scripts/setup_env.ps1`
- Create: all `__init__.py` stubs

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "musicue"
version = "0.1.0"
description = "Convert songs into typed event timelines for DCC tools"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torchaudio>=2.2",
    "demucs>=4.0",
    "librosa>=0.10",
    "pyloudnorm>=0.1.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "pydantic>=2.0",
    "typer>=0.12",
    "pyyaml>=6.0",
    "soundfile>=0.12",
]

[project.scripts]
musicue = "musicue.cli:app"

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "pyright>=1.1", "matplotlib>=3.8"]
clap = ["laion-clap"]
midi = ["mido>=1.3"]
osc = ["python-osc>=1.8"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "basic"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
markers = ["integration: marks tests that require real model inference (slow)"]
```

- [ ] **Step 2: Create scripts/setup_env.ps1**

```powershell
# scripts/setup_env.ps1
# One-shot Windows 11 environment setup for MusiCue.
# Run from repo root: .\scripts\setup_env.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Installing uv..." -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

Write-Host "Creating .venv with Python 3.11..." -ForegroundColor Cyan
uv venv .venv --python 3.11

Write-Host "Activating venv..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

Write-Host "Installing PyTorch + torchaudio (CUDA 12.4 wheel)..." -ForegroundColor Cyan
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing MusiCue + dev deps..." -ForegroundColor Cyan
uv pip install -e ".[dev]"

Write-Host "Verifying CUDA..." -ForegroundColor Cyan
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
```

- [ ] **Step 3: Create directory structure and empty __init__.py files**

Run in PowerShell from repo root:

```powershell
New-Item -ItemType Directory -Force musicue/analysis, musicue/compile, musicue/exporters, tests, scripts, grammars, runs
foreach ($f in @("musicue/__init__.py","musicue/analysis/__init__.py","musicue/compile/__init__.py","musicue/exporters/__init__.py","tests/__init__.py")) {
    New-Item -ItemType File -Force $f | Out-Null
}
"" | Set-Content runs/.gitkeep
```

- [ ] **Step 4: Commit**

```
git init
git add pyproject.toml scripts/ musicue/ tests/ grammars/ runs/.gitkeep
git commit -m "chore: project scaffolding, pyproject.toml, Windows setup script"
```

---

### Task 2: Pydantic schemas

**Files:**
- Create: `musicue/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas.py`:

```python
import pytest
from musicue.schemas import (
    AnalysisResult, SourceInfo, AnalysisConfig, TimedCurve,
    CueSheet, CueTrack, ADSREnvelope, BeatEvent, OnsetEvent,
)


def _minimal_analysis_dict():
    return {
        "schema_version": "1.1",
        "source": {"path": "song.wav", "sha256": "abc123", "duration_sec": 10.0, "sample_rate": 44100},
        "analysis_config": {"demucs_model": "htdemucs_ft", "demucs_version": "4.0.1"},
        "stems": {"drums": "stems/drums.wav"},
    }


def test_analysis_result_validates():
    result = AnalysisResult.model_validate(_minimal_analysis_dict())
    assert result.source.duration_sec == 10.0
    assert result.schema_version == "1.1"
    assert result.beats == []
    assert result.onsets == {}


def test_analysis_result_roundtrip():
    result = AnalysisResult.model_validate(_minimal_analysis_dict())
    dumped = result.model_dump(mode="json")
    result2 = AnalysisResult.model_validate(dumped)
    assert result2.source.sha256 == "abc123"


def test_beat_event_timescale_defaults_to_micro():
    b = BeatEvent(t=0.5, beat_in_bar=1, bar=1, is_downbeat=True, confidence=0.9)
    assert b.timescale == "micro"


def test_onset_event_labels_default_empty():
    o = OnsetEvent(t=0.5, strength=0.8)
    assert o.labels == []
    assert o.drum_class is None


def test_cuesheet_roundtrip():
    cs = CueSheet(
        source_sha256="abc",
        grammar="test",
        duration_sec=10.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[{"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []}],
            )
        ],
    )
    dumped = cs.model_dump(mode="json")
    cs2 = CueSheet.model_validate(dumped)
    assert cs2.tracks[0].name == "kick"
    assert cs2.tracks[0].events[0]["t"] == pytest.approx(0.5)


def test_adsr_fields():
    env = ADSREnvelope(a=0.01, d=0.1, s=0.5, r=0.3)
    assert env.a == 0.01
    assert env.s == 0.5


def test_timed_curve():
    c = TimedCurve(hop_sec=0.04, values=[-20.0, -18.0, -22.0])
    assert len(c.values) == 3
    assert c.hop_sec == pytest.approx(0.04)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_schemas.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.schemas'`

- [ ] **Step 3: Implement musicue/schemas.py**

```python
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class TimedCurve(BaseModel):
    hop_sec: float
    values: list[float]


class SourceInfo(BaseModel):
    path: str
    sha256: str
    duration_sec: float
    sample_rate: int


class AnalysisConfig(BaseModel):
    demucs_model: str = "htdemucs_ft"
    demucs_version: str = ""
    allin1_model: str = "harmonix-all"
    allin1_version: str = ""
    clap_model: str = "music_audioset_epoch_15_esc_90.14.pt"
    clap_version: str = ""
    basic_pitch_model: str = "icassp_2022"
    basic_pitch_version: str = ""
    drum_classifier_version: str = ""
    beat_backend: Literal["allin1", "librosa"] = "allin1"


class TempoInfo(BaseModel):
    bpm_global: float
    bpm_curve: list[dict[str, float]] = Field(default_factory=list)
    time_signature: list[int] = Field(default=[4, 4])


class Label(BaseModel):
    label: str
    score: float
    source: str


class BeatEvent(BaseModel):
    t: float
    beat_in_bar: int
    bar: int
    is_downbeat: bool
    confidence: float
    timescale: Literal["micro", "meso", "macro"] = "micro"


class SectionEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: float
    timescale: Literal["micro", "meso", "macro"] = "macro"


class RampEvidence(BaseModel):
    spectral_flux_rise: float
    lufs_rise_db: float


class SectionTransition(BaseModel):
    t: float
    from_section: str = Field(alias="from")
    to: str
    ramp: dict[str, Any]
    ramp_evidence: RampEvidence

    model_config = {"populate_by_name": True}


class OnsetEvent(BaseModel):
    t: float
    strength: float
    timescale: Literal["micro", "meso", "macro"] = "micro"
    drum_class: str | None = None
    drum_class_conf: float | None = None
    labels: list[Label] = Field(default_factory=list)


class MidiNote(BaseModel):
    t: float
    duration: float
    pitch: int
    velocity: int


class PhraseEvent(BaseModel):
    t_start: float
    t_end: float
    timescale: Literal["micro", "meso", "macro"] = "meso"
    note_count: int
    pitch_peak: int
    pitch_low: int
    pitch_contour: list[int]
    energy_curve: TimedCurve
    labels: list[Label] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    schema_version: str = "1.1"
    source: SourceInfo
    analysis_config: AnalysisConfig
    stems: dict[str, str]
    tempo: TempoInfo | None = None
    beats: list[BeatEvent] = Field(default_factory=list)
    sections: list[SectionEvent] = Field(default_factory=list)
    section_transitions: list[SectionTransition] = Field(default_factory=list)
    onsets: dict[str, list[OnsetEvent]] = Field(default_factory=dict)
    midi: dict[str, list[MidiNote]] = Field(default_factory=dict)
    phrases: dict[str, list[PhraseEvent]] = Field(default_factory=dict)
    curves: dict[str, TimedCurve] = Field(default_factory=dict)


class ADSREnvelope(BaseModel):
    a: float
    d: float
    s: float
    r: float


class CueTrack(BaseModel):
    name: str
    type: Literal["impulse", "envelope", "step", "ramp", "continuous"]
    timescale: Literal["micro", "meso", "macro"]
    events: list[dict[str, Any]] = Field(default_factory=list)
    hop_sec: float | None = None
    values: list[float] | None = None


class CueSheet(BaseModel):
    schema_version: str = "1.1"
    source_sha256: str
    grammar: str
    duration_sec: float
    tempo_map: list[dict[str, float]] = Field(default_factory=list)
    tracks: list[CueTrack] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify pass**

```
pytest tests/test_schemas.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic v2 schemas for analysis.json and cuesheet.json"
```

---

### Task 3: Config models

**Files:**
- Create: `musicue/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:

```python
from pathlib import Path
import yaml
import pytest
from musicue.config import MusiCueConfig


def test_default_analysis_config():
    cfg = MusiCueConfig()
    assert cfg.analysis.demucs_model == "htdemucs_ft"
    assert cfg.analysis.beat_backend == "allin1"
    assert cfg.analysis.curve_hop_sec == pytest.approx(0.04)


def test_default_compile_config():
    cfg = MusiCueConfig()
    assert cfg.compile.grammar == "concert_visuals"


def test_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "analysis": {"demucs_model": "htdemucs"},
        "compile": {"grammar": "lighting"},
    }))
    cfg = MusiCueConfig.from_yaml(config_file)
    assert cfg.analysis.demucs_model == "htdemucs"
    assert cfg.compile.grammar == "lighting"


def test_cache_dir_default():
    cfg = MusiCueConfig()
    assert cfg.cache_dir == Path.home() / ".musicue" / "cache"


def test_from_yaml_missing_keys_use_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"analysis": {"demucs_model": "htdemucs"}}))
    cfg = MusiCueConfig.from_yaml(config_file)
    assert cfg.compile.grammar == "concert_visuals"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.config'`

- [ ] **Step 3: Implement musicue/config.py**

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class AnalysisRunConfig(BaseModel):
    demucs_model: str = "htdemucs_ft"
    beat_backend: str = "allin1"
    phrase_gap_sec: dict[str, float] = Field(default_factory=lambda: {"vocals": 0.6, "other": 0.4})
    clap_top_k: int = 3
    clap_threshold: float = 0.55
    music2latent: bool = False
    curve_hop_sec: float = 0.04


class CompileRunConfig(BaseModel):
    grammar: str = "concert_visuals"
    grammars_dir: Path = Path("grammars")


class MusiCueConfig(BaseModel):
    analysis: AnalysisRunConfig = Field(default_factory=AnalysisRunConfig)
    compile: CompileRunConfig = Field(default_factory=CompileRunConfig)
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".musicue" / "cache")
    runs_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: Path) -> MusiCueConfig:
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(data)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_config.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/config.py tests/test_config.py
git commit -m "feat: config models with YAML loading"
```

---

### Task 4: Cache module

**Files:**
- Create: `musicue/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache.py`:

```python
import json
from pathlib import Path
import pytest
from musicue.cache import Cache, build_audio_cache_key


def test_cache_miss_returns_none(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get("abc123", "analysis.json") is None


def test_cache_put_and_get(tmp_path):
    cache = Cache(tmp_path)
    src = tmp_path / "source.json"
    src.write_text('{"hello": "world"}')
    cache.put("abc123", "analysis.json", src)
    result = cache.get("abc123", "analysis.json")
    assert result is not None
    assert json.loads(result.read_text()) == {"hello": "world"}


def test_cache_key_changes_with_audio_content(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(b"\x00\x01\x02")
    wav_b.write_bytes(b"\x00\x01\x03")
    config = {"demucs_model": "htdemucs_ft", "demucs_version": "4.0.1"}
    assert build_audio_cache_key(wav_a, config) != build_audio_cache_key(wav_b, config)


def test_cache_key_changes_with_model_version(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00\x01\x02")
    config_a = {"demucs_version": "4.0.1"}
    config_b = {"demucs_version": "4.0.2"}
    assert build_audio_cache_key(wav, config_a) != build_audio_cache_key(wav, config_b)


def test_cache_key_is_stable(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00\x01\x02")
    config = {"demucs_version": "4.0.1"}
    assert build_audio_cache_key(wav, config) == build_audio_cache_key(wav, config)


def test_cache_put_bytes_and_get(tmp_path):
    cache = Cache(tmp_path)
    cache.put_bytes("key1", "stems/drums.wav", b"\x01\x02\x03")
    result = cache.get("key1", "stems/drums.wav")
    assert result is not None
    assert result.read_bytes() == b"\x01\x02\x03"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_cache.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.cache'`

- [ ] **Step 3: Implement musicue/cache.py**

```python
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path


def build_audio_cache_key(audio_path: Path, config_dict: dict) -> str:
    h = hashlib.sha256()
    h.update(audio_path.read_bytes())
    h.update(json.dumps(config_dict, sort_keys=True).encode())
    return h.hexdigest()


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _entry_path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / key / suffix

    def get(self, key: str, suffix: str) -> Path | None:
        p = self._entry_path(key, suffix)
        return p if p.exists() else None

    def put(self, key: str, suffix: str, src: Path) -> Path:
        dest = self._entry_path(key, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def put_bytes(self, key: str, suffix: str, data: bytes) -> Path:
        dest = self._entry_path(key, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_cache.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/cache.py tests/test_cache.py
git commit -m "feat: sha256 + model-version-keyed disk cache"
```

---

### Task 5: Demucs separation wrapper

**Files:**
- Create: `musicue/analysis/separation.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis.py`:

```python
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from musicue.analysis.separation import separate, demucs_version


def test_demucs_version_returns_string():
    v = demucs_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_separate_raises_on_demucs_failure(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        with pytest.raises(RuntimeError, match="Demucs failed"):
            separate(wav, tmp_path / "out")


def test_separate_returns_four_stem_paths(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals", "other"):
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        stems = separate(wav, out_dir, model=model)

    assert set(stems.keys()) == {"drums", "bass", "vocals", "other"}
    for p in stems.values():
        assert p.exists()


def test_separate_raises_when_stem_missing(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals"):  # "other" missing
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with pytest.raises(FileNotFoundError, match="other"):
            separate(wav, out_dir, model=model)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_analysis.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.separation'`

- [ ] **Step 3: Implement musicue/analysis/separation.py**

```python
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

try:
    from importlib.metadata import version as _pkg_version

    def demucs_version() -> str:
        try:
            return _pkg_version("demucs")
        except Exception:
            return "unknown"
except ImportError:
    def demucs_version() -> str:
        return "unknown"


def separate(
    audio_path: Path,
    out_dir: Path,
    model: str = "htdemucs_ft",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(out_dir),
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    stem_dir = out_dir / model / audio_path.stem
    stems: dict[str, Path] = {}
    for stem_name in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{stem_name}.wav"
        if not p.exists():
            raise FileNotFoundError(f"Demucs did not produce expected stem: {p}")
        stems[stem_name] = p
    return stems
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_analysis.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/separation.py tests/test_analysis.py
git commit -m "feat: Demucs separation wrapper with subprocess mock tests"
```

---

### Task 6: Onset detection + LUFS/RMS curves

**Files:**
- Create: `musicue/analysis/onsets.py`
- Create: `musicue/analysis/curves.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Create synthetic WAV fixture**

Create `tests/conftest.py`:

```python
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory) -> Path:
    """10-second 44100 Hz mono WAV: 440 Hz sine + 4 transient bursts at 0.5, 2.5, 5.0, 7.5s."""
    sr = 44100
    duration = 10.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(42)
    for burst_t in (0.5, 2.5, 5.0, 7.5):
        onset = int(burst_t * sr)
        length = min(2205, n - onset)
        decay = np.exp(-np.arange(length) / 220.0)
        signal[onset : onset + length] += 0.6 * decay * rng.standard_normal(length)
    signal = np.clip(signal, -1.0, 1.0).astype(np.float32)
    p = tmp_path_factory.mktemp("fixtures") / "test_song.wav"
    sf.write(str(p), signal, sr)
    return p
```

- [ ] **Step 2: Add onset + curve tests**

Append to `tests/test_analysis.py`:

```python
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.curves import compute_lufs_curve, compute_rms_curve


def test_detect_onsets_returns_list(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    assert isinstance(onsets, list)
    assert len(onsets) >= 2


def test_onset_event_fields(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    assert len(onsets) > 0
    o = onsets[0]
    assert "t" in o and "strength" in o and "timescale" in o and "labels" in o
    assert 0.0 <= o["strength"] <= 1.0
    assert o["timescale"] == "micro"
    assert o["labels"] == []
    assert o["drum_class"] is None


def test_onset_times_ascending(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    times = [o["t"] for o in onsets]
    assert times == sorted(times)


def test_onsets_detect_all_four_bursts(synthetic_wav):
    onsets = detect_onsets(synthetic_wav)
    times = [o["t"] for o in onsets]
    for burst_t in (0.5, 2.5, 5.0, 7.5):
        assert any(abs(t - burst_t) < 0.25 for t in times), f"No onset near {burst_t}s"


def test_lufs_curve_structure(synthetic_wav):
    curve = compute_lufs_curve(synthetic_wav, hop_sec=0.04)
    assert "hop_sec" in curve and "values" in curve
    assert curve["hop_sec"] == pytest.approx(0.04, abs=0.01)
    assert len(curve["values"]) > 0
    assert all(isinstance(v, float) for v in curve["values"])


def test_lufs_curve_range(synthetic_wav):
    curve = compute_lufs_curve(synthetic_wav, hop_sec=0.04)
    assert all(-70.0 <= v <= 0.0 for v in curve["values"])


def test_rms_curve_non_negative(synthetic_wav):
    curve = compute_rms_curve(synthetic_wav, hop_sec=0.04)
    assert len(curve["values"]) > 0
    assert all(v >= 0.0 for v in curve["values"])
```

- [ ] **Step 3: Run to verify failures**

```
pytest tests/test_analysis.py -v -k "onset or lufs or rms"
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.onsets'`

- [ ] **Step 4: Implement musicue/analysis/onsets.py**

```python
from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path


def detect_onsets(audio_path: Path, sr: int = 22050) -> list[dict]:
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.max() == 0:
        return []
    frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.07,
        wait=int(0.03 * sr / 512),
    )
    times = librosa.frames_to_time(frames, sr=sr)
    peak = float(onset_env.max())
    return [
        {
            "t": float(t),
            "strength": float(np.clip(onset_env[f] / peak, 0.0, 1.0)),
            "timescale": "micro",
            "drum_class": None,
            "drum_class_conf": None,
            "labels": [],
        }
        for t, f in zip(times, frames)
    ]
```

- [ ] **Step 5: Implement musicue/analysis/curves.py**

```python
from __future__ import annotations
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf
from pathlib import Path


def compute_lufs_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    data, rate = sf.read(str(audio_path))
    if data.ndim == 1:
        data = data[:, np.newaxis]
    meter = pyln.Meter(rate)
    hop = max(int(hop_sec * rate), 100)
    values: list[float] = []
    for i in range(0, len(data), hop):
        chunk = data[i : i + hop]
        if len(chunk) < 100:
            values.append(values[-1] if values else -70.0)
            continue
        try:
            loudness = meter.integrated_loudness(chunk)
            values.append(float(np.clip(loudness, -70.0, 0.0)))
        except Exception:
            values.append(-70.0)
    return {"hop_sec": hop / rate, "values": values}


def compute_rms_curve(audio_path: Path, hop_sec: float = 0.04) -> dict:
    y, rate = librosa.load(str(audio_path), sr=None, mono=True)
    hop = max(1, int(hop_sec * rate))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    return {"hop_sec": hop / rate, "values": [float(v) for v in rms]}
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_analysis.py -v
```
Expected: all tests PASS

- [ ] **Step 7: Commit**

```
git add musicue/analysis/onsets.py musicue/analysis/curves.py tests/conftest.py tests/test_analysis.py
git commit -m "feat: onset detection, LUFS and RMS curve computation"
```

---

### Task 7: Analysis pipeline (M0)

**Files:**
- Create: `musicue/analysis/pipeline.py`
- Modify: `tests/test_analysis.py`

The pipeline: compute sha256 → check cache → Demucs → onsets per stem → LUFS + per-stem RMS → assemble `AnalysisResult` → write + cache.

- [ ] **Step 1: Add pipeline tests**

Append to `tests/test_analysis.py`:

```python
import shutil as _shutil
from musicue.analysis.pipeline import run_analysis
from musicue.config import MusiCueConfig
from musicue.schemas import AnalysisResult


def _make_cfg(tmp_path):
    cfg = MusiCueConfig()
    cfg.cache_dir = tmp_path / "cache"
    cfg.runs_dir = tmp_path / "runs"
    return cfg


def _fake_separate(audio_path, out_dir, model):
    stem_dir = out_dir / model / audio_path.stem
    stem_dir.mkdir(parents=True)
    stems = {}
    for s in ("drums", "bass", "vocals", "other"):
        p = stem_dir / f"{s}.wav"
        _shutil.copy(audio_path, p)
        stems[s] = p
    return stems


def test_pipeline_returns_analysis_result(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    assert isinstance(result, AnalysisResult)
    assert result.source.duration_sec > 0
    assert set(result.stems.keys()) == {"drums", "bass", "vocals", "other"}
    assert "drums" in result.onsets
    assert "lufs" in result.curves
    assert "rms_drums" in result.curves


def test_pipeline_source_sha256_matches_file(tmp_path, synthetic_wav):
    import hashlib
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate):
        result = run_analysis(synthetic_wav, cfg)
    expected = hashlib.sha256(synthetic_wav.read_bytes()).hexdigest()
    assert result.source.sha256 == expected


def test_pipeline_caches_and_skips_demucs_on_rerun(tmp_path, synthetic_wav):
    cfg = _make_cfg(tmp_path)
    with patch("musicue.analysis.pipeline.separate", side_effect=_fake_separate) as mock_sep:
        run_analysis(synthetic_wav, cfg)
        run_analysis(synthetic_wav, cfg)
    assert mock_sep.call_count == 1  # demucs ran only once
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_analysis.py -v -k "pipeline"
```
Expected: `ModuleNotFoundError: No module named 'musicue.analysis.pipeline'`

- [ ] **Step 3: Implement musicue/analysis/pipeline.py**

```python
from __future__ import annotations
import hashlib
import soundfile as sf
from pathlib import Path

from musicue.analysis.curves import compute_lufs_curve, compute_rms_curve
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.separation import separate, demucs_version
from musicue.cache import Cache, build_audio_cache_key
from musicue.config import MusiCueConfig
from musicue.schemas import (
    AnalysisConfig, AnalysisResult, OnsetEvent, SourceInfo, TimedCurve,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_dict(cfg: MusiCueConfig) -> dict:
    return {
        "demucs_model": cfg.analysis.demucs_model,
        "demucs_version": demucs_version(),
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

    onsets: dict[str, list[OnsetEvent]] = {}
    for stem_name, stem_path in stems.items():
        onsets[stem_name] = [OnsetEvent.model_validate(o) for o in detect_onsets(stem_path)]

    curves: dict[str, TimedCurve] = {
        "lufs": TimedCurve(**compute_lufs_curve(audio_path, hop_sec=cfg.analysis.curve_hop_sec))
    }
    for stem_name, stem_path in stems.items():
        curves[f"rms_{stem_name}"] = TimedCurve(
            **compute_rms_curve(stem_path, hop_sec=cfg.analysis.curve_hop_sec)
        )

    result = AnalysisResult(
        source=SourceInfo(
            path=str(audio_path),
            sha256=sha256,
            duration_sec=duration_sec,
            sample_rate=info.samplerate,
        ),
        analysis_config=AnalysisConfig(
            demucs_model=cfg.analysis.demucs_model,
            demucs_version=demucs_version(),
            beat_backend=cfg.analysis.beat_backend,
        ),
        stems=stems_str,
        onsets=onsets,
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
Expected: all tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/analysis/pipeline.py tests/test_analysis.py
git commit -m "feat: M0 analysis pipeline with caching"
```

---

### Task 8: Minimal compiler

**Files:**
- Create: `musicue/compile/compiler.py`
- Create: `tests/test_compile.py`

The M0 compiler is hardcoded — no YAML DSL (that's M2). It emits drum onsets as an impulse track and LUFS as a continuous energy track.

- [ ] **Step 1: Write failing tests**

Create `tests/test_compile.py`:

```python
import pytest
from musicue.schemas import (
    AnalysisConfig, AnalysisResult, CueSheet, OnsetEvent, SourceInfo, TimedCurve,
)
from musicue.compile.compiler import compile_analysis


def _make_analysis(onsets=None, lufs_values=None) -> AnalysisResult:
    return AnalysisResult(
        source=SourceInfo(path="song.wav", sha256="abc", duration_sec=10.0, sample_rate=44100),
        analysis_config=AnalysisConfig(demucs_version="4.0.1"),
        stems={"drums": "stems/drums.wav"},
        onsets={"drums": onsets or []},
        curves={"lufs": TimedCurve(hop_sec=0.04, values=lufs_values or ([-20.0] * 250))},
    )


def test_compile_returns_cuesheet():
    cs = compile_analysis(_make_analysis(), grammar="concert_visuals")
    assert isinstance(cs, CueSheet)
    assert cs.grammar == "concert_visuals"
    assert cs.duration_sec == pytest.approx(10.0)


def test_compile_drums_impulse_track():
    onsets = [OnsetEvent(t=0.5, strength=0.9), OnsetEvent(t=1.0, strength=0.8)]
    cs = compile_analysis(_make_analysis(onsets=onsets))
    drum_tracks = [t for t in cs.tracks if t.name == "drums"]
    assert len(drum_tracks) == 1
    assert drum_tracks[0].type == "impulse"
    assert len(drum_tracks[0].events) == 2
    assert drum_tracks[0].events[0]["t"] == pytest.approx(0.5)
    assert drum_tracks[0].events[0]["strength"] == pytest.approx(0.9)


def test_compile_drum_event_has_envelope():
    onsets = [OnsetEvent(t=0.5, strength=0.9)]
    cs = compile_analysis(_make_analysis(onsets=onsets))
    env = cs.tracks[0].events[0]["envelope"]
    assert "a" in env and "d" in env and "s" in env and "r" in env


def test_compile_energy_continuous_track():
    cs = compile_analysis(_make_analysis(lufs_values=[-20.0] * 100))
    energy_tracks = [t for t in cs.tracks if t.name == "energy"]
    assert len(energy_tracks) == 1
    assert energy_tracks[0].type == "continuous"
    assert energy_tracks[0].hop_sec is not None
    assert len(energy_tracks[0].values) == 100


def test_compile_carries_source_sha256():
    cs = compile_analysis(_make_analysis())
    assert cs.source_sha256 == "abc"


def test_compile_empty_onsets_produces_no_drums_track():
    cs = compile_analysis(_make_analysis(onsets=[]))
    assert all(t.name != "drums" for t in cs.tracks)
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_compile.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.compile.compiler'`

- [ ] **Step 3: Implement musicue/compile/compiler.py**

```python
from __future__ import annotations
from musicue.schemas import AnalysisResult, CueSheet, CueTrack


def compile_analysis(analysis: AnalysisResult, grammar: str = "concert_visuals") -> CueSheet:
    tracks: list[CueTrack] = []

    drum_onsets = analysis.onsets.get("drums", [])
    if drum_onsets:
        events = [
            {
                "t": o.t,
                "strength": o.strength,
                "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
                "tags": [o.drum_class] if o.drum_class else [],
            }
            for o in drum_onsets
        ]
        tracks.append(CueTrack(name="drums", type="impulse", timescale="micro", events=events))

    if "lufs" in analysis.curves:
        lufs = analysis.curves["lufs"]
        tracks.append(CueTrack(
            name="energy",
            type="continuous",
            timescale="macro",
            hop_sec=lufs.hop_sec,
            values=list(lufs.values),
        ))

    return CueSheet(
        source_sha256=analysis.source.sha256,
        grammar=grammar,
        duration_sec=analysis.source.duration_sec,
        tempo_map=analysis.tempo.bpm_curve if analysis.tempo else [],
        tracks=tracks,
    )
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_compile.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/compile/compiler.py tests/test_compile.py
git commit -m "feat: M0 minimal compiler (hardcoded drum + energy grammar)"
```

---

### Task 9: CSV and JSON exporters

**Files:**
- Create: `musicue/exporters/csv.py`
- Create: `musicue/exporters/json_export.py`
- Create: `tests/test_exporters.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_exporters.py`:

```python
import csv
import json
import pytest
from pathlib import Path
from musicue.schemas import CueSheet, CueTrack


def _make_cuesheet() -> CueSheet:
    return CueSheet(
        source_sha256="abc",
        grammar="concert_visuals",
        duration_sec=5.0,
        tempo_map=[],
        tracks=[
            CueTrack(
                name="drums",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
                    {"t": 2.0, "strength": 0.7, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": []},
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


def test_json_export_creates_valid_file(tmp_path):
    from musicue.exporters.json_export import export
    out = tmp_path / "cuesheet.json"
    export(_make_cuesheet(), out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["grammar"] == "concert_visuals"
    assert len(data["tracks"]) == 2


def test_json_export_roundtrip(tmp_path):
    from musicue.exporters.json_export import export
    out = tmp_path / "cuesheet.json"
    export(_make_cuesheet(), out)
    cs2 = CueSheet.model_validate_json(out.read_text())
    assert cs2.duration_sec == pytest.approx(5.0)
    assert cs2.tracks[0].name == "drums"


def test_csv_export_creates_file(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    assert out.exists()


def test_csv_has_time_sec_column(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "time_sec" in headers


def test_csv_has_track_columns(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "energy" in headers
    assert "drums" in headers


def test_csv_row_count_matches_continuous_track(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # energy has 5 values at 1.0s hop → 5 rows
    assert len(rows) == 5


def test_csv_impulse_column_fires_at_event_time(tmp_path):
    from musicue.exporters.csv import export
    out = tmp_path / "cuesheet.csv"
    export(_make_cuesheet(), out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # event at t=0.5s → should be non-zero in the row nearest 0.5s
    drums_col = [float(r["drums"]) for r in rows]
    assert max(drums_col) > 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_exporters.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.json_export'`

- [ ] **Step 3: Implement musicue/exporters/json_export.py**

```python
from __future__ import annotations
from pathlib import Path
from musicue.schemas import CueSheet


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cuesheet.model_dump_json(indent=2))
```

- [ ] **Step 4: Implement musicue/exporters/csv.py**

```python
from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from musicue.schemas import CueSheet, CueTrack


def _time_grid(cuesheet: CueSheet, default_hop: float = 0.04) -> np.ndarray:
    hop = default_hop
    for track in cuesheet.tracks:
        if track.type == "continuous" and track.hop_sec:
            hop = min(hop, track.hop_sec)
    n = max(1, int(np.ceil(cuesheet.duration_sec / hop)))
    return np.linspace(0.0, cuesheet.duration_sec, n, endpoint=False)


def _continuous_col(track: CueTrack, times: np.ndarray) -> list[float]:
    if not track.values or not track.hop_sec:
        return [0.0] * len(times)
    src_t = np.arange(len(track.values)) * track.hop_sec
    return list(np.interp(times, src_t, track.values))


def _impulse_col(track: CueTrack, times: np.ndarray) -> list[float]:
    col = np.zeros(len(times))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    for event in track.events:
        t = event.get("t") or event.get("t_start", 0.0)
        idx = int(round(float(t) / hop))
        if 0 <= idx < len(col):
            col[idx] = float(event.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    times = _time_grid(cuesheet)
    columns: dict[str, list[float]] = {"time_sec": list(times)}
    for track in cuesheet.tracks:
        if track.type == "continuous":
            columns[track.name] = _continuous_col(track, times)
        else:
            columns[track.name] = _impulse_col(track, times)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for i in range(len(times)):
            writer.writerow({k: v[i] for k, v in columns.items()})
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_exporters.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/exporters/json_export.py musicue/exporters/csv.py tests/test_exporters.py
git commit -m "feat: CSV and JSON exporters"
```

---

### Task 10: CLI

**Files:**
- Create: `musicue/cli.py`
- Modify: `musicue/__init__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
import subprocess
import sys
import pytest


def cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "musicue"] + list(args),
        capture_output=True, text=True,
    )


def test_top_level_help():
    r = cli("--help")
    assert r.returncode == 0
    assert "analyze" in r.stdout


def test_analyze_help():
    r = cli("analyze", "--help")
    assert r.returncode == 0
    assert "SONG" in r.stdout or "song" in r.stdout


def test_compile_help():
    r = cli("compile", "--help")
    assert r.returncode == 0


def test_export_help():
    r = cli("export", "--help")
    assert r.returncode == 0
    assert "--target" in r.stdout


def test_render_help():
    r = cli("render", "--help")
    assert r.returncode == 0


def test_export_unknown_target_exits_nonzero(tmp_path):
    import json
    from musicue.schemas import CueSheet
    cs = CueSheet(source_sha256="x", grammar="g", duration_sec=1.0, tempo_map=[], tracks=[])
    cuesheet_path = tmp_path / "cs.json"
    cuesheet_path.write_text(cs.model_dump_json())
    r = cli("export", str(cuesheet_path), "--target", "nonexistent_format", "--out", str(tmp_path / "out"))
    assert r.returncode != 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_cli.py -v
```
Expected: errors (no cli module)

- [ ] **Step 3: Implement musicue/cli.py**

```python
from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(name="musicue", help="Convert songs to typed event timelines for DCC tools.")

_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
}


@app.command()
def analyze(
    song: Path = typer.Argument(..., help="Input audio file (.wav, .flac, .mp3)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory for analysis.json"),
) -> None:
    """Run Layer 1 analysis → write analysis.json."""
    from musicue.analysis.pipeline import run_analysis
    from musicue.config import MusiCueConfig

    cfg = MusiCueConfig.from_yaml(config) if config else MusiCueConfig()
    if out:
        cfg.runs_dir = out
    result = run_analysis(song, cfg)
    out_path = (out or cfg.runs_dir / song.stem) / "analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2))
    typer.echo(f"Analysis written to {out_path}")


@app.command()
def compile(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    grammar: str = typer.Option("concert_visuals", "--grammar", "-g"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
) -> None:
    """Run Layer 2 compiler: analysis.json → cuesheet.json."""
    from musicue.compile.compiler import compile_analysis
    from musicue.schemas import AnalysisResult

    analysis = AnalysisResult.model_validate_json(analysis_path.read_text())
    cuesheet = compile_analysis(analysis, grammar=grammar)
    out_path = out or analysis_path.parent / "cuesheet.json"
    out_path.write_text(cuesheet.model_dump_json(indent=2))
    typer.echo(f"Cuesheet written to {out_path}")


@app.command()
def export(
    cuesheet_path: Path = typer.Argument(..., help="Path to cuesheet.json"),
    target: str = typer.Option(..., "--target", "-t", help=f"Export target: {', '.join(_EXPORTERS)}"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
) -> None:
    """Run Layer 3 exporter: cuesheet.json → target format."""
    from musicue.schemas import CueSheet
    import importlib

    if target not in _EXPORTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(_EXPORTERS)}", err=True)
        raise typer.Exit(code=1)

    cuesheet = CueSheet.model_validate_json(cuesheet_path.read_text())
    module_name, suffix = _EXPORTERS[target]
    out_path = out or cuesheet_path.parent / f"cuesheet{suffix}"
    mod = importlib.import_module(module_name)
    mod.export(cuesheet, out_path)
    typer.echo(f"Exported to {out_path}")


@app.command()
def render(
    song: Path = typer.Argument(..., help="Input audio file"),
    grammar: str = typer.Option("concert_visuals", "--grammar", "-g"),
    target: str = typer.Option("csv", "--target", "-t"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Convenience: analyze → compile → export in one shot."""
    import importlib
    from musicue.analysis.pipeline import run_analysis
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig

    if target not in _EXPORTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(_EXPORTERS)}", err=True)
        raise typer.Exit(code=1)

    cfg = MusiCueConfig.from_yaml(config) if config else MusiCueConfig()
    analysis = run_analysis(song, cfg)
    cuesheet = compile_analysis(analysis, grammar=grammar)
    module_name, suffix = _EXPORTERS[target]
    out_path = out or Path(song.stem + suffix)
    importlib.import_module(module_name).export(cuesheet, out_path)
    typer.echo(f"Rendered to {out_path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Update musicue/__init__.py**

```python
from musicue.analysis.pipeline import run_analysis as analyze
from musicue.compile.compiler import compile_analysis as compile
from musicue.exporters.json_export import export

__all__ = ["analyze", "compile", "export"]
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_cli.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/cli.py musicue/__init__.py tests/test_cli.py
git commit -m "feat: Typer CLI — analyze/compile/export/render commands"
```

---

### Task 11: End-to-end integration test

**Files:**
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Add integration test**

Append to `tests/test_analysis.py`:

```python
@pytest.mark.integration
def test_full_pipeline_wav_to_csv(tmp_path, synthetic_wav):
    """Full pipeline on real audio. Requires demucs installed. Mark: integration."""
    import csv as csv_mod
    from musicue.analysis.pipeline import run_analysis
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig
    from musicue.exporters.csv import export as csv_export

    cfg = MusiCueConfig()
    cfg.cache_dir = tmp_path / "cache"
    cfg.runs_dir = tmp_path / "runs"

    analysis = run_analysis(synthetic_wav, cfg)
    assert analysis.source.duration_sec > 0
    assert "drums" in analysis.onsets
    assert "lufs" in analysis.curves

    cuesheet = compile_analysis(analysis, grammar="concert_visuals")
    assert cuesheet.duration_sec > 0
    assert len(cuesheet.tracks) >= 1

    out_csv = tmp_path / "output.csv"
    csv_export(cuesheet, out_csv)
    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) > 0
    assert "time_sec" in rows[0]
```

- [ ] **Step 2: Run unit tests (no integration)**

```
pytest tests/ -v -m "not integration"
```
Expected: all unit tests PASS

- [ ] **Step 3: Run integration test (requires demucs)**

First ensure demucs is installed:
```
uv pip install demucs
```
Then run:
```
pytest tests/test_analysis.py::test_full_pipeline_wav_to_csv -v -m integration
```
Expected: PASS (takes ~30-60 seconds — demucs separates the synthetic WAV)

- [ ] **Step 4: Final commit**

```
git add tests/test_analysis.py
git commit -m "test: M0 end-to-end integration test — WAV → analysis → compile → CSV"
```

---

## M0 Complete

At this point `musicue render song.wav --target csv` works end-to-end. Continue with the M1 plan to add the full analysis layer (All-In-One beats/sections, Basic Pitch, phrases, CLAP re-ranker, full curves).
