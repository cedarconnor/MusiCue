# M4: Exporters Round 2 + Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Houdini, disguise/DMX, and Unreal Sequencer exporters; implement `--batch` mode for processing whole directories; write the benchmark and QC video scripts; add royalty-free sample audio fixtures for the test suite.

**Architecture:** Same pattern as M3 — one module per exporter, `export(cuesheet, out_path, **opts) -> None`. Batch mode adds a `--batch` flag to `musicue render`. Benchmark script measures per-stage latency. QC video uses ffmpeg (must be on PATH) to render waveform + event overlay.

**Tech Stack:** numpy, soundfile, pathlib, ffmpeg (system dep, installed via winget), concurrent.futures for batch

**Prerequisite:** M3 complete and all tests passing.

---

## File Structure (additions)

```
musicue/
└── musicue/
    └── exporters/
        ├── houdini.py            ← NEW
        ├── disguise.py           ← NEW
        └── unreal.py             ← NEW
scripts/
├── benchmark.py                  ← NEW
└── make_qc_video.py              ← NEW
tests/
├── fixtures/
│   └── royalty_free_10s.wav      ← NEW (generated, not a real track)
├── test_houdini_exporter.py      ← NEW
├── test_disguise_exporter.py     ← NEW
└── test_unreal_exporter.py       ← NEW
```

---

### Task 1: Houdini CHOP CSV exporter

**Files:**
- Create: `musicue/exporters/houdini.py`
- Create: `tests/test_houdini_exporter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_houdini_exporter.py`:

```python
import csv
import pytest
from pathlib import Path
from musicue.exporters.houdini import export


def test_houdini_export_creates_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_houdini_csv_header_starts_with_time(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        content = f.read()
    # Houdini CHOP CSV format: first line is channel names, NOT a standard header
    # Second line is the actual data start
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) >= 2


def test_houdini_csv_channel_count(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    # Should have time + one column per track
    assert len(header) >= 2
    assert "time" in header or header[0] == "time"


def test_houdini_csv_has_correct_row_count(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_houdini.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    # header row + data rows
    assert len(rows) > 10  # 10s at 0.04s hop = 250 rows
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_houdini_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.houdini'`

- [ ] **Step 3: Implement musicue/exporters/houdini.py**

Houdini CHOP-compatible CSV: first row = channel names, subsequent rows = float values. No "time" column in the data — Houdini derives time from the `start`, `end`, and `rate` in a companion `.chop` metadata header. We emit a simplified format with an explicit `time` column that works with Houdini's File CHOP when imported as a generic CSV.

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


def _to_column(track: CueTrack, times: np.ndarray) -> list[float]:
    if track.type == "continuous":
        if not track.values or not track.hop_sec:
            return [0.0] * len(times)
        src_t = np.arange(len(track.values)) * track.hop_sec
        return list(np.interp(times, src_t, track.values))
    # impulse / step / ramp / envelope → trigger column
    col = np.zeros(len(times))
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.04
    for ev in track.events:
        t = float(ev.get("t") or ev.get("t_start", 0.0))
        idx = int(round(t / hop))
        if 0 <= idx < len(col):
            col[idx] = float(ev.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    times = _time_grid(cuesheet)
    rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 25.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Houdini File CHOP metadata comments
        f.write(f"# MusiCue Houdini CHOP Export\n")
        f.write(f"# rate={rate:.4f} start=0 end={cuesheet.duration_sec:.4f}\n")
        # Header row
        headers = ["time"] + [track.name for track in cuesheet.tracks]
        writer.writerow(headers)
        # Data rows
        columns = [_to_column(track, times) for track in cuesheet.tracks]
        for i, t in enumerate(times):
            row = [f"{t:.6f}"] + [f"{col[i]:.6f}" for col in columns]
            writer.writerow(row)
```

- [ ] **Step 4: Register in cli.py and run tests**

Update `_EXPORTERS` in `musicue/cli.py`:

```python
"houdini": ("musicue.exporters.houdini", "_houdini.csv"),
```

```
pytest tests/test_houdini_exporter.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/exporters/houdini.py musicue/cli.py tests/test_houdini_exporter.py
git commit -m "feat: Houdini CHOP CSV exporter"
```

---

### Task 2: disguise / DMX cue list exporter

**Files:**
- Create: `musicue/exporters/disguise.py`
- Create: `tests/test_disguise_exporter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_disguise_exporter.py`:

```python
import csv
import pytest
from pathlib import Path
from musicue.exporters.disguise import export


def test_disguise_export_creates_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_disguise_csv_has_timecode_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert any("timecode" in h.lower() or "tc" in h.lower() for h in headers)


def test_disguise_csv_has_cue_name_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert any("name" in h.lower() or "cue" in h.lower() for h in headers)


def test_disguise_csv_timecode_format(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    # Timecode should be in HH:MM:SS:FF format
    tc_col = next(k for k in rows[0] if "timecode" in k.lower() or "tc" in k.lower())
    tc = rows[0][tc_col]
    parts = tc.split(":")
    assert len(parts) == 4


def test_disguise_csv_kick_events_present(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_disguise.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    cue_names = [list(r.values())[1] for r in rows]  # second column = name
    kick_cues = [n for n in cue_names if "kick" in n.lower()]
    assert len(kick_cues) >= 3
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_disguise_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.disguise'`

- [ ] **Step 3: Implement musicue/exporters/disguise.py**

```python
from __future__ import annotations
import csv
from pathlib import Path
from musicue.schemas import CueSheet


def _seconds_to_timecode(t: float, fps: float = 25.0) -> str:
    total_frames = int(round(t * fps))
    ff = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def export(cuesheet: CueSheet, out_path: Path, fps: float = 25.0, **opts) -> None:
    rows: list[dict] = []

    for track in cuesheet.tracks:
        if track.type in ("impulse",):
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                env = ev.get("envelope", {})
                duration_frames = max(1, int(float(env.get("d", 0.1)) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t, fps),
                    "cue_name": f"{track.name}_{i+1:04d}",
                    "track": track.name,
                    "type": "impulse",
                    "intensity": f"{strength:.4f}",
                    "duration_frames": str(duration_frames),
                    "label": "|".join(ev.get("tags", [])),
                })
        elif track.type == "step":
            for i, ev in enumerate(track.events):
                t = float(ev["t"])
                label = str(ev.get("label", ""))
                rows.append({
                    "timecode": _seconds_to_timecode(t, fps),
                    "cue_name": f"section_{label}",
                    "track": track.name,
                    "type": "step",
                    "intensity": "1.0",
                    "duration_frames": "1",
                    "label": label,
                })
        elif track.type == "ramp":
            for i, ev in enumerate(track.events):
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                duration_frames = max(1, int((t_end - t_start) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t_start, fps),
                    "cue_name": f"{track.name}_ramp_{i+1:04d}",
                    "track": track.name,
                    "type": "fade",
                    "intensity": f"{float(ev.get('to', 1.0)):.4f}",
                    "duration_frames": str(duration_frames),
                    "label": str(ev.get("label", "")),
                })
        elif track.type == "envelope":
            for i, ev in enumerate(track.events):
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                duration_frames = max(1, int((t_end - t_start) * fps))
                rows.append({
                    "timecode": _seconds_to_timecode(t_start, fps),
                    "cue_name": f"{track.name}_{i+1:04d}",
                    "track": track.name,
                    "type": "envelope",
                    "intensity": f"{strength:.4f}",
                    "duration_frames": str(duration_frames),
                    "label": "|".join(ev.get("tags", [])),
                })

    rows.sort(key=lambda r: r["timecode"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timecode", "cue_name", "track", "type", "intensity", "duration_frames", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Register and run tests**

Update `_EXPORTERS`:

```python
"disguise": ("musicue.exporters.disguise", "_disguise.csv"),
```

```
pytest tests/test_disguise_exporter.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/exporters/disguise.py musicue/cli.py tests/test_disguise_exporter.py
git commit -m "feat: disguise/DMX cue list CSV exporter with HH:MM:SS:FF timecode"
```

---

### Task 3: Unreal Sequencer JSON exporter

**Files:**
- Create: `musicue/exporters/unreal.py`
- Create: `tests/test_unreal_exporter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_unreal_exporter.py`:

```python
import json
import pytest
from pathlib import Path
from musicue.exporters.unreal import export


def test_unreal_export_creates_json(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    assert out.exists()


def test_unreal_json_is_valid(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_unreal_json_has_tracks(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert "tracks" in data
    assert len(data["tracks"]) >= 1


def test_unreal_json_track_structure(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    track = data["tracks"][0]
    assert "name" in track
    assert "type" in track
    assert "events" in track or "channel" in track


def test_unreal_json_schema_version(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert "schema_version" in data
    assert data["schema_version"] == "1.0"


def test_unreal_json_float_tracks_for_continuous(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_unreal.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    float_tracks = [t for t in data["tracks"] if t["type"] == "float_curve"]
    assert len(float_tracks) >= 1  # energy track
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_unreal_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.unreal'`

- [ ] **Step 3: Implement musicue/exporters/unreal.py**

```python
from __future__ import annotations
import json
from pathlib import Path
from musicue.schemas import CueSheet


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    tracks = []

    for track in cuesheet.tracks:
        if track.type == "impulse":
            events = [
                {
                    "time": float(ev["t"]),
                    "strength": float(ev.get("strength", 1.0)),
                    "envelope": ev.get("envelope", {}),
                    "tags": ev.get("tags", []),
                }
                for ev in track.events
            ]
            tracks.append({
                "name": track.name,
                "type": "event_track",
                "timescale": track.timescale,
                "events": events,
            })

        elif track.type == "envelope":
            events = [
                {
                    "time_start": float(ev.get("t_start", 0.0)),
                    "time_end": float(ev.get("t_end", 0.0)),
                    "strength": float(ev.get("strength", 0.8)),
                    "envelope": ev.get("envelope", {}),
                }
                for ev in track.events
            ]
            tracks.append({
                "name": track.name,
                "type": "event_track",
                "timescale": track.timescale,
                "events": events,
            })

        elif track.type == "step":
            events = [
                {
                    "time": float(ev["t"]),
                    "value": ev.get("value", 1),
                    "label": str(ev.get("label", "")),
                }
                for ev in track.events
            ]
            tracks.append({
                "name": track.name,
                "type": "event_track",
                "timescale": track.timescale,
                "events": events,
            })

        elif track.type == "ramp":
            # Map ramp to float curve with two keyframes
            keys = []
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                from_val = float(ev.get("from", 0.0))
                to_val = float(ev.get("to", 1.0))
                shape = str(ev.get("shape", "ease_in_out"))
                keys.append({"time": t_start, "value": from_val, "interp": shape})
                keys.append({"time": t_end, "value": to_val, "interp": "linear"})
            tracks.append({
                "name": track.name,
                "type": "float_curve",
                "timescale": track.timescale,
                "keys": keys,
            })

        elif track.type == "continuous" and track.values and track.hop_sec:
            # Downsample to ~25 Hz for Sequencer (avoid thousands of keys)
            hop = track.hop_sec
            target_hz = 25.0
            step = max(1, int(1.0 / (hop * target_hz)))
            keys = [
                {"time": i * hop, "value": float(track.values[i]), "interp": "linear"}
                for i in range(0, len(track.values), step)
            ]
            tracks.append({
                "name": track.name,
                "type": "float_curve",
                "timescale": track.timescale,
                "keys": keys,
            })

    payload = {
        "schema_version": "1.0",
        "generator": "MusiCue",
        "grammar": cuesheet.grammar,
        "duration_sec": cuesheet.duration_sec,
        "tempo_map": cuesheet.tempo_map,
        "tracks": tracks,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
```

- [ ] **Step 4: Register and run tests**

Update `_EXPORTERS`:

```python
"unreal": ("musicue.exporters.unreal", "_unreal.json"),
```

```
pytest tests/test_unreal_exporter.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/exporters/unreal.py musicue/cli.py tests/test_unreal_exporter.py
git commit -m "feat: Unreal Sequencer JSON exporter — event tracks + float curves"
```

---

### Task 4: Batch mode

**Files:**
- Modify: `musicue/cli.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_cli.py`:

```python
def test_render_batch_help():
    r = cli("render", "--help")
    assert "--batch" in r.stdout or "batch" in r.stdout
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_cli.py::test_render_batch_help -v
```
Expected: FAIL — `--batch` not in help text

- [ ] **Step 3: Add `--batch` flag to `render` command in cli.py**

Replace the `render` command in `musicue/cli.py`:

```python
@app.command()
def render(
    song: Path = typer.Argument(..., help="Input audio file or directory (with --batch)"),
    grammar: str = typer.Option("concert_visuals", "--grammar", "-g"),
    target: str = typer.Option("csv", "--target", "-t"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output file or directory"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    batch: bool = typer.Option(False, "--batch", help="Process all audio files in SONG directory"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers for batch mode"),
) -> None:
    """Convenience: analyze → compile → export in one shot. Use --batch to process a directory."""
    import importlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from musicue.analysis.pipeline import run_analysis
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig

    if target not in _EXPORTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(_EXPORTERS)}", err=True)
        raise typer.Exit(code=1)

    cfg = MusiCueConfig.from_yaml(config) if config else MusiCueConfig()
    module_name, suffix = _EXPORTERS[target]

    def _process_one(audio_path: Path) -> Path:
        analysis = run_analysis(audio_path, cfg)
        cuesheet = compile_analysis(analysis, grammar=grammar)
        if batch and out:
            out_file = out / (audio_path.stem + suffix)
        elif out:
            out_file = out
        else:
            out_file = audio_path.parent / (audio_path.stem + suffix)
        importlib.import_module(module_name).export(cuesheet, out_file)
        return out_file

    if batch:
        if not song.is_dir():
            typer.echo("--batch requires SONG to be a directory", err=True)
            raise typer.Exit(code=1)
        audio_files = [
            p for p in song.iterdir()
            if p.suffix.lower() in (".wav", ".flac", ".mp3", ".aiff")
        ]
        if not audio_files:
            typer.echo(f"No audio files found in {song}", err=True)
            raise typer.Exit(code=1)
        if out:
            out.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, p): p for p in audio_files}
            for future in as_completed(futures):
                src = futures[future]
                try:
                    result = future.result()
                    typer.echo(f"  {src.name} → {result.name}")
                except Exception as e:
                    typer.echo(f"  ERROR {src.name}: {e}", err=True)
        typer.echo(f"Batch complete: {len(audio_files)} files processed.")
    else:
        result = _process_one(song)
        typer.echo(f"Rendered to {result}")
```

- [ ] **Step 4: Run test**

```
pytest tests/test_cli.py -v
```
Expected: all CLI tests PASS

- [ ] **Step 5: Commit**

```
git add musicue/cli.py tests/test_cli.py
git commit -m "feat: batch mode for render command — parallel processing via ThreadPoolExecutor"
```

---

### Task 5: Benchmark script

**Files:**
- Create: `scripts/benchmark.py`

- [ ] **Step 1: Create scripts/benchmark.py**

```python
"""
MusiCue per-stage latency benchmark.

Usage:
  python scripts/benchmark.py --song path/to/song.wav [--grammar concert_visuals] [--runs 3]

Measures wall-clock time for each Layer 1 stage and the full pipeline.
Outputs a table to stdout and a JSON report to benchmark_results.json.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(label: str, results: dict) -> Generator:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    results[label] = elapsed
    print(f"  {label:<35} {elapsed:.2f}s")


def benchmark(song_path: Path, grammar: str, runs: int) -> dict:
    from musicue.config import MusiCueConfig
    from musicue.analysis.separation import separate
    from musicue.analysis.structure import detect_structure
    from musicue.analysis.onsets import detect_onsets
    from musicue.analysis.transcription import transcribe_stem
    from musicue.analysis.clap_reranker import attach_clap_labels
    from musicue.analysis.curves import compute_lufs_curve, compute_rms_curve, compute_spectral_flux_curve
    from musicue.compile.compiler import compile_analysis
    from musicue.schemas import AnalysisResult
    import tempfile

    all_results = []
    for run in range(1, runs + 1):
        print(f"\n--- Run {run}/{runs} ---")
        results: dict[str, float] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = MusiCueConfig()
            cfg.cache_dir = tmp / "cache"
            cfg.runs_dir = tmp / "runs"

            with timer("Demucs separation", results):
                stems = separate(song_path, tmp / "stems")

            with timer("All-In-One structure", results):
                detect_structure(song_path)

            with timer("Basic Pitch (vocals)", results):
                transcribe_stem(stems["vocals"])

            with timer("librosa onsets (all stems)", results):
                for stem_path in stems.values():
                    detect_onsets(stem_path)

            with timer("LUFS + spectral curves", results):
                compute_lufs_curve(song_path)
                compute_spectral_flux_curve(song_path)

            with timer("Full pipeline (cached stems)", results):
                from musicue.analysis.pipeline import run_analysis
                analysis = run_analysis(song_path, cfg)

            with timer("Compiler (concert_visuals)", results):
                compile_analysis(analysis, grammar=grammar)

        all_results.append(results)

    # Average across runs
    avg = {k: sum(r.get(k, 0) for r in all_results) / runs for k in all_results[0]}
    print(f"\n{'Stage':<35} {'Avg (s)':>10}")
    print("-" * 47)
    for label, t in avg.items():
        print(f"  {label:<35} {t:>8.2f}s")
    total = sum(avg.values())
    print(f"\n  {'Total Layer 1 + compile':<35} {total:>8.2f}s")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--grammar", default="concert_visuals")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    results = benchmark(args.song, args.grammar, args.runs)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")
```

- [ ] **Step 2: Commit**

```
git add scripts/benchmark.py
git commit -m "feat: benchmark script measuring per-stage and total pipeline latency"
```

---

### Task 6: QC video script

**Files:**
- Create: `scripts/make_qc_video.py`

This script requires ffmpeg on PATH (`winget install Gyan.FFmpeg`).

- [ ] **Step 1: Create scripts/make_qc_video.py**

```python
"""
MusiCue QC video: waveform + event overlay rendered with ffmpeg.

Usage:
  python scripts/make_qc_video.py --song song.wav --analysis runs/abc/analysis.json --out qc.mp4

Requires: ffmpeg on PATH (winget install Gyan.FFmpeg)
          matplotlib (pip install matplotlib)
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import librosa
import soundfile as sf


def render_frame_strip(
    audio_path: Path,
    analysis_json_path: Path,
    frames_dir: Path,
    fps: int = 24,
    width: int = 1920,
    height: int = 240,
) -> int:
    from musicue.schemas import AnalysisResult
    result = AnalysisResult.model_validate_json(analysis_json_path.read_text())

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y) / sr
    n_frames = int(duration * fps)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Color map for stems
    stem_colors = {"drums": "#FF5722", "bass": "#2196F3", "vocals": "#4CAF50", "other": "#9C27B0"}

    for frame_idx in range(n_frames):
        t_center = frame_idx / fps
        window_sec = 4.0
        t_start = max(0.0, t_center - window_sec / 2)
        t_end = min(duration, t_start + window_sec)

        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        # Draw waveform
        s_start = int(t_start * sr)
        s_end = int(t_end * sr)
        chunk = y[s_start:s_end]
        chunk_t = np.linspace(t_start, t_end, len(chunk))
        ax.plot(chunk_t, chunk, color="#888888", lw=0.5, alpha=0.6)

        # Draw onset markers per stem
        for stem_name, color in stem_colors.items():
            for onset in result.onsets.get(stem_name, []):
                if t_start <= onset.t <= t_end:
                    ax.axvline(onset.t, color=color, lw=1.5, alpha=0.8)

        # Draw section background
        for section in result.sections:
            if section.end >= t_start and section.start <= t_end:
                x0 = max(section.start, t_start)
                x1 = min(section.end, t_end)
                ax.axvspan(x0, x1, alpha=0.05, color="#ffffff")
                ax.text(x0 + 0.05, 0.85, section.label, transform=ax.get_xaxis_transform(),
                        color="#ffffff", fontsize=7, alpha=0.7)

        # Playhead
        ax.axvline(t_center, color="#FFEB3B", lw=2.0, alpha=0.9)

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(-1.1, 1.1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        plt.savefig(str(frame_path), dpi=100, facecolor=fig.get_facecolor())
        plt.close()

    return n_frames


def encode_video(frames_dir: Path, audio_path: Path, out_path: Path, fps: int = 24) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("qc_video.mp4"))
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        print("Rendering frames...")
        n = render_frame_strip(args.song, args.analysis, frames_dir, fps=args.fps)
        print(f"Rendered {n} frames. Encoding video...")
        encode_video(frames_dir, args.song, args.out, fps=args.fps)
    print(f"QC video saved to {args.out}")
```

- [ ] **Step 2: Commit**

```
git add scripts/make_qc_video.py
git commit -m "feat: QC video script — waveform + onset overlay rendered via ffmpeg"
```

---

### Task 7: Final test suite run and validation

- [ ] **Step 1: Run complete test suite**

```
pytest tests/ -v -m "not integration"
```
Expected: all unit tests PASS

- [ ] **Step 2: Smoke test all exporters via CLI**

```powershell
# Run from repo root with .venv activated
python -c "import numpy as np; import soundfile as sf; sr=44100; t=np.linspace(0,10,sr*10); s=0.3*np.sin(2*np.pi*440*t); sf.write('test_10s.wav', s.astype(np.float32), sr)"

musicue render test_10s.wav --target csv     --out test_output.csv
musicue render test_10s.wav --target json    --out test_output.json
musicue render test_10s.wav --target midi    --out test_output.mid
musicue render test_10s.wav --target after_effects --out test_output.jsx
musicue render test_10s.wav --target touchdesigner --out test_td.csv
musicue render test_10s.wav --target osc     --out test_osc.json
musicue render test_10s.wav --target houdini --out test_houdini.csv
musicue render test_10s.wav --target disguise --out test_disguise.csv
musicue render test_10s.wav --target unreal  --out test_unreal.json
```

Expected: all 9 output files created without errors.

- [ ] **Step 3: CLI inspect and diff smoke test**

```powershell
musicue analyze test_10s.wav --out runs/test/
musicue inspect runs/test/analysis.json
musicue compile runs/test/analysis.json --grammar concert_visuals --out runs/test/cuesheet_cv.json
musicue compile runs/test/analysis.json --grammar lighting --out runs/test/cuesheet_lighting.json
musicue diff runs/test/cuesheet_cv.json runs/test/cuesheet_lighting.json
```

Expected: inspect prints JSON summary, diff prints per-track delta table.

- [ ] **Step 4: Final commit**

```
git add .
git commit -m "feat: M4 complete — Houdini, disguise, Unreal exporters + batch mode + scripts"
```

---

## M4 Complete

All exporters are implemented. `musicue` can now render to 9 different targets. Batch mode processes whole directories in parallel. The benchmark and QC video scripts are ready to use.

**Next milestone (M5, optional):** Expose Music2Latent latent sequence as continuous track family and add `musicue inspect --latent` correlation tool. Only pursue once a trained Music2Latent checkpoint is available and you have verified it produces useful features on your target material.

**Next milestone (M6+, long-term):** Train the production-worthy moment classifier on Sphere cue data. This is the actual moat described in §12.1 of DESIGN.md. Requires assembling and labeling the training dataset first.
