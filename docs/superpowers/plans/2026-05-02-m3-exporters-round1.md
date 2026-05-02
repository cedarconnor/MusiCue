# M3: Exporters Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MIDI, After Effects `.jsx`, TouchDesigner CHOP CSV, and OSC bundle exporters — the four highest-demand targets for Cedar's workflow.

**Architecture:** Each exporter is a single module in `musicue/exporters/` implementing `export(cuesheet, out_path, **opts) -> None`. The CLI `export` command already dispatches to these modules; this plan only adds the modules and registers them in `_EXPORTERS`. Each exporter is tested with a synthetic `CueSheet` that exercises all five event types.

**Tech Stack:** mido (MIDI), python-osc (OSC), standard library only for AE jsx + TD CSV

**Prerequisite:** M2 complete and all tests passing.

---

## File Structure (additions)

```
musicue/
└── musicue/
    └── exporters/
        ├── midi.py                ← NEW
        ├── aftereffects.py        ← NEW
        ├── touchdesigner.py       ← NEW
        └── osc.py                 ← NEW
tests/
├── test_midi_exporter.py          ← NEW
├── test_ae_exporter.py            ← NEW
├── test_td_exporter.py            ← NEW
└── test_osc_exporter.py           ← NEW
```

---

### Shared Test Fixture (append to tests/conftest.py)

Add this fixture to `tests/conftest.py` so all exporter tests can use it:

```python
from musicue.schemas import CueSheet, CueTrack

@pytest.fixture()
def full_cuesheet() -> CueSheet:
    """CueSheet exercising all five track types."""
    return CueSheet(
        source_sha256="abc123",
        grammar="concert_visuals",
        duration_sec=10.0,
        tempo_map=[{"t": 0.0, "bpm": 120.0}],
        tracks=[
            CueTrack(
                name="kick",
                type="impulse",
                timescale="micro",
                events=[
                    {"t": 0.5, "strength": 0.9, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": ["kick"]},
                    {"t": 1.0, "strength": 0.8, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": ["kick"]},
                    {"t": 2.0, "strength": 0.7, "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0}, "tags": ["kick"]},
                ],
            ),
            CueTrack(
                name="vocal_phrase",
                type="envelope",
                timescale="meso",
                events=[
                    {"t_start": 3.0, "t_end": 5.5, "strength": 0.85,
                     "envelope": {"a": 0.30, "d": 0.20, "s": 0.7, "r": 0.50}, "tags": ["vocal_entry"]},
                ],
            ),
            CueTrack(
                name="section_change",
                type="step",
                timescale="macro",
                events=[
                    {"t": 0.0, "value": 1, "label": "intro"},
                    {"t": 5.0, "value": 2, "label": "chorus"},
                ],
            ),
            CueTrack(
                name="section_ramp",
                type="ramp",
                timescale="macro",
                events=[
                    {"t_start": 4.0, "t_end": 5.0, "from": 0.0, "to": 1.0, "shape": "ease_in_out", "label": "intro→chorus"},
                ],
            ),
            CueTrack(
                name="energy",
                type="continuous",
                timescale="macro",
                hop_sec=0.1,
                values=[-20.0 + i * 0.05 for i in range(100)],
            ),
        ],
    )
```

- [ ] **Step 1: Add `full_cuesheet` fixture to conftest.py**

Append the fixture above to `tests/conftest.py`.

- [ ] **Step 2: Commit conftest update**

```
git add tests/conftest.py
git commit -m "test: add full_cuesheet fixture covering all five track types"
```

---

### Task 1: MIDI exporter

**Files:**
- Create: `musicue/exporters/midi.py`
- Create: `tests/test_midi_exporter.py`

- [ ] **Step 1: Install mido**

```
uv pip install mido
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_midi_exporter.py`:

```python
import pytest
import mido
from pathlib import Path
from musicue.exporters.midi import export


def test_midi_export_creates_file(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_midi_export_is_valid_midi(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) >= 1


def test_midi_export_has_correct_tempo(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    tempo_msgs = [m for t in mid.tracks for m in t if m.type == "set_tempo"]
    assert len(tempo_msgs) >= 1
    # 120 BPM = 500000 microseconds per beat
    assert tempo_msgs[0].tempo == pytest.approx(500000, abs=5000)


def test_midi_export_has_note_events(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    note_ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert len(note_ons) >= 3  # at least the 3 kick events


def test_midi_export_continuous_as_cc(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    cc_msgs = [m for t in mid.tracks for m in t if m.type == "control_change"]
    assert len(cc_msgs) > 0  # energy track → CC messages


def test_midi_export_step_as_text_marker(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.mid"
    export(full_cuesheet, out)
    mid = mido.MidiFile(str(out))
    markers = [m for t in mid.tracks for m in t if m.type == "marker"]
    labels = [m.text for m in markers]
    assert any("intro" in lbl or "chorus" in lbl for lbl in labels)
```

- [ ] **Step 3: Run to verify failure**

```
pytest tests/test_midi_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.midi'`

- [ ] **Step 4: Implement musicue/exporters/midi.py**

```python
from __future__ import annotations
from pathlib import Path
import mido
import numpy as np
from musicue.schemas import CueSheet, CueTrack

# MIDI channel assignments by track name
_CHANNEL_MAP: dict[str, int] = {
    "kick": 9,       # channel 10 (0-indexed: 9) = GM drums
    "snare": 9,
    "hat": 9,
    "hihat": 9,
    "downbeat": 9,
    "downbeat_pulse": 9,
}
# GM drum note numbers
_NOTE_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hat": 42,
    "hihat": 42,
    "downbeat": 75,
    "downbeat_pulse": 75,
    "vocal_phrase": 64,
    "drop": 37,
    "impact": 39,
}
_DEFAULT_NOTE = 60
_ENERGY_CC = 74  # CC74: filter cutoff — repurposed as energy follower


def _ticks(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    beats = seconds * 1_000_000 / tempo_us
    return max(0, int(round(beats * ticks_per_beat)))


def export(cuesheet: CueSheet, out_path: Path, ticks_per_beat: int = 480, **opts) -> None:
    # Determine BPM from tempo_map (default 120)
    bpm = 120.0
    if cuesheet.tempo_map:
        bpm = float(cuesheet.tempo_map[0].get("bpm", 120.0))
    tempo_us = int(60_000_000 / bpm)

    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    # One MIDI track per CueSheet track
    for track in cuesheet.tracks:
        midi_track = mido.MidiTrack()
        mid.tracks.append(midi_track)
        midi_track.append(mido.MetaMessage("track_name", name=track.name, time=0))

        channel = _CHANNEL_MAP.get(track.name, 0)
        note = _NOTE_MAP.get(track.name, _DEFAULT_NOTE)
        msgs: list[tuple[int, mido.Message | mido.MetaMessage]] = []

        if track.type == "impulse":
            for event in track.events:
                t = float(event["t"])
                velocity = max(1, min(127, int(float(event.get("strength", 0.8)) * 127)))
                tick = _ticks(t, ticks_per_beat, tempo_us)
                env = event.get("envelope", {})
                dur = float(env.get("d", 0.1))
                off_tick = _ticks(t + dur, ticks_per_beat, tempo_us)
                msgs.append((tick, mido.Message("note_on", channel=channel, note=note,
                                                velocity=velocity, time=0)))
                msgs.append((off_tick, mido.Message("note_off", channel=channel, note=note,
                                                    velocity=0, time=0)))

        elif track.type == "envelope":
            for event in track.events:
                t_start = float(event.get("t_start", 0.0))
                t_end = float(event.get("t_end", t_start + 1.0))
                velocity = max(1, min(127, int(float(event.get("strength", 0.8)) * 127)))
                tick_on = _ticks(t_start, ticks_per_beat, tempo_us)
                tick_off = _ticks(t_end, ticks_per_beat, tempo_us)
                msgs.append((tick_on, mido.Message("note_on", channel=channel, note=note,
                                                   velocity=velocity, time=0)))
                msgs.append((tick_off, mido.Message("note_off", channel=channel, note=note,
                                                    velocity=0, time=0)))

        elif track.type == "step":
            for event in track.events:
                t = float(event["t"])
                tick = _ticks(t, ticks_per_beat, tempo_us)
                label = str(event.get("label", ""))
                msgs.append((tick, mido.MetaMessage("marker", text=label, time=0)))

        elif track.type == "continuous" and track.values and track.hop_sec:
            # Downsample to ~10 Hz for CC density (avoid flooding MIDI with CC)
            hop = track.hop_sec
            target_hz = 10.0
            step = max(1, int(target_hz * hop))
            for i in range(0, len(track.values), step):
                t = i * hop
                val = float(track.values[i])
                # Normalize to [0, 127]: assume values are in [-70, 0] dB (LUFS range)
                cc_val = max(0, min(127, int((val + 70) / 70 * 127)))
                tick = _ticks(t, ticks_per_beat, tempo_us)
                msgs.append((tick, mido.Message("control_change", channel=channel,
                                                control=_ENERGY_CC, value=cc_val, time=0)))

        # Sort by tick and convert to delta times
        msgs.sort(key=lambda x: x[0])
        prev_tick = 0
        for abs_tick, msg in msgs:
            delta = abs_tick - prev_tick
            msg.time = delta
            midi_track.append(msg)
            prev_tick = abs_tick

        midi_track.append(mido.MetaMessage("end_of_track", time=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))
```

- [ ] **Step 5: Register in cli.py**

In `musicue/cli.py`, update `_EXPORTERS`:

```python
_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
    "midi": ("musicue.exporters.midi", ".mid"),
}
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_midi_exporter.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 7: Commit**

```
git add musicue/exporters/midi.py musicue/cli.py tests/test_midi_exporter.py
git commit -m "feat: MIDI exporter — impulse→notes, step→markers, continuous→CC74, envelope→note holds"
```

---

### Task 2: After Effects exporter

**Files:**
- Create: `musicue/exporters/aftereffects.py`
- Create: `tests/test_ae_exporter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ae_exporter.py`:

```python
import pytest
from pathlib import Path
from musicue.exporters.aftereffects import export


def test_ae_export_creates_jsx_file(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_ae_export_is_valid_jsx(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "app.project" in content or "function" in content


def test_ae_export_contains_track_names(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "MusiCue_kick" in content
    assert "MusiCue_energy" in content


def test_ae_export_contains_composition_markers(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    assert "marker" in content.lower() or "Marker" in content


def test_ae_export_contains_keyframes_for_continuous(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.jsx"
    export(full_cuesheet, out)
    content = out.read_text(encoding="utf-8")
    # Energy continuous track → keyframes
    assert "setValue" in content or "setValueAtTime" in content
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_ae_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.aftereffects'`

- [ ] **Step 3: Implement musicue/exporters/aftereffects.py**

```python
from __future__ import annotations
from pathlib import Path
from musicue.schemas import CueSheet


def _jsx_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def export(cuesheet: CueSheet, out_path: Path, fps: float = 24.0, **opts) -> None:
    lines: list[str] = []
    a = lines.append

    a("// MusiCue After Effects ExtendScript")
    a(f'// Grammar: {cuesheet.grammar}  Duration: {cuesheet.duration_sec:.3f}s')
    a("(function() {")
    a("  var comp = app.project.activeItem;")
    a("  if (!comp || !(comp instanceof CompItem)) {")
    a('    alert("MusiCue: No active composition found.");')
    a("    return;")
    a("  }")
    a(f"  var fps = {fps};")
    a("")

    # Composition-level markers for impulse + step tracks
    a("  // Composition markers for impulse and step events")
    a("  var compMarkers = comp.markerProperty;")
    for track in cuesheet.tracks:
        if track.type in ("impulse",):
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                label = f"{track.name} s={strength:.2f}"
                a(f'  var mk_{track.name.replace("-","_")} = new MarkerValue("{_jsx_escape(label)}");')
                a(f"  compMarkers.setValueAtTime({t:.4f}, mk_{track.name.replace('-','_')});")
        elif track.type == "step":
            for ev in track.events:
                t = float(ev["t"])
                label = str(ev.get("label", ""))
                a(f'  var mkS = new MarkerValue("section: {_jsx_escape(label)}");')
                a(f"  compMarkers.setValueAtTime({t:.4f}, mkS);")

    a("")
    a("  app.beginUndoGroup('MusiCue Import');")

    # Slider Control layers for continuous + impulse tracks
    for track in cuesheet.tracks:
        safe_name = f"MusiCue_{track.name}"
        a(f"  // Track: {track.name} ({track.type})")
        a(f"  var layer_{track.name.replace('-','_')} = comp.layers.addNull();")
        a(f'  layer_{track.name.replace("-","_")}.name = "{safe_name}";')
        a(f"  var effect_{track.name.replace('-','_')} = layer_{track.name.replace('-','_')}.Effects.addProperty('ADBE Slider Control');")
        slider_ref = f"effect_{track.name.replace('-','_')}('ADBE Slider Control-0001')"

        if track.type == "continuous" and track.values and track.hop_sec:
            hop = track.hop_sec
            for i, val in enumerate(track.values):
                t = i * hop
                # normalize LUFS to 0-100 slider range
                slider_val = max(0.0, min(100.0, (val + 70) / 70 * 100))
                a(f"  {slider_ref}.setValueAtTime({t:.4f}, {slider_val:.2f});")

        elif track.type == "impulse":
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                env = ev.get("envelope", {})
                a_time = float(env.get("a", 0.005))
                d_time = float(env.get("d", 0.1))
                a(f"  {slider_ref}.setValueAtTime({max(0.0, t - 0.001):.4f}, 0.0);")
                a(f"  {slider_ref}.setValueAtTime({t + a_time:.4f}, {strength * 100:.2f});")
                a(f"  {slider_ref}.setValueAtTime({t + a_time + d_time:.4f}, 0.0);")

        elif track.type == "envelope":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                env = ev.get("envelope", {})
                a(f"  {slider_ref}.setValueAtTime({max(0.0, t_start - 0.001):.4f}, 0.0);")
                a(f"  {slider_ref}.setValueAtTime({t_start + float(env.get('a', 0.3)):.4f}, {strength * 100:.2f});")
                a(f"  {slider_ref}.setValueAtTime({t_end:.4f}, 0.0);")

    a("")
    a("  app.endUndoGroup();")
    a('  alert("MusiCue: Import complete. " + comp.layers.length + " layers added.");')
    a("})();")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Register in cli.py**

```python
_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
    "midi": ("musicue.exporters.midi", ".mid"),
    "after_effects": ("musicue.exporters.aftereffects", ".jsx"),
}
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_ae_exporter.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/exporters/aftereffects.py musicue/cli.py tests/test_ae_exporter.py
git commit -m "feat: After Effects ExtendScript exporter — markers, Slider Control keyframes"
```

---

### Task 3: TouchDesigner exporter

**Files:**
- Create: `musicue/exporters/touchdesigner.py`
- Create: `tests/test_td_exporter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_td_exporter.py`:

```python
import csv
import json
import pytest
from pathlib import Path
from musicue.exporters.touchdesigner import export


def test_td_export_creates_chop_csv(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    assert out.exists()


def test_td_export_chop_has_time_column(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "time" in headers  # TD CHOP uses 'time' not 'time_sec'


def test_td_export_chop_has_all_tracks(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    with open(out, newline="") as f:
        headers = csv.DictReader(f).fieldnames
    assert "kick" in headers
    assert "energy" in headers


def test_td_export_events_csv_created(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    events_csv = tmp_path / "cuesheet_events.csv"
    assert events_csv.exists()


def test_td_events_csv_has_track_time_strength(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet.csv"
    export(full_cuesheet, out)
    events_csv = tmp_path / "cuesheet_events.csv"
    with open(events_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames
    assert "track" in headers
    assert "t" in headers
    assert "strength" in headers
    assert len(rows) > 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_td_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.touchdesigner'`

- [ ] **Step 3: Implement musicue/exporters/touchdesigner.py**

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
        t = float(event.get("t") or event.get("t_start", 0.0))
        idx = int(round(t / hop))
        if 0 <= idx < len(col):
            col[idx] = float(event.get("strength", 1.0))
    return list(col)


def export(cuesheet: CueSheet, out_path: Path, **opts) -> None:
    times = _time_grid(cuesheet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CHOP CSV: one channel per track, 'time' column (TD convention)
    columns: dict[str, list[float]] = {"time": list(times)}
    for track in cuesheet.tracks:
        if track.type == "continuous":
            columns[track.name] = _continuous_col(track, times)
        else:
            columns[track.name] = _impulse_col(track, times)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for i in range(len(times)):
            writer.writerow({k: v[i] for k, v in columns.items()})

    # Events CSV: discrete event list for Table DAT
    events_path = out_path.parent / (out_path.stem + "_events.csv")
    event_rows: list[dict] = []
    for track in cuesheet.tracks:
        if track.type in ("impulse", "envelope"):
            for ev in track.events:
                t = float(ev.get("t") or ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t + float(ev.get("envelope", {}).get("d", 0.1))))
                event_rows.append({
                    "track": track.name,
                    "t": f"{t:.4f}",
                    "t_end": f"{t_end:.4f}",
                    "strength": f"{float(ev.get('strength', 1.0)):.4f}",
                    "tags": "|".join(ev.get("tags", [])),
                })
        elif track.type == "step":
            for ev in track.events:
                event_rows.append({
                    "track": track.name,
                    "t": f"{float(ev['t']):.4f}",
                    "t_end": f"{float(ev['t']):.4f}",
                    "strength": "1.0",
                    "tags": str(ev.get("label", "")),
                })

    if event_rows:
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["track", "t", "t_end", "strength", "tags"])
            writer.writeheader()
            writer.writerows(event_rows)
```

- [ ] **Step 4: Register in cli.py**

```python
_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
    "midi": ("musicue.exporters.midi", ".mid"),
    "after_effects": ("musicue.exporters.aftereffects", ".jsx"),
    "touchdesigner": ("musicue.exporters.touchdesigner", ".csv"),
}
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_td_exporter.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```
git add musicue/exporters/touchdesigner.py musicue/cli.py tests/test_td_exporter.py
git commit -m "feat: TouchDesigner CHOP CSV + events CSV exporter"
```

---

### Task 4: OSC bundle exporter

**Files:**
- Create: `musicue/exporters/osc.py`
- Create: `tests/test_osc_exporter.py`

- [ ] **Step 1: Install python-osc**

```
uv pip install python-osc
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_osc_exporter.py`:

```python
import json
import pytest
from pathlib import Path
from musicue.exporters.osc import export


def test_osc_export_creates_json_bundle(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    assert out.exists()


def test_osc_bundle_is_valid_json(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "messages" in data


def test_osc_bundle_message_fields(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    messages = data["messages"]
    assert len(messages) > 0
    msg = messages[0]
    assert "t" in msg
    assert "address" in msg
    assert "args" in msg


def test_osc_bundle_address_pattern(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    addresses = [m["address"] for m in data["messages"]]
    assert any(addr.startswith("/musicue/") for addr in addresses)


def test_osc_bundle_kick_events_present(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    data = json.loads(out.read_text())
    kick_msgs = [m for m in data["messages"] if "kick" in m["address"]]
    assert len(kick_msgs) == 3  # 3 kick events in full_cuesheet


def test_osc_player_script_created(tmp_path, full_cuesheet):
    out = tmp_path / "cuesheet_osc.json"
    export(full_cuesheet, out)
    player = tmp_path / "play_osc.py"
    assert player.exists()
    content = player.read_text()
    assert "pythonosc" in content or "python-osc" in content or "osc_message" in content.lower()
```

- [ ] **Step 3: Run to verify failure**

```
pytest tests/test_osc_exporter.py -v
```
Expected: `ModuleNotFoundError: No module named 'musicue.exporters.osc'`

- [ ] **Step 4: Implement musicue/exporters/osc.py**

```python
from __future__ import annotations
import json
from pathlib import Path
from musicue.schemas import CueSheet


def export(cuesheet: CueSheet, out_path: Path, host: str = "127.0.0.1", port: int = 9000, **opts) -> None:
    messages: list[dict] = []

    for track in cuesheet.tracks:
        address = f"/musicue/{track.name}"

        if track.type in ("impulse",):
            for ev in track.events:
                t = float(ev["t"])
                strength = float(ev.get("strength", 1.0))
                messages.append({"t": t, "address": address, "args": [strength]})

        elif track.type == "envelope":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                strength = float(ev.get("strength", 0.8))
                messages.append({"t": t_start, "address": f"{address}/on", "args": [strength]})
                messages.append({"t": t_end, "address": f"{address}/off", "args": [0.0]})

        elif track.type == "step":
            for ev in track.events:
                t = float(ev["t"])
                value = ev.get("value", 1)
                label = str(ev.get("label", ""))
                messages.append({"t": t, "address": f"{address}/label", "args": [label]})
                messages.append({"t": t, "address": address, "args": [float(value)]})

        elif track.type == "ramp":
            for ev in track.events:
                t_start = float(ev.get("t_start", 0.0))
                t_end = float(ev.get("t_end", t_start + 1.0))
                messages.append({"t": t_start, "address": f"{address}/ramp_start", "args": [0.0]})
                messages.append({"t": t_end, "address": f"{address}/ramp_end", "args": [1.0]})

        elif track.type == "continuous" and track.values and track.hop_sec:
            # Downsample to ~10 Hz for OSC
            hop = track.hop_sec
            step = max(1, int(hop * 10))
            for i in range(0, len(track.values), step):
                t = i * hop
                val = float(track.values[i])
                messages.append({"t": t, "address": address, "args": [val]})

    messages.sort(key=lambda m: m["t"])
    bundle = {
        "grammar": cuesheet.grammar,
        "duration_sec": cuesheet.duration_sec,
        "target_host": host,
        "target_port": port,
        "message_count": len(messages),
        "messages": messages,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))

    # Write player script
    player_path = out_path.parent / "play_osc.py"
    player_path.write_text(
        '#!/usr/bin/env python3\n'
        '"""Play a MusiCue OSC bundle. Requires python-osc: pip install python-osc"""\n'
        'import json, time, sys\n'
        'from pythonosc.udp_client import SimpleUDPClient\n'
        '\n'
        'bundle_path = sys.argv[1] if len(sys.argv) > 1 else "cuesheet_osc.json"\n'
        'bundle = json.loads(open(bundle_path).read())\n'
        'client = SimpleUDPClient(bundle["target_host"], bundle["target_port"])\n'
        'messages = bundle["messages"]\n'
        'start_time = time.monotonic()\n'
        'for i, msg in enumerate(messages):\n'
        '    target = start_time + msg["t"]\n'
        '    now = time.monotonic()\n'
        '    if target > now:\n'
        '        time.sleep(target - now)\n'
        '    client.send_message(msg["address"], msg["args"])\n'
        'print("Playback complete.")\n'
    )


```

- [ ] **Step 5: Register in cli.py**

```python
_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
    "midi": ("musicue.exporters.midi", ".mid"),
    "after_effects": ("musicue.exporters.aftereffects", ".jsx"),
    "touchdesigner": ("musicue.exporters.touchdesigner", ".csv"),
    "osc": ("musicue.exporters.osc", "_osc.json"),
}
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_osc_exporter.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 7: Commit**

```
git add musicue/exporters/osc.py musicue/cli.py tests/test_osc_exporter.py
git commit -m "feat: OSC bundle exporter with playback script"
```

---

### Task 5: Run full test suite

- [ ] **Step 1: Run all tests**

```
pytest tests/ -v -m "not integration"
```
Expected: all unit tests PASS

- [ ] **Step 2: Run M3 integration smoke test**

Pick any short `.wav` file and verify all four new exporters work end-to-end via CLI:

```powershell
musicue render song.wav --target midi --out song.mid
musicue render song.wav --target after_effects --out song.jsx
musicue render song.wav --target touchdesigner --out song_td.csv
musicue render song.wav --target osc --out song_osc.json
```
Expected: four output files created without errors.

- [ ] **Step 3: Final commit**

```
git add .
git commit -m "test: M3 exporter smoke test verification"
```

---

## M3 Complete

MIDI, After Effects, TouchDesigner, and OSC exporters are live. Continue with M4 to add Houdini, disguise/DMX, and Unreal Sequencer exporters, batch mode, and the benchmark + QC video scripts.
