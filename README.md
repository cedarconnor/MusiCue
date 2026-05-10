# MusiCue

Convert songs into typed event timelines for DCC tools.

MusiCue is a three-layer pipeline that takes an audio file and produces a structured **cuesheet** — a timeline of impulses, envelopes, ramps, steps, and continuous curves — ready to drive animation, lighting, video, and live-show tools.

## How it works

```
audio.wav ─► [Layer 1: Analyze] ─► analysis.json ─► [Layer 2: Compile] ─► cuesheet.json ─► [Layer 3: Export] ─► <target format>
```

- **Layer 1 (Analysis)** — Demucs source separation, All-In-One beat/section detection (with librosa fallback), librosa onset detection, Basic Pitch polyphonic MIDI, phrase grouping, CLAP semantic re-ranking, LUFS + spectral curves, stereo width/pan, section transition ramps. GPU-accelerated where available; gracefully degrades when optional ML packages are missing.
- **Layer 2 (Compile)** — A YAML grammar DSL turns the analysis into a typed cuesheet. Filter expressions (`drum_class == 'kick'`, `any_label('sub bass drop', min_score=0.6)`, `near_downbeat(0.05)`), per-track scoring with multipliers, hierarchy weights, rarity decay, cooldowns. Four built-in grammars: `concert_visuals`, `character_animation`, `lighting`, `camera_edit`.
- **Layer 3 (Export)** — Nine target formats: CSV, JSON, MIDI, After Effects (.jsx), TouchDesigner (CHOP CSV + events), OSC bundle, Houdini CHOP CSV, disguise/DMX cue list, Unreal Sequencer JSON.

## Quick start

```powershell
# Install
pip install -e ".[dev]"

# Optional ML extras (for the full Layer 1 pipeline)
pip install -e ".[models,clap]"

# Render a song to a TouchDesigner CHOP CSV in one shot
musicue render song.wav --target touchdesigner --out song_td.csv

# Or step through the pipeline manually
musicue analyze song.wav --out runs/song/
musicue inspect runs/song/analysis.json
musicue compile runs/song/analysis.json --grammar concert_visuals --out cuesheet.json
musicue export cuesheet.json --target after_effects --out cuesheet.jsx
```

## CLI commands

| Command | Purpose |
|---|---|
| `musicue analyze <song>` | Layer 1 — write `analysis.json` |
| `musicue compile <analysis.json>` | Layer 2 — write `cuesheet.json` |
| `musicue export <cuesheet.json> --target <name>` | Layer 3 — emit target format |
| `musicue render <song>` | All three layers in one shot |
| `musicue render <dir> --batch --workers 4` | Parallel batch over a directory |
| `musicue inspect <analysis.json>` | Print human-readable summary |
| `musicue plot <analysis.json> --out plot.png` | Render matplotlib timeline |
| `musicue listen <cuesheet.json> --audio <song.wav>` | Render QC click-track WAV |
| `musicue diff <a.json> <b.json>` | Compare two cuesheets per-track |

## Built-in grammars

- **`concert_visuals`** — downbeat pulse, per-class drum tracks, drop labels, vocal phrase envelopes, section ramps, energy curve
- **`character_animation`** — vocal/melody phrase envelopes, downbeat accents, per-stem energy
- **`lighting`** — fast attack drum tracks, hihat with rarity decay, build-up cues, intensity curve
- **`camera_edit`** — section cuts as primary, downbeat bar markers, impact hits, slow energy curve

Custom grammars are plain YAML — see `musicue/grammars/concert_visuals.yaml` for the format.

## Exporters

| Target | Output | Notes |
|---|---|---|
| `csv` | Single CSV with `time_sec` column | Generic time-series |
| `json` | Pydantic-validated JSON | Round-trip safe |
| `midi` | Standard MIDI file | Impulse→GM drum notes, continuous→CC74, step→meta markers |
| `after_effects` | ExtendScript `.jsx` | Null layers + Slider Control keyframes + comp markers |
| `touchdesigner` | CHOP CSV + events CSV | `time` column convention, plus Table DAT events |
| `osc` | JSON bundle + `play_osc.py` | UDP playback script bundled |
| `houdini` | CHOP-compatible CSV | Metadata header, `time` channel, per-track channels |
| `disguise` | Cue list CSV | HH:MM:SS:FF timecode at configurable fps |
| `unreal` | Sequencer JSON | Event tracks + float curves with interp keys |

## Configuration

Optional `config.yaml` for tuning:

```yaml
analysis:
  demucs_model: htdemucs_ft
  beat_backend: allin1            # or 'librosa' to skip All-In-One
  curve_hop_sec: 0.04
  clap_top_k: 3
  clap_threshold: 0.55
  phrase_gap_sec:
    vocals: 0.6
    other: 0.4

cache_dir: ~/.musicue/cache
runs_dir: runs
```

`musicue render --config config.yaml song.wav ...`

## Optional ML dependencies

The pipeline degrades gracefully when these aren't installed:

- **`allin1`** — joint beat/downbeat/section detection. Falls back to `librosa.beat.beat_track` + empty sections list.
- **`basic-pitch`** — polyphonic MIDI transcription. Falls back to empty MIDI.
- **`laion-clap`** — semantic event labeling. Falls back to no labels (≈4 GB model download on first use).
- **`models/drum_cnn.pt`** — drum classifier checkpoint. Falls back to onsets without `drum_class`. Train via `scripts/train_drum_classifier.py` on ENST-Drums + MDB Drums.

## Development

```powershell
# Run the test suite (unit tests; integration tests require demucs)
pytest

# Run all including integration
pytest -m ""

# Lint
ruff check .

# Type check
pyright
```

### Web UI (v0.1a, dev mode)

The frontend bundle isn't tracked in git (`musicue/ui/static/` is gitignored)
and there's no `pip install` build hook yet -- packaging the wheel with the
React assets is a v1.0 milestone. Until then, build it manually after a fresh
checkout:

```powershell
cd musicue/ui/web
npm install
npm run build      # writes the bundle to ../static/

# Then back at the repo root, run the server:
cd ../../..
python -m uvicorn musicue.ui.server:create_app --factory
```

Open <http://127.0.0.1:8000/library>. Default bind is localhost; do NOT bind
to `0.0.0.0` over an untrusted network -- the URL-ingest endpoint can fetch
arbitrary URLs (with private/loopback IPs blocked) and would benefit from
auth before being exposed.

Test count at HEAD: **190 unit tests passing** across 5 milestones (M0 walking skeleton, M1 full analysis, M2 compiler+grammars+drum CNN, M3 exporters round 1, M4 exporters round 2 + batch + scripts).

## Operational scripts

- `scripts/benchmark.py` — per-stage latency timer for the full pipeline
- `scripts/make_qc_video.py` — waveform + onset/section overlay video (requires `ffmpeg` on PATH)
- `scripts/train_drum_classifier.py` — drum classifier CNN training (requires HDF5 dataset)

## Status

M0-M4 implementation complete. The full pipeline runs end-to-end on Windows 11 + Python 3.11 with PyTorch CUDA 12.4. Stable APIs:

- `analysis.json` — schema v1.1 (frozen)
- `cuesheet.json` — schema v1.1 (frozen)
- 9 exporter targets shipped

## License

(Project is currently unlicensed — add a LICENSE file before public release.)
