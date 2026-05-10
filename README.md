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

### Web UI (v0.2a, dev mode)

A local web app for browsing your library and inspecting analyses. The whole
thing runs on your own machine — there is no cloud component.

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
python -m musicue ui --no-open
```

Open <http://localhost:8765/>. Default bind is localhost; do NOT bind to
`0.0.0.0` over an untrusted network -- the URL-ingest endpoint can fetch
arbitrary URLs (with private/loopback IPs blocked) and would benefit from
auth before being exposed.

Test count at HEAD: **340 unit tests passing** across 5 backend milestones plus the v0.1a–d / v0.2a web UI work.

#### What you'll see

The UI has two pages: a **Library** to manage songs, and an **Editor** to look at one song in depth.

**Library**

![Library page](docs/screenshots/library.png)

- **Drag a file** onto the drop zone, or **paste a URL** (YouTube, SoundCloud, Bandcamp, etc.) to bring a new song in. The server downloads, separates the stems, and runs analysis in the background.
- **Search and filter** by title, by whether stems are ready, by tempo bucket, or by how recently the song was added.
- **Trash** lets you remove songs without immediately deleting the audio; you can restore them or empty the trash later.

**Editor**

![Editor — full view, zoomed in for clarity](docs/screenshots/editor_zoomed.png)

The Editor stacks five horizontal lanes from top to bottom:

1. **Mix waveform** — the song as you'd hear it.
2. **Section bar** — labelled blocks (intro, verse, chorus, …) showing the song's structure.
3. **Onset / beat strip** — every drum hit, attack, and beat on the mix.
4. **Four stem lanes** — drums, bass, vocals, other — each as its own waveform with its own coloured onset markers.
5. **Transport** — play / pause, time readout, click-track toggle.
6. **Curves panel** — a single continuous measurement (loudness, brightness, etc.) drawn over the song's full duration.

Below the panel, a hint reminds you of the keyboard shortcuts: **Space** play/pause, **I** loop-in, **O** loop-out, **L** toggle loop, **Esc** clear loop.

#### Feature details

##### Header (tempo, duration, loudness)

Top-left of the Editor shows the song's duration, average tempo, and overall loudness in **LUFS** (a broadcast-standard "how loud does this feel" number — closer to 0 is louder; -14 LUFS is roughly streaming-platform target). When the song speeds up or slows down meaningfully, the BPM display switches from a single number to a range (e.g. **75 BPM** → **70–95 BPM**) so you know at a glance the tempo isn't constant.

##### Mix waveform + section bar (with transition ramps)

The grey waveform at the top is the full mix. Just under it, the **section bar** shows where each part of the song lives — intro, verse, chorus, bridge, solo, outro — with the label written into each block.

At every section boundary you'll see a small coloured **ramp shape** rising into the next section. The ramp's curvature tells you *how* the song transitions:

- A **steep, late-rising shape** is an "ease-in" — energy stays low and then snaps up at the last moment. Classic chorus drop.
- A **gentle early rise** is an "ease-out" — the song eases out of the previous section gradually.
- An **S-curve** is "ease-in-out" — a smooth swell.
- A **straight line** is a linear transition.

The colour of the ramp matches the section it's leading *into* (warm yellow for chorus, cool blue for intro, grey for outro, etc.) so you can scan the structure at a glance.

**Hover any ramp** to see a tooltip with the underlying numbers — the spectral-flux rise (how dramatic the textural change is) and the LUFS rise (how much louder it gets).

![Section transition tooltip on hover](docs/screenshots/editor_transition_tooltip.png)

##### Per-stem lanes with RMS tint and onset markers

Each stem lane (drums, bass, vocals, other) shows that stem's waveform in its own colour, with two extra layers drawn on top:

- **Onset markers** — short vertical ticks at every detected attack. Drum onsets are colour-coded by class (kick, snare, hihat, etc.) when the drum classifier is shipped; otherwise they share the stem's colour.
- **Phrase blocks** (vocals and "other" only) — translucent rectangles showing where a singer or melody is actually phrasing, grouped by short pauses. Useful for spotting where a vocal entrance hits.
- **RMS tint** — a faint coloured background that brightens when the stem is loud and fades when it's quiet. So even at low zoom, you can see that, for example, the vocals are silent in the intro but heavy in the second chorus.

The **solo button** at the left of each lane lets you mute everything except that stem. Click it again to return to the full mix.

##### Curves panel

![Curves panel showing LUFS](docs/screenshots/editor_overview.png)

A single continuous curve drawn underneath the transport, the full width of the song. Pick which one to view from the dropdown:

- **LUFS** — perceived loudness, in the same units as the header. Useful for finding quiet verses and loud drops.
- **Spectral Centroid** — "brightness" of the sound, in Hertz. Higher = more treble / more shimmer; lower = darker, bassier mix.
- **Spectral Flux** — how fast the sound is changing moment-to-moment. Spikes line up with hits, drops, and transitions.
- **Stereo Width** — how spread out the mix is between left and right ear. Mono moments read as zero; wide stereo verbs read high.
- **Stereo Pan** — how far the centre-of-mass leans left vs. right (-1 = hard left, +1 = hard right, 0 = centred).

![Curves panel — Spectral Centroid (fixed range)](docs/screenshots/editor_curve_centroid.png)

The **fixed range / autoscale** toggle on the right of the panel changes the y-axis. *Fixed range* uses sensible musical defaults (e.g. -40 to 0 dB for LUFS) so curves are comparable across songs. *Autoscale* zooms the y-axis to the song's own min/max — handy when you want to see fine detail in a song that has a narrow dynamic range.

![Curves panel — Spectral Centroid (autoscale)](docs/screenshots/editor_curve_autoscale.png)

A **yellow vertical line** on the curve tracks the playhead in real time as the song plays — so you can immediately see what the loudness, brightness, etc. is doing at the exact moment you're listening to.

##### Curves toggle (collapse + RMS gate)

![Curves panel collapsed](docs/screenshots/editor_curves_collapsed.png)

The little **▾ Curves** button at the left of the panel collapses everything below it. When collapsed, the per-stem RMS tint also disappears from the lanes — so the editor becomes a more traditional, less colourful "just the waveforms" view. Your choice persists across reloads.

##### Looping for practice

Press **I** to set a loop-in point at the cursor, **O** to set the loop-out, and **L** to toggle the loop on/off. The loop region is highlighted on the mix and survives reloads — server-side persistence means even if you close the tab and come back tomorrow, the loop is exactly where you left it. **Esc** clears the loop.

##### Click track

The **Click track: off / on** toggle in the transport plays a metronome lined up with the detected beats and downbeats — handy for confirming the analysis got the tempo right, or for practising along.

##### Zoom

The **Zoom slider** above the transport stretches the timeline horizontally up to 20×, so you can scrub down to individual onset markers, examine ramp shapes in detail, or read every section label without overlap.

##### Export *(v0.1d)*

![Export modal — After Effects format with FPS field](docs/screenshots/export_modal_aftereffects.png)

The **Export ▶** button on the right of the transport row opens a dialog that turns the current song into a file your downstream tool can read. There are two top-level choices:

- **Format** — the file type. Nine options today:
  - **CSV** / **JSON** — simple tabular or structured data, good for spreadsheets, Python, or custom tooling.
  - **MIDI** — a standard `.mid` file. Drum hits become General-MIDI drum notes; continuous curves become CC74. Imports into any DAW.
  - **After Effects** — an ExtendScript `.jsx` you run inside AE; it builds null layers with Slider Control keyframes and drops comp markers at every section boundary.
  - **TouchDesigner** — a CHOP-friendly CSV plus a separate events CSV for Table DAT triggers.
  - **OSC** — a JSON bundle plus a tiny Python playback script; broadcasts events live to whatever address you pick.
  - **Houdini** — CHOP-friendly CSV with a metadata header, a `time` channel, and one channel per track.
  - **disguise** — a cue list CSV in `HH:MM:SS:FF` timecode at your chosen frame rate.
  - **Unreal Sequencer** — JSON with event tracks and float curves with interp keys; drag into a Sequencer asset.

- **Grammar** — *what* events get emitted. The four built-in presets are tuned for different creative jobs:
  - **Concert visuals** — downbeat pulse, per-class drum tracks, drop labels, vocal phrase envelopes, section ramps, energy curve. Good default for music-reactive video.
  - **Character animation** — vocal/melody phrase envelopes, downbeat accents, per-stem energy. Good for lipsync-adjacent or full-body character work.
  - **Lighting** — fast-attack drum tracks, hihat with rarity decay, build-up cues, intensity curve. Tuned for stage-lighting timing.
  - **Camera edit** — section cuts as primary, downbeat bar markers, impact hits, slow energy curve. Tuned for video editing markers and cut points.

The dialog adapts to the format you pick:

- **After Effects** and **disguise** ask for **FPS** (so timecode and keyframes line up with your project rate).
- **MIDI** asks for **ticks/beat** (480, 960, or 1920 — higher means tighter timing resolution).
- **OSC** asks for **host** and **port** so the bundle is pre-addressed at your show network.

![Export modal — OSC format with host + port fields](docs/screenshots/export_modal_osc.png)

The file streams straight to your browser as a normal download — nothing is left on disk on the server side.

##### Frame rate / timecode *(v0.2a)*

Every export carries a **frame rate** that the cuesheet uses for animation timing — pick from the dropdown at the bottom of the dialog (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60). At 29.97 or 59.94 a **drop-frame** checkbox appears for broadcast/SMPTE workflows that need their timecode to track real-time clocks accurately.

![Export modal — FPS picker with drop-frame toggle at 29.97](docs/screenshots/export_modal_fps_2997_dropframe.png)

The chosen FPS is recorded on the cuesheet itself and on every event in `analysis.json` and `cuesheet.json`. CSV exports gain a `frame_number` column; After Effects, disguise, TouchDesigner, and Houdini exports write their timecodes at this rate. So if you change FPS in the dialog, every downstream tool sees the right frame numbers without any per-tool conversion.

CLI users get the same thing via `--fps` and `--drop-frame` flags on `musicue analyze`, `musicue compile`, and `musicue render`. Old `analysis.json` files (schema 1.1) keep working — they just lack the frame fields until you re-analyze.

> Audio export (reference mix, individual stems) and video export (timeline render) are planned for **v0.2d**. The CLI commands (`musicue export`, `musicue render`, `scripts/make_qc_video.py`) cover those today.

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
