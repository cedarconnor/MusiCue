# MusiCue — Design Document

**Status:** v1.1 design, pre-implementation
**Author:** Cedar Connor
**Target hardware:** Single NVIDIA A6000 (48 GB), Windows 11, Python 3.11
**Repository:** `musicue` (suggested)

### Changelog

**v1.1**
- CLAP repositioned as a re-ranker / labeler over candidate events, not as a primary event generator. Timing-imprecise by design; only used to attach semantic labels to events found by tighter detectors. (§5.5, §4.1)
- Phrase layer added. Basic Pitch MIDI is grouped into musical phrases with energy / pitch contour / peak features, exposed as both an event source and a continuous track. (§5.3.1, §4.1)
- Event hierarchy added: every event carries a `timescale` of `macro` / `meso` / `micro`. Grammars weight by timescale. (§4.1, §6)
- Concrete `rarity_bonus` formula specified. (§6.2)
- Cache key now includes per-model version hashes. Prevents silent staleness across model upgrades. (§5.7)
- Drum classifier (kick/snare/hat/tom/cymbal CNN) pulled forward from M5 to M2. The v1 spectral heuristic is acknowledged as not good enough to ship. (§5.4, §13)
- Section transitions emit *both* a `step` track (hard cut) and a derived `ramp` track (gradual rise from spectral flux + LUFS). Grammars choose. (§4.2, §6.4)
- Multi-task head spec for the future moment classifier: `(cue_type, strength, timescale)` instead of binary. (§12.2)
- Music2Latent plan made realistic: expose latent as a continuous track + an `inspect --latent` correlator, then *manually* author per-grammar mappings as a v1.5 feature, with learned mapping deferred. (§12.3)
- New CLI command: `musicue diff` for comparing cuesheets across grammars / config changes. (§8.1)

**v1** — initial design.

---

## 1. Goal

Convert a song into a structured, typed, exportable timeline of musical events that can drive animation, lighting, camera, and edit decisions across DCC tools (After Effects, TouchDesigner, Unreal, Houdini, disguise, MIDI/DMX rigs).

Not real-time. Offline analysis only — this is the major design freedom that lets us use heavy bidirectional models without latency constraints.

The output is **not raw audio amplitude curves**. The output is a **cue sheet**: a list of typed events with timestamps, intensities, and per-target ADSR envelopes, plus a small set of continuous control curves where appropriate.

## 2. Non-Goals

- Real-time / live performance use.
- Audio generation, remixing, or stem manipulation.
- Music recommendation or playlist work.
- Replacing a human visuals director on a high-stakes production. The tool produces a strong first draft; the operator refines.
- Generating the visuals themselves. MusiCue produces the control signal; downstream tools render the result.

## 3. Architecture

Three strict layers with thin interfaces between them. Each layer must be runnable independently for debugging and partial re-runs.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: ANALYSIS                                          │
│                                                             │
│  song.wav ──► Demucs ──► stems/                             │
│            ──► All-In-One ──► beats, downbeats, sections    │
│            ──► Basic Pitch (per stem) ──► MIDI events       │
│                              │                              │
│                              └──► phrase grouping ──► phrases│
│            ──► librosa onsets (per stem) ──► transients     │
│                              │                              │
│            ┌─────────────────┴── candidate events ──┐       │
│            ▼                                        ▼       │
│         CLAP windowed re-rank ──► attach labels & scores    │
│            ──► loudness / spectral curves ──► continuous    │
│                                                             │
│  Output: analysis.json (the "fat" intermediate format)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: COMPILER                                          │
│                                                             │
│  analysis.json + grammar.yaml ──►                           │
│            event scoring                                    │
│            event filtering / promotion                      │
│            ADSR envelope generation per event type          │
│            continuous curve smoothing & quantization        │
│                                                             │
│  Output: cuesheet.json (the deliverable timeline)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: EXPORTERS                                         │
│                                                             │
│  cuesheet.json ──► .mid (general MIDI + CC)                 │
│                ──► .osc / OSC bundle                        │
│                ──► AE .jsx (markers + property keyframes)   │
│                ──► TouchDesigner .tox or CHOP CSV           │
│                ──► Unreal Sequencer .json                   │
│                ──► Houdini CHOP CSV                         │
│                ──► disguise / DMX cue list                  │
│                ──► generic CSV                              │
└─────────────────────────────────────────────────────────────┘
```

**Why three layers:**

Layer 1 is opinion-free analysis. It's expensive (GPU, minutes per song) and should run once per song and be cached.

Layer 2 is where taste lives. It's cheap, reruns in seconds, and is the layer most worth iterating on. Animation grammar presets are pure config.

Layer 3 is dumb adapters. Each exporter is small, targets one tool, and never reaches back into the analysis layer.

This separation also enables the eventual fine-tuning path (§12): you train models that improve Layer 1 outputs or Layer 2 scoring without touching the rest of the system.

## 4. Intermediate Formats

### 4.1 `analysis.json` (Layer 1 → Layer 2)

Fat, unopinionated, includes everything detectable. Versioned.

Every discrete event carries a `timescale` field (`micro` / `meso` / `macro`) used by Layer 2 grammars for hierarchical weighting. CLAP labels are *attached to candidate events* rather than emitted as standalone events — see §5.5.

```json
{
  "schema_version": "1.1",
  "source": {
    "path": "input/song.wav",
    "sha256": "...",
    "duration_sec": 213.456,
    "sample_rate": 44100
  },
  "analysis_config": {
    "demucs_model": "htdemucs_ft",
    "demucs_version": "4.0.1",
    "allin1_model": "harmonix-all",
    "allin1_version": "1.1.0",
    "clap_model": "music_audioset_epoch_15_esc_90.14.pt",
    "clap_version": "1.1.5",
    "basic_pitch_model": "icassp_2022",
    "basic_pitch_version": "0.4.0",
    "drum_classifier_version": "0.1.0"
  },
  "stems": {
    "drums": "stems/drums.wav",
    "bass": "stems/bass.wav",
    "vocals": "stems/vocals.wav",
    "other": "stems/other.wav"
  },
  "tempo": {
    "bpm_global": 124.0,
    "bpm_curve": [{"t": 0.0, "bpm": 124.0}, ...],
    "time_signature": [4, 4]
  },
  "beats": [
    {"t": 0.483, "beat_in_bar": 1, "bar": 1, "is_downbeat": true,  "confidence": 0.97, "timescale": "meso"},
    {"t": 0.967, "beat_in_bar": 2, "bar": 1, "is_downbeat": false, "confidence": 0.95, "timescale": "micro"},
    ...
  ],
  "sections": [
    {"start": 0.0,    "end": 17.2,  "label": "intro",    "confidence": 0.91, "timescale": "macro"},
    {"start": 17.2,   "end": 51.6,  "label": "verse",    "confidence": 0.88, "timescale": "macro"},
    {"start": 51.6,   "end": 86.0,  "label": "chorus",   "confidence": 0.93, "timescale": "macro"},
    ...
  ],
  "section_transitions": [
    {
      "t": 17.2, "from": "intro", "to": "verse",
      "ramp": {"t_start": 15.8, "t_end": 17.2, "shape": "ease_in"},
      "ramp_evidence": {"spectral_flux_rise": 0.74, "lufs_rise_db": 6.2}
    },
    ...
  ],
  "onsets": {
    "drums":  [
      {"t": 0.483, "strength": 0.92, "drum_class": "kick",  "drum_class_conf": 0.88, "timescale": "micro",
       "labels": [{"label": "punchy kick", "score": 0.61, "source": "clap"}]},
      ...
    ],
    "bass":   [{"t": 0.50, "strength": 0.71, "timescale": "micro", "labels": []}, ...],
    "vocals": [...],
    "other":  [...]
  },
  "midi": {
    "vocals": [{"t": 17.34, "duration": 0.42, "pitch": 64, "velocity": 88}, ...],
    "other":  [...]
  },
  "phrases": {
    "vocals": [
      {
        "t_start": 17.34, "t_end": 22.10, "timescale": "meso",
        "note_count": 12, "pitch_peak": 67, "pitch_low": 55,
        "pitch_contour": [60, 62, 64, 67, 65, 62, 60],
        "energy_curve": {"hop_sec": 0.04, "values": [...]},
        "labels": [{"label": "vocal melisma", "score": 0.72, "source": "clap"}]
      },
      ...
    ],
    "other": [...]
  },
  "curves": {
    "lufs":              {"hop_sec": 0.04, "values": [...]},
    "spectral_centroid": {"hop_sec": 0.04, "values": [...]},
    "spectral_flux":     {"hop_sec": 0.04, "values": [...]},
    "rms_drums":         {"hop_sec": 0.04, "values": [...]},
    "rms_bass":          {"hop_sec": 0.04, "values": [...]},
    "rms_vocals":        {"hop_sec": 0.04, "values": [...]},
    "rms_other":         {"hop_sec": 0.04, "values": [...]}
  }
}
```

**Notes on the schema:**

- `timescale` is set by the detector. Default mapping: section + section_transition = `macro`; downbeat + phrase + multi-bar build = `meso`; everything else = `micro`. Grammars can override via scoring.
- `labels` on each event is a list (possibly empty). Each label has `label`, `score`, `source` (`"clap"`, `"drum_classifier"`, future `"moment_classifier"`). This is how CLAP attaches semantic meaning to events found by tighter detectors.
- `section_transitions` is *derived*: it pairs each section boundary with a ramp window estimated from spectral flux + LUFS rise just before the boundary. Layer 2 grammars choose to emit a hard `step` cut, a soft `ramp`, or both.

### 4.2 `cuesheet.json` (Layer 2 → Layer 3)

Opinionated, filtered, ready to drive a target. Carries ADSR-shaped envelope intent.

```json
{
  "schema_version": "1.1",
  "source_sha256": "...",
  "grammar": "concert_visuals",
  "duration_sec": 213.456,
  "tempo_map": [...],
  "tracks": [
    {
      "name": "downbeat_pulse",
      "type": "impulse",
      "timescale": "meso",
      "events": [
        {"t": 0.483, "strength": 0.97, "envelope": {"a": 0.02, "d": 0.18, "s": 0.0, "r": 0.0}, "tags": ["downbeat", "bar:1"]},
        ...
      ]
    },
    {
      "name": "kick",
      "type": "impulse",
      "timescale": "micro",
      "events": [...]
    },
    {
      "name": "snare",
      "type": "impulse",
      "timescale": "micro",
      "events": [...]
    },
    {
      "name": "vocal_phrase",
      "type": "envelope",
      "timescale": "meso",
      "events": [
        {
          "t_start": 17.34, "t_end": 22.10, "strength": 0.84,
          "envelope": {"a": 0.30, "d": 0.20, "s": 0.7, "r": 0.50},
          "shape_curve": {"hop_sec": 0.04, "values": [...]},
          "tags": ["vocal_entry", "phrase:0"]
        }
      ]
    },
    {
      "name": "section_change",
      "type": "step",
      "timescale": "macro",
      "events": [
        {"t": 17.2,  "value": 1, "label": "verse"},
        {"t": 51.6,  "value": 2, "label": "chorus"},
        ...
      ]
    },
    {
      "name": "section_ramp",
      "type": "ramp",
      "timescale": "macro",
      "events": [
        {"t_start": 15.8, "t_end": 17.2, "from": 0.0, "to": 1.0, "shape": "ease_in_out", "label": "intro→verse"},
        ...
      ]
    },
    {
      "name": "energy",
      "type": "continuous",
      "timescale": "macro",
      "hop_sec": 0.04,
      "values": [...]
    }
  ]
}
```

Five event types — `impulse` (single time + ADSR), `envelope` (start/end + ADSR + optional shape curve), `step` (state change), `ramp` (interpolated transition), `continuous` (sampled curve). Every exporter understands all five primitives.

## 5. Layer 1: Analysis

### 5.1 Source separation — Demucs

- Model: `htdemucs_ft` (fine-tuned hybrid transformer Demucs).
- Library: `demucs` (Meta, MIT license).
- Output: 4 stems at 44.1 kHz stereo WAV (`drums`, `bass`, `vocals`, `other`).
- A6000 runtime: ~real-time × 3–5 (a 3-minute song separates in ~40–60 seconds).
- Cache stems on disk; downstream stages reuse them.

### 5.2 Beat / downbeat / structure — All-In-One

- Library: `allin1` (Kim et al., 2023, ISMIR best paper).
- Single model produces beats, downbeats, functional segment labels (`intro`, `verse`, `chorus`, `bridge`, `outro`, `inst`, `solo`), and tempo.
- Internally uses Demucs as preprocessing — pass our cached stems to avoid double work.
- Output: directly populates `tempo`, `beats`, `sections` in `analysis.json`.
- Fallback: if All-In-One fails on a difficult track (low percussion, ambient), fall back to `librosa.beat.beat_track` + librosa structural segmentation. `madmom` is not supported on Windows and has been dropped (§14, Q5). Configure via `analysis_config.beat_backend`.

### 5.3 Polyphonic transcription — Basic Pitch

- Library: `basic-pitch` (Spotify, Apache 2.0).
- Tiny model (~17 MB). CPU-fast, but use GPU for batch.
- Run on `vocals` and `other` stems (skip drums, skip bass — bass tracks transcribe poorly and we already get bass via onset detection).
- Output: per-stem MIDI-like events with onset, duration, pitch, velocity, optional pitch bend. Stored under `midi.<stem>`.

#### 5.3.1 Phrase grouping

Raw MIDI notes are immediately post-processed into **phrases** — the meso-scale musical unit that drives character animation, vocal-led camera moves, and lighting swells.

- **Segmentation rule:** consecutive notes within a stem belong to the same phrase if the inter-note gap is below `phrase_gap_sec` (default 0.6s for vocals, 0.4s for melodic instruments). Configurable per stem.
- **Per-phrase features:**
  - `t_start`, `t_end`, `note_count`
  - `pitch_peak`, `pitch_low`, downsampled `pitch_contour` (10 Hz)
  - `energy_curve` from stem RMS over the phrase span
  - `labels` attached by the CLAP re-ranker (e.g. `"vocal melisma"`, `"piano arpeggio"`)
- Output stored under `phrases.<stem>` in `analysis.json`.
- Phrases are also emitted as a continuous curve (`curves.phrase_active_<stem>`, 0/1 gate) for grammars that want a "vocals are happening" signal without per-phrase event handling.

### 5.4 Per-stem onset detection — librosa + drum classifier

- Run librosa's `onset.onset_detect` with `backtrack=True` on each stem at 22050 Hz.
- Score each onset using `onset_strength_multi` for spectral richness.
- Apply a per-stem peak-picking pre-filter: median window 0.05 s, delta 0.07, wait 0.03 s.

**Drum classification (kick / snare / hat / tom / cymbal):**

The v1 spectral heuristic was acknowledged in v1 of this doc as a known weak link. **Promoted to M2** as a small CNN, because the heuristic isn't good enough to ship — it confuses kicks with toms and snares with claps across genres.

- Architecture: 4-conv-block CNN over 64-band log-mel patches (50 ms window centered on each detected onset), 5-class head.
- Training data: ENST-Drums + MDB Drums + Slakh2100 isolated drum stems. ~50k labeled onsets.
- Output per onset: `drum_class` (one of `kick`, `snare`, `hat`, `tom`, `cymbal`, `other`) + `drum_class_conf`.
- Versioned via `analysis_config.drum_classifier_version` (cache key includes this).
- Training script: `scripts/train_drum_classifier.py`.
- Runtime: <1s per song on A6000.

### 5.5 Semantic labeling — CLAP (re-ranker, not generator)

CLAP is **not** a precise event detector. Audio-text contrastive models trained on 10-second clips are great at "what is this audio about" but bad at "exactly when does the drop hit." A CLAP window centered at t=86.0s scoring high on `"sub bass drop"` might be reacting to anything in the surrounding ±1 second.

So CLAP is used **only to attach semantic labels to events found by tighter detectors**.

**Pipeline:**

1. Collect candidate events from precise sources: drum onsets, vocal onsets, melodic onsets, section boundaries, phrase starts.
2. For each candidate event, extract a 2.0 s audio window centered on the event time (clamped to song bounds).
3. Score each window against the **prompt bank** from `grammar.yaml` using CLAP cosine similarity.
4. Attach top-k labels above threshold (default top-3, score > 0.55) to the event's `labels` field.

**Default prompt bank:**
`"sub bass drop"`, `"vocal stab"`, `"guitar stab"`, `"cymbal swell"`, `"vocal melisma"`, `"snare roll"`, `"riser"`, `"impact hit"`, `"breakdown"`, `"build up"`, `"silence"`, `"808 slide"`, `"orchestral hit"`, `"reverse cymbal"`, `"piano arpeggio"`, `"punchy kick"`, `"deep kick"`.

**Library:** `laion-clap` (LAION, CC0/MIT).

**What this gives Layer 2:** when the compiler sees an onset at t=86.0 with `labels: [{"label": "sub bass drop", "score": 0.81}]`, it has both the *exact timing* (from onset detection) and the *semantic type* (from CLAP). Grammars filter on label, weight by score, and use the precise onset time. This is the right division of labor.

**What this does not do:** invent events that no other detector found. If you want a track of "every moment that sounds like a riser," that requires a dedicated detector — likely the future moment classifier (§12.2), not CLAP.

### 5.6 Continuous curves

Sampled at 25 Hz (40 ms hop) by default — close enough to film-frame rate (24/30 fps) and to Music2Latent's ~10 Hz native rate that downstream resampling is trivial. Configurable per grammar.

- **LUFS short-term**: ITU-R BS.1770 momentary loudness via `pyloudnorm`.
- **Spectral centroid, spectral flux, RMS per stem**: librosa.
- **Stereo width and pan**: derived from L/R channel correlation and energy balance. Useful for spatial visuals — especially relevant for Sphere-style immersive work.
- **Phrase activity gates**: 0/1 per stem, derived from §5.3.1.
- **Optional Music2Latent embedding curve**: if `analysis_config.music2latent: true`, also store the 64-dim latent sequence (~10 Hz). See §12.2 for the realistic plan.

### 5.7 Caching

`analysis.json` and stems are keyed by:

```
cache_key = sha256(
    audio_bytes
  + canonical_json(analysis_config)
  + model_versions_blob
)
```

where `model_versions_blob` concatenates the version strings of every model used in the run (Demucs, All-In-One, Basic Pitch, CLAP, drum classifier, etc.). This is critical: bumping CLAP weights or All-In-One without invalidating the cache would silently serve stale `analysis.json` files and confuse all downstream debugging.

Each model wrapper is responsible for reporting its own version string (model checkpoint hash where possible, package version as fallback). The `analysis_config` block in `analysis.json` records all version strings so any cached file is fully self-describing.

Re-running with the same song and same config + versions hits cache instantly. Layer 1 is the expensive layer — caching is mandatory, not optional.

## 6. Layer 2: Compiler

The compiler consumes `analysis.json` + a **grammar** definition and produces `cuesheet.json`.

### 6.1 Animation grammars

A grammar is a YAML file describing:

- Which event types to emit as tracks.
- How to score and threshold each event.
- ADSR envelope shape per event type.
- Curve sources and smoothing for continuous tracks.
- Optional CLAP label routing.

Ship four built-in grammars in `grammars/`:

**`concert_visuals.yaml`** — strong downbeats, chorus hits, drops, builds. Vocal phrases sustained. Section changes drive lighting state. Optimized for the kind of work Cedar does at Sphere.

**`character_animation.yaml`** — vocal phrasing dominant, melody accents, breath gaps. Drums de-emphasized. Slower envelopes, more sustain.

**`lighting.yaml`** — kick / snare / downbeat focus. Hi-hat as 16th-note shimmer track. Builds and drops as macro brightness moves.

**`camera_edit.yaml`** — section changes and bar starts as cut points. Major accents only. Long ADSR releases for camera moves.

Grammar example (excerpt):

```yaml
name: concert_visuals

# Hierarchy weighting — applied as a multiplier on top of per-track scores.
# Lets a single setting dial overall feel from "macro-driven cinematic" to "micro-driven busy".
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

  # Drops are now sourced from labeled onsets (CLAP attached the label upstream),
  # not from a standalone CLAP event stream. Timing comes from the onset.
  - name: drop
    type: impulse
    source: onsets.*
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
    shape_curve_from: "energy_curve"   # exporter can render this as a sampled shape

  # Hard cuts at section boundaries — for edits, lighting state changes.
  - name: section_change
    type: step
    source: sections
    emit: section_boundary

  # Soft ramps derived from spectral_flux + LUFS rise — for camera moves, brightness sweeps.
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

### 6.2 Event scoring

Per-event score:

```
score = base_score
      × Π multiplier_factors_where_condition_true
      × stem_weight
      × hierarchy_weight[event.timescale]
      × rarity_bonus
      − cooldown_penalty
```

**`hierarchy_weight`** is a grammar-level dict (see §6.1) keyed by `macro` / `meso` / `micro`. Tuning these three numbers is the single highest-leverage knob in the system — a `concert_visuals` grammar with `{macro: 1.5, meso: 1.2, micro: 0.8}` feels cinematic; flipping to `{macro: 0.7, meso: 1.0, micro: 1.4}` feels busy and percussive. Same analysis, completely different output.

**`rarity_bonus`** rewards events that haven't fired recently in the same track. Concrete formula:

```
rarity_bonus = exp(-recent_event_count / decay)
```

where `recent_event_count` counts events from the same track within the last `window_sec` seconds before the candidate event. Configurable per track:

```yaml
rarity:
  window_sec: 1.0   # look-back window
  decay: 4.0        # higher decay = milder rarity penalty
```

With defaults (`window_sec: 1.0`, `decay: 4.0`): no recent events → bonus = 1.0; one recent event → 0.78; four recent → 0.37. Prevents 16th-note kick patterns from saturating downstream visuals while still letting individual hits through.

**`cooldown_penalty`** is a hard floor — events within `cooldown_sec` of a previously emitted event from the same track are dropped entirely (not just dampened). Use for tracks where you want at most one event per N seconds (drops, section markers).

**`stem_weight`** lets a grammar say "I trust the vocals stem more than the other stem for melodic onsets." Default 1.0 for all stems.

### 6.3 ADSR envelope generation

Each impulse or envelope event carries an ADSR. The compiler doesn't bake the curve into samples — it stores the shape parameters. **The exporter decides** whether to render the curve as keyframes (AE), as MIDI velocity + CC (general MIDI), as a CHOP shape (TouchDesigner), or as raw cue triggers (DMX). This matters: rendering ADSR to dense samples at compile time would 100× the file size and lock us into one frame rate.

ADSR semantics:
- `a` (attack): seconds from 0 to peak.
- `d` (decay): seconds from peak to sustain.
- `s` (sustain): 0.0–1.0 sustain level. For impulse events, `s=0` and `r=0` (pure pluck).
- `r` (release): seconds from sustain release to 0.

### 6.4 Continuous curves

- Resample to a common rate (default 25 Hz; configurable per grammar).
- Optional EMA smoothing (`tau_sec` parameter).
- Optional normalization (min/max, percentile, or z-score).
- Optional quantization to N steps (useful for stepped lighting cues).

### 6.5 Ramp tracks

Ramps sit between `step` and `continuous`: they describe a *named transition* between two values over a finite window, with a shape function. The compiler emits ramps from `analysis.section_transitions` (and other ramp-shaped sources). Exporters render them however makes sense — AE renders two keyframes with eased interpolation; MIDI renders a CC sweep; DMX renders a fade cue.

Ramp shapes: `linear`, `ease_in`, `ease_out`, `ease_in_out`, `s_curve`, `exp_in`, `exp_out`. Defined as parametric functions, not sampled — same principle as ADSR (§6.3).

### 6.6 Phrase tracks (envelope type)

Phrases from `analysis.phrases.<stem>` map naturally to `envelope` tracks. Two emission modes:

- **Static ADSR mode** (default): each phrase emits an `envelope` event with a fixed ADSR per the grammar.
- **Shape-curve mode** (`shape_curve_from: "energy_curve"`): the phrase's actual energy curve is attached to the event. Exporters that support arbitrary shapes (TouchDesigner CHOP, AE Slider keyframes) render the real shape; exporters that only do ADSR fall back to fitting an approximate ADSR to the curve.

## 7. Layer 3: Exporters

Each exporter is a single Python module under `musicue/exporters/<name>.py` with one function:

```python
def export(cuesheet: CueSheet, out_path: Path, **opts) -> None: ...
```

### 7.1 MIDI (`midi.py`)

- Library: `mido`.
- Each impulse track → MIDI channel + drum-map note. ADSR rendered as note-on velocity + CC74 envelope follower.
- Each continuous track → CC channel.
- Each step track → MIDI program change or text marker.
- Output: standard `.mid` importable into Ableton, Logic, Reaper, Pro Tools.

### 7.2 OSC (`osc.py`)

- Library: `python-osc`.
- Emit a `.json` bundle file with timestamped OSC messages, plus a tiny player script that streams them at playback time.
- Address pattern: `/musicue/<track_name>` with float arg.

### 7.3 After Effects (`aftereffects.py`)

- Generates `.jsx` ExtendScript file.
- Composition layer markers for impulse/step events (visible in timeline).
- Slider Control properties pre-keyframed for continuous tracks (one Slider per track, samples at composition frame rate).
- Pickwhip-friendly naming: `MusiCue_kick`, `MusiCue_energy`, etc.

### 7.4 TouchDesigner (`touchdesigner.py`)

- Two output options:
  - CHOP CSV (one channel per track, sampled to common rate, plus a separate `events.csv` for impulse triggers).
  - `.tox` component containing a Table DAT of events + a CHOP network of pre-built channels. Generated via TD's command-line tox export if available, otherwise a `.json` that the user drops into a provided MusiCue.tox loader.

### 7.5 Unreal Sequencer (`unreal.py`)

- Emit `.json` in the format consumed by a companion Unreal plugin (`MusiCueImporter`, separate repo, out of scope for v1 backend but the JSON shape is designed for it).
- Maps to Sequencer Event Tracks (impulse/step) and Float Tracks (continuous).

### 7.6 Houdini CHOP (`houdini.py`)

- CSV in CHOP-compatible format.
- Plus a `.hda` digital asset stub (later).

### 7.7 disguise / DMX (`disguise.py`)

- Cue list CSV with timecode column compatible with disguise track import.
- ADSR rendered to fixture intensity curves at user-specified frame rate.

### 7.8 Generic (`csv.py`, `json.py`)

- Pretty-printed JSON (the cuesheet itself with optional rendered envelopes).
- Wide CSV (one column per track, common time grid).

## 8. CLI & Python API

### 8.1 CLI

```
musicue analyze SONG.wav [--config config.yaml] [--out runs/<name>/]
musicue compile  runs/<name>/analysis.json --grammar concert_visuals [--out cuesheet.json]
musicue export   cuesheet.json --target after_effects --out song.jsx
musicue render   SONG.wav --grammar concert_visuals --target midi --out song.mid
```

`render` is a convenience that runs analyze → compile → export in one shot using cached intermediates.

Plus inspection:

```
musicue inspect runs/<name>/analysis.json   # pretty summary
musicue plot    runs/<name>/analysis.json   # matplotlib timeline preview
musicue listen  cuesheet.json --click       # render audio with metronome + accent clicks for QC
musicue diff    cuesheet_a.json cuesheet_b.json   # compare two cuesheets
```

The `listen` command is non-negotiable for QC — playing the original song with synthesized clicks on every detected event is the only way to actually evaluate detection quality. Use distinct click sounds per track type and pan them stereo (kick center, snare slightly right, hats hard right, downbeat ping center, vocal entry left) so a busy mix stays legible.

The `diff` command is the iteration loop for grammar tuning. It compares two cuesheets (typically same song, different grammar or different config) and reports per-track event count deltas, timing drift on shared events, and added/removed tracks. Output is both terminal-readable and a JSON report. Combined with `listen`, this is how you A/B grammars without listening to ten minutes of clicks per attempt.

For Music2Latent inspection (see §12.3):

```
musicue inspect --latent runs/<name>/analysis.json   # correlate latent dims with audio features
```

### 8.2 Python API

```python
from musicue import analyze, compile, export

analysis = analyze("song.wav", config="default")
cuesheet = compile(analysis, grammar="concert_visuals")
export(cuesheet, target="after_effects", out_path="song.jsx")
```

Each step returns a dataclass; intermediate files are optional but enabled by default for caching.

## 9. Repository Structure

```
musicue/
├── pyproject.toml
├── README.md
├── DESIGN.md                  # this document
├── musicue/
│   ├── __init__.py
│   ├── cli.py                 # Typer-based CLI
│   ├── config.py              # pydantic config models
│   ├── schemas.py             # pydantic models for analysis.json + cuesheet.json
│   ├── cache.py               # sha256-keyed disk cache
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── pipeline.py        # orchestrates the full Layer 1 run
│   │   ├── separation.py      # Demucs wrapper
│   │   ├── structure.py       # All-In-One wrapper, librosa beat fallback
│   │   ├── transcription.py   # Basic Pitch wrapper
│   │   ├── phrases.py         # MIDI → phrase grouping (§5.3.1)
│   │   ├── onsets.py          # librosa onsets
│   │   ├── drum_classifier.py # CNN: kick/snare/hat/tom/cymbal (M2)
│   │   ├── clap_reranker.py   # CLAP attaching labels to candidate events (§5.5)
│   │   ├── curves.py          # LUFS, spectral, RMS, stereo width/pan
│   │   └── transitions.py     # section_transitions ramp derivation
│   ├── compile/
│   │   ├── __init__.py
│   │   ├── compiler.py
│   │   ├── grammar.py         # YAML loader + validation
│   │   ├── scoring.py         # event scoring DSL + rarity + hierarchy
│   │   └── envelopes.py       # ADSR + ramp utilities
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── midi.py
│   │   ├── osc.py
│   │   ├── aftereffects.py
│   │   ├── touchdesigner.py
│   │   ├── unreal.py
│   │   ├── houdini.py
│   │   ├── disguise.py
│   │   ├── csv.py
│   │   └── json_export.py
│   ├── inspect.py             # plot + summarize + --latent
│   ├── diff.py                # cuesheet diff (§8.1)
│   └── listen.py              # QC click-track renderer (stereo-placed)
├── grammars/
│   ├── concert_visuals.yaml
│   ├── character_animation.yaml
│   ├── lighting.yaml
│   └── camera_edit.yaml
├── prompt_banks/
│   └── default_clap_prompts.yaml
├── tests/
│   ├── fixtures/              # short royalty-free audio clips
│   ├── test_analysis.py
│   ├── test_compile.py
│   ├── test_exporters.py
│   └── test_grammar_dsl.py
├── scripts/
│   ├── benchmark.py
│   ├── make_qc_video.py       # waveform + event overlay video, ffmpeg
│   └── setup_env.ps1          # one-shot Windows env setup (uv venv + torch CUDA wheel install)
└── runs/                      # gitignored, per-song output
```

## 10. Dependencies

**Core (required):**
- `python>=3.11` (install via [python.org](https://www.python.org/downloads/windows/) or `winget install Python.Python.3.11`)
- `torch>=2.2` with CUDA 12.x — install via the CUDA wheel index:
  ```
  uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- `demucs>=4.0`
- `allin1`
- `basic-pitch`
- `librosa>=0.10`
- `pyloudnorm`
- `numpy`, `scipy`
- `pydantic>=2.0`
- `typer`
- `pyyaml`
- `soundfile`

**Zero-shot labeling:**
- `laion-clap`

**Exporters:**
- `mido` (MIDI)
- `python-osc` (OSC)

**Optional:**
- `music2latent` (v2 embeddings)

**Not supported on Windows:**
- `madmom` — requires Cython compilation that is consistently broken on Windows + Python 3.11. **Do not ship as a fallback on Windows.** If All-In-One fails on a difficult track, fall back to librosa's beat tracker only (`librosa.beat.beat_track`). See §14, Q5.

**Dev:**
- `pytest`, `ruff`, `pyright`
- `matplotlib` (inspect + plot)
- `ffmpeg` — install via `winget install Gyan.FFmpeg` or `scoop install ffmpeg`; ensure `ffmpeg.exe` is on `PATH`

**Package manager:** Use `uv` with an explicit lockfile (`uv.lock`). `allin1` has known dependency conflicts; `uv`'s resolver handles these better than pip. On Windows, always use a virtual environment (`uv venv`) — do not install into the system Python.

Pin everything in `pyproject.toml`.

## 11. Hardware & Runtime Notes

Target: single A6000 (48 GB VRAM, plenty of headroom) on Windows 11.

**Windows CUDA setup:** Verify the full chain before first run:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Required: NVIDIA driver ≥ 550, CUDA Toolkit 12.4, torch built against `cu124`. Install driver from NVIDIA's site; the CUDA toolkit ships bundled with the PyTorch wheel and does not require a separate system-level CUDA install.

**Path handling:** All file paths in the codebase must use `pathlib.Path` — never string concatenation with `/` or `os.sep`. This is the only reliable way to handle both Windows absolute paths (`D:\runs\song\`) and the relative paths used in tests.

**Shell scripts:** Utility scripts under `scripts/` are PowerShell (`.ps1`). The CLI itself (`musicue` command) is cross-platform via Typer and works in PowerShell, CMD, and Windows Terminal.

Approximate per-song runtime for a 3-minute track:

| Stage                       | Time   | VRAM peak |
|-----------------------------|--------|-----------|
| Demucs separation           | 45 s   | 4 GB      |
| All-In-One                  | 20 s   | 3 GB      |
| Basic Pitch (×2) + phrases  | 12 s   | 1 GB      |
| librosa onsets              | 8 s    | CPU       |
| Drum classifier CNN         | <1 s   | <1 GB     |
| CLAP re-rank (per-event)    | 15 s   | 5 GB      |
| Curves + transitions        | 6 s    | CPU       |
| **Total Layer 1**           | ~2 min | ~5 GB     |
| Layer 2 compile             | <2 s   | CPU       |
| Layer 3 export              | <1 s   | CPU       |

CLAP cost dropped vs. v1 of this doc (was 25s sliding window, now 15s for ~200 candidate events at batch size 32). The savings will grow as we tune candidate event filtering.

Plenty of room to batch multiple songs in parallel on the A6000 (limit by VRAM of the heaviest stage, ~5 GB, so 4–6 songs in flight is realistic).

Layer 1 is GPU-bound but **trivially parallelizable across songs** — the analysis pipeline ships with a `--batch` mode that processes a directory of audio files concurrently using a thread pool over CUDA streams. The cache key (sha256 + config + model versions, see §5.7) makes this safe.

## 12. Future Work / Fine-tuning Path

The drum classifier (§5.4) is now in M2, not future work — the spectral heuristic isn't usable. The remaining ML upgrade paths below are genuinely optional and don't block any v1 capability.

### 12.1 "Production-worthy moment" classifier — multi-task

This is the high-value model. Frame the problem as: *given audio context (Layer 1 features + MERT embedding), predict whether a human concert-visuals director would mark this moment as a cue, what kind, how strong, and at what timescale.*

**Bootstrap dataset (already exists):**
- The Kenny Chesney / Sphere cue breakdown work is exactly this kind of label set. Per-song unique look TRTs are timestamped human decisions about visual importance.
- Other Sphere shows where Cedar has access to cue lists are equally usable.
- Label format: `(song_path, t_seconds, cue_type, cue_strength, cue_timescale)`.

**Backbone:**
- MERT-95M as frozen audio encoder (HuBERT-style, 13 transformer layers, 75 Hz frame rate).
- Or CLAP audio encoder if semantic prompting matters.

**Multi-task head** (on top of a shared 4-layer transformer over 30s windows of MERT features, predictions every 40 ms):

| Head | Output | Loss |
|------|--------|------|
| `is_cue` | binary | BCE |
| `cue_type` | softmax over `{impact, transition, sustain, texture}` | cross-entropy, masked to cue frames |
| `cue_strength` | scalar in [0, 1] | MSE, masked to cue frames |
| `cue_timescale` | softmax over `{micro, meso, macro}` | cross-entropy, masked to cue frames |

This aligns the model's outputs directly with the rest of the system: `cue_type` maps to track families, `cue_strength` feeds `base_score`, `cue_timescale` feeds the hierarchy weighting from §6.2. No glue code needed downstream.

**Integration:**
- Output becomes a new event source in `analysis.json`: `cue_proposals: [{t, type, strength, timescale, score}]`.
- Grammars opt in via `source: cue_proposals`. Existing grammars keep working unchanged.

This upgrade path is **the actual moat** for the project. Off-the-shelf MIR tools find beats; this would learn what *makes a cue land*. Nobody else has the labeled data.

### 12.2 Music2Latent for continuous control

Music2Latent encodes 44.1 kHz audio into a ~10 Hz, 64-dimensional latent sequence. That's already at animation control rate, and the latent appears (anecdotally) to capture musical "vibe" features rather than raw signal.

**Realistic plan, not the hype version:**

- **v1.1 (low risk):** expose the latent sequence as a continuous track family (`curves.m2l_0` … `curves.m2l_63`). No interpretation, no opinion. Grammars can pipe specific dimensions to specific control parameters if the user has done the legwork to find a useful one.
- **v1.2:** ship `musicue inspect --latent`, a tool that renders a side-by-side plot of all 64 latent dimensions against the audio waveform + spectrogram + detected events for a known song. The user identifies "dim 12 ramps with chorus arrival on this song" by eye, then writes a grammar that uses dim 12.
- **v2.0 (deferred):** train a small probe network to map the 64-dim latent onto a labeled set of musical features (`brightness`, `density`, `tension`, `arousal`). This requires a labeled dataset — could be derived from the same Sphere cue data, or from existing music-emotion datasets like DEAM.

Refusing to overclaim here: "dim 12 = hue" without inspection is wishful thinking. The latents are useful, but the mapping is empirical and per-model.

## 13. Milestones

**M0 — Walking skeleton (1 week)**
- Repo scaffolding, pyproject, CLI shell, schemas (v1.1), cache with model-version-aware keying.
- Demucs wrapper + librosa onsets only.
- Compiler with one hardcoded grammar.
- JSON + CSV exporters only.
- End-to-end `musicue render song.wav --target csv` works.

**M1 — Full Layer 1 (1.5 weeks)**
- All-In-One integration.
- Basic Pitch integration + phrase grouping (§5.3.1).
- CLAP **as re-ranker** attaching labels to candidate events (§5.5).
- Continuous curves including stereo width / pan.
- Section transition ramp derivation.
- `inspect` and `plot` commands.

**M2 — Compiler + grammars + drum classifier (1.5 weeks)**
- YAML grammar DSL with scoring expressions, hierarchy weights, concrete rarity formula.
- All four built-in grammars.
- ADSR envelope system.
- Step + ramp track types.
- **Drum classifier CNN** (kick/snare/hat/tom/cymbal). Pulled forward from M5 because the spectral heuristic isn't shippable.
- `listen` QC command with per-track stereo-placed clicks.
- `diff` command.

**M3 — Exporters round 1 (1 week)**
- MIDI.
- After Effects `.jsx`.
- TouchDesigner CSV + simple `.tox` loader.
- OSC bundle.

**M4 — Exporters round 2 + polish (1 week)**
- Houdini, disguise, Unreal JSON.
- Batch mode.
- Documentation, examples, sample songs (royalty-free).
- Benchmark suite.

**M5 — Music2Latent inspection (open-ended, optional)**
- Expose latent as continuous tracks.
- `inspect --latent` correlator tool.

**M6+ — Production-worthy moment classifier (open-ended)**
- Multi-task head on MERT features, trained on Sphere cue data.
- Drops in as a new event source for grammars.

## 14. Open Questions

1. **All-In-One vs. BeatNet+** — All-In-One is the strongest single-model option, but BeatNet+ is more robust on non-percussive material. Decision: ship All-In-One as default, BeatNet+ as a configurable fallback. Revisit after M1 with real test material. Multi-hypothesis ensemble reconciliation deferred until we have real failure data.

2. **Unreal Sequencer plugin** — the JSON shape is designed but the Unreal-side importer is a separate project. Out of scope for v1 backend unless Cedar wants to scope it in.

3. **Tempo changes** — All-In-One assumes a roughly constant tempo. Songs with significant rubato or tempo modulation (e.g., live recordings, classical) need either madmom DBN tracking or a dedicated variable-tempo path. Defer to v2.

4. **Anticipation events** — generating "lead" events 0.2–0.5s ahead of strong impulses for visuals that should arrive *with* the beat rather than after it. This is naturally a Layer 2 grammar transformation, not a Layer 1 detector. Spec out as a built-in grammar primitive (`anticipate: {lead_sec: 0.3, threshold: 0.7}`) once the base system is working.

5. **Madmom dependency burden** — does not build on Windows + Python 3.11 (Cython compilation fails). **Decision made: drop madmom entirely.** The beat fallback is `librosa.beat.beat_track` (pure Python, no compilation). All-In-One is the primary; librosa beat tracker is the fallback. This simplifies the dependency tree and removes the only non-Windows-compatible hard dependency.

6. **Licensing of CLAP prompt banks** — the default prompts are descriptive English. No issue. But for fine-tuned per-genre prompts, document that prompt banks are user-contributed config.

7. **Acceleration via ONNX / TensorRT** — Demucs and CLAP both have ONNX paths. Worth it on the A6000? Probably no — runtime is already comfortable. Revisit only if batch throughput becomes a bottleneck.

8. **Sample rate for continuous curves** — currently 25 Hz default. Consider exposing per-grammar override and shipping 24 Hz (film-native) and 30/60 Hz (game-native) presets.

## 15. Success Criteria

A successful v1 means:

- A 3-minute song goes from .wav to a usable AE / TouchDesigner / MIDI cue file in under 3 minutes on the A6000.
- The QC click-track playback (`musicue listen --click`) is judged "tight enough to mix to" by a working VFX/lighting professional on at least 8 out of 10 test songs spanning rock, EDM, hip-hop, and acoustic material.
- A new exporter target can be added in under a day by writing one ~200-line module.
- A new animation grammar can be authored and tested in under an hour without touching Python code.
- The system is fully reproducible: re-running on the same audio with the same config produces byte-identical `analysis.json` and `cuesheet.json`.
