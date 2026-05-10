# Follow-ups

Open items left after the benchmark + QC video work (commits `b761ba4`..`5139e42`).
Each entry has enough context to pick up cold.

## All-In-One spectrogram cache variance

**Symptom**: AIO's structure-detection stage swings between ~22 s and ~113 s
between adjacent benchmark runs on the same song with a persistent
`--cache-root`. See `MusicTests/out/benchmark_report.md` v2 R1 vs R2.

**Why this matters**: makes per-stage benchmark numbers noisy and inflates
`Total Layer 1 + compile` by ~90 s on the slow run. AIO is the second-largest
stage cost after CLAP.

**Where to look**:
- `musicue/analysis/structure.py:84` — `allin1.analyze(..., multiprocess=False)`
- AIO's own cache dir (it writes spectrograms outside our `cache_root`).
  Run `python -c "import allin1, inspect; print(inspect.getfile(allin1))"`
  and trace where it caches.
- The fast-run log line `Found 0 spectrograms already extracted, 1 to extract.`
  appears on *every* run, suggesting the cache check fails even when the
  spectrogram exists from the prior run.

**Possible causes**: cache key includes a path that varies (resolved vs.
relative, case, separators), or the cache is keyed on input mtime which
shifts between runs.

## Replace librosa's `audioread` fallback before librosa 1.0

**Symptom**: every m4a/mp3 load emits
`UserWarning: PySoundFile failed. Trying audioread instead.` plus
`FutureWarning: librosa.core.audio.__audioread_load deprecated as of librosa 0.10.0`.

**Why this matters**: librosa 1.0 will remove the audioread shim entirely.
Once that lands, every `librosa.load` on a compressed input will fail and
take down the curves + CLAP fallback paths added in `91d6e67` and `7d5feee`.

**Affected code**:
- `musicue/analysis/curves.py` — `_read_audio_2d`, `compute_rms_curve`,
  `compute_spectral_centroid_curve`, `compute_spectral_flux_curve`
- `musicue/analysis/clap_reranker.py` — `_load_full_audio_mono`
- `musicue/analysis/pipeline.py:102` — `librosa.get_duration` /
  `get_samplerate` fallback for `sf.info`

**Options**:
1. Direct ffmpeg subprocess to decode to a NumPy array (smallest dep
   surface; ffmpeg is already required for `make_qc_video.py`).
2. Use `pydub` (wraps ffmpeg, returns AudioSegments).
3. Pin libsndfile against an ffmpeg-enabled build so `sf.read` accepts
   compressed formats directly. Complicated cross-platform.

Option 1 is probably right — wrap ffmpeg in a small `_decode_to_array`
helper next to the existing audio shims and route everything through it.

## `musicue/listen.py` will crash on m4a inputs

**Symptom**: `listen.py:65` calls `sf.read(str(source_audio))` on
user-supplied audio. Same libsndfile-rejects-m4a bug that `91d6e67` and
`7d5feee` fixed elsewhere — just not on the benchmark/render path so it
wasn't caught.

**Reproducer**:
```
musicue listen cuesheet.json --audio "song.m4a"
```

**Fix**: route through the same shim used by the curves / CLAP fallback
once the librosa-1.0 follow-up above lands. Until then, the same
`try sf.read / except LibsndfileError: librosa.load` pattern works.

## v0.2c beat patterns require allin1 to be useful

**Symptom**: `populate_beat_pattern_fields` runs on every analyze (commit
`3db5e99`+), but `BeatEvent.bar` and `beat_in_bar` are only meaningfully
populated when `cfg.analysis.beat_backend == "allin1"`. With the librosa
fallback (used when allin1 isn't installed or fails), every beat gets
`bar = 0`, so phrase autocorrelation collapses to a single phrase covering
the whole song and `is_fill()` / `is_phrase_start()` degrade to no-ops.

**Why this matters**: the README's v0.2c section reads "every analyze gets
phrase / fill / syncopation fields", which is technically true (the fields
are present) but creatively misleading — librosa-fallback users get nothing
useful from them.

**Where to look**:
- `musicue/analysis/beats.py` — librosa path. Currently sets `bar = 0` for
  all beats; could synthesize bars by assuming 4/4 and grouping every 4
  consecutive beats into a bar.
- `musicue/analysis/patterns.py:_bar_count` — degrades silently when bars
  are all 0 (returns 1, no useful phrases).

**Options**:
1. Synthesize bars in the librosa path: assume 4/4, group beats by 4,
   set `bar = i // 4`, `beat_in_bar = (i % 4) + 1`, `is_downbeat = i % 4 == 0`.
   Wrong for 3/4 / 6/8 songs, but better than the current "all bar 0".
2. Add an `analysis.patterns_quality` field flagging "low — librosa
   fallback used, bars synthesized" so consumers can downweight pattern
   filters.
3. Document the limitation in README + raise a warning at analyze time.

Option 1 + Option 3 together is probably right.
