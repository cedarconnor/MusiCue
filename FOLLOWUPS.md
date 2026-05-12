# Follow-ups

Known issues, technical debt, and limitations that aren't worth fixing right
now but should be tracked. Each entry is self-contained — pick one up cold
without needing prior conversation context.

## madmom won't build from sdist on Windows without VS Build Tools

**Symptom**: `install.bat` reports `[WARN] madmom : madmom build failed.`
The readiness chip shows `MIS allin1` with the detail
`allin1 not importable: No module named 'madmom'.`

**Why this matters**: All-In-One beat/downbeat/section detection depends on
madmom for its onset prefilter. When madmom is missing, allin1 fails to
import entirely, and analysis silently falls back to librosa beat tracking
with no section detection. Pattern-aware fields (`phrase_id`, `is_fill`,
`syncopation`) degrade because the librosa path leaves `bar = 0` on every
beat (see the related follow-up below).

**Why it's hard**: madmom has no Python 3.11 wheels on PyPI — only an sdist.
The build requires a C compiler (MSVC), Cython, and numpy headers. madmom's
setup.py also imports `mido` at setup-time, which complicates fresh-venv
builds. The author's last release (0.16.1) predates Cython 3.0 and numpy
2.0.

**Workarounds available today**:
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   ("Desktop development with C++" workload), then re-run `install.bat`.
   The build then succeeds because MSVC is on PATH.
2. Use a conda environment instead of a venv — `conda-forge` has a prebuilt
   madmom wheel for Windows.
3. Skip allin1 entirely; the librosa fallback covers basic beat tracking.

**Long-term fix**: host a prebuilt madmom wheel in our own GitHub Releases
(or vendor one in `vendor/`) and install from that URL when the from-sdist
build fails. About a day of work to set up; would let `install.bat` succeed
cleanly without VS Build Tools.

## `.env` file is created but never actually loaded

**Symptom**: `install.bat` writes `.env` from `.env.example` declaring
`MUSICUE_STORAGE_ROOT`, `MUSICUE_HOST`, `MUSICUE_PORT`, and
`MUSICUE_FFMPEG_PATH`. None of those variables are read anywhere in
`musicue/`. `grep -rn MUSICUE_STORAGE_ROOT musicue/` returns zero hits.

**Why this matters**: a user editing `.env` to point storage at, say,
`D:\Music\analysis` will silently get the hardcoded default
(`~/.musicue/`) instead. The `.env.example` comments lie.

**Where to fix**:
- `musicue/ui/server.py` — `create_app(storage_root=...)` defaults to
  `Path.home() / ".musicue"`. Should read `MUSICUE_STORAGE_ROOT` first.
- `run.bat` — currently hardcodes `--host 127.0.0.1 --port 8000`. Should
  source `MUSICUE_HOST` / `MUSICUE_PORT` from `.env` (use a small Python
  preflight or PowerShell to parse the env file).

**Options**:
1. Use `python-dotenv` (already a transitive dep) in `create_app` and
   `run.bat`'s preflight. Cleanest.
2. Drop `.env.example` and document the env vars in the README instead.
   Less infrastructure, but loses the discoverability of a local config file.

## Replace librosa's `audioread` fallback before librosa 1.0

**Symptom**: every m4a/mp3 load emits
`UserWarning: PySoundFile failed. Trying audioread instead.` plus
`FutureWarning: librosa.core.audio.__audioread_load deprecated as of librosa 0.10.0`.

**Why this matters**: librosa 1.0 will remove the audioread shim entirely.
Once that lands, every `librosa.load` on a compressed input will fail and
take down the curves + CLAP fallback paths.

**Affected code** (line numbers may have drifted; grep for the calls):
- `musicue/analysis/curves.py` — `_read_audio_2d`, `compute_rms_curve`,
  `compute_spectral_centroid_curve`, `compute_spectral_flux_curve`
- `musicue/analysis/clap_reranker.py` — `_load_full_audio_mono`
- `musicue/analysis/pipeline.py` — `librosa.get_duration` /
  `librosa.get_samplerate` fallback for `sf.info`
- `musicue/listen.py` — already uses the try/except shim, but would benefit
  from sharing the same helper

**Options**:
1. Direct ffmpeg subprocess to decode to a NumPy array (smallest dep
   surface; ffmpeg is now bundled by the installer).
2. Use `pydub` (wraps ffmpeg, returns AudioSegments).
3. Pin libsndfile against an ffmpeg-enabled build so `sf.read` accepts
   compressed formats directly. Complicated cross-platform.

Option 1 is probably right — wrap ffmpeg in a small `_decode_to_array`
helper next to the existing audio shims and route everything through it.

## All-In-One spectrogram cache variance

**Symptom**: AIO's structure-detection stage swings between ~22 s and ~113 s
between adjacent benchmark runs on the same song with a persistent
`--cache-root`. See `MusicTests/out/benchmark_report.md` v2 R1 vs R2.

**Why this matters**: makes per-stage benchmark numbers noisy and inflates
`Total Layer 1 + compile` by ~90 s on the slow run. Only observable once
allin1 is actually installed (see the madmom follow-up above).

**Where to look**:
- `musicue/analysis/structure.py` — `allin1.analyze(..., multiprocess=False)`
- AIO's own cache dir (it writes spectrograms outside our `cache_root`).
  Run `python -c "import allin1, inspect; print(inspect.getfile(allin1))"`
  and trace where it caches.
- The fast-run log line `Found 0 spectrograms already extracted, 1 to extract.`
  appears on *every* run, suggesting the cache check fails even when the
  spectrogram exists from the prior run.

**Possible causes**: cache key includes a path that varies (resolved vs.
relative, case, separators), or the cache is keyed on input mtime which
shifts between runs.

## v0.2c beat patterns require allin1 to be useful

**Symptom**: `populate_beat_pattern_fields` runs on every analyze, but
`BeatEvent.bar` and `beat_in_bar` are only meaningfully populated when
`cfg.analysis.beat_backend == "allin1"`. With the librosa fallback (used
when allin1 isn't installed or fails — which is the default on Windows
until the madmom follow-up is resolved), every beat gets `bar = 0`, so
phrase autocorrelation collapses to a single phrase covering the whole
song and `is_fill()` / `is_phrase_start()` degrade to no-ops.

**Why this matters**: the README's v0.2c section reads "every analyze gets
phrase / fill / syncopation fields", which is technically true (the fields
are present) but creatively misleading — librosa-fallback users get nothing
useful from them.

**Where to look**:
- `musicue/analysis/beats.py` — librosa path. Currently sets `bar = 0` for
  all beats; could synthesize bars by assuming 4/4 and grouping every 4
  consecutive beats into a bar.
- `musicue/analysis/patterns.py` — `_bar_count` degrades silently when bars
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

## No frontend test framework

**Symptom**: `musicue/ui/web/` has no `vitest`, `@testing-library/react`,
`playwright`, or any other test runner. Components like `ReadinessChip`
and `ExportModal` rely on manual visual verification via the dev server +
Playwright screenshots.

**Why this matters**: regressions in client-side logic (the
`parseFilenameFromContentDisposition` regex, `chipSummary` rollup, export
form state) only surface during a manual smoke test. The recent `.bin`
filename bug went undetected for an entire release cycle because nothing
exercised the download flow end-to-end.

**Options**:
1. Add `vitest` + `@testing-library/react` and start with unit tests for
   the pure functions in `lib/readiness.ts` and `lib/exportApi.ts`. About
   1 hour of setup; high ROI for the testable logic.
2. Add Playwright as a dev dependency and write a small E2E suite that
   exercises the upload → analyze → export flow. More setup, but catches
   the kinds of bugs that unit tests can't.

The pragmatic minimum is Option 1.

## README "Test count at HEAD" is stale

**Symptom**: `README.md` claims `Test count at HEAD: 383 unit tests
passing`. Current actual count: 456+ across the v0.1c–v0.2c web UI work
plus the health probes, installer scripts, and export fixes.

**Fix**: replace with a less brittle phrasing ("450+ unit and integration
tests") or remove the count entirely — a number that needs manual
maintenance in a README is destined to drift.
