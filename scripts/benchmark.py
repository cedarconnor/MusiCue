"""
MusiCue per-stage latency benchmark.

Usage:
  python scripts/benchmark.py --song path/to/song.wav [--grammar concert_visuals] [--runs 3]

Measures wall-clock time for each Layer 1 stage and the full pipeline.
Outputs a table to stdout and a JSON report to benchmark_results.json.

Cache strategy
--------------
The "Full pipeline (cached stems)" stage is what the label promises: a
full-pipeline run with separation already done. To make that real:

* Demucs is run ONCE before the timing loop, into the deterministic
  ``run_dir`` that ``run_analysis`` will look at (computed via
  ``compute_run_dir``).
* ``separate()`` is idempotent — if the four stems are present, it skips the
  subprocess. So the per-iteration "Demucs separation" timer measures the
  cache-hit cost, and the "Full pipeline" timer doesn't pay separation cost.
* Each run uses a fresh ``cache_dir`` so the analysis.json cache doesn't
  short-circuit the pipeline. Stems live under ``runs_dir`` and are shared.

Note: real measurements require all M1 ML deps (allin1, basic-pitch, demucs,
etc.) to be installed. The script imports them lazily inside ``benchmark()``
so the file itself loads cleanly without those packages.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


@contextmanager
def timer(label: str, results: dict) -> Generator:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    results[label] = elapsed
    print(f"  {label:<35} {elapsed:.2f}s")


def benchmark(song_path: Path, grammar: str, runs: int, cache_root: Path) -> dict:
    from musicue.analysis.curves import (
        compute_lufs_curve,
        compute_spectral_flux_curve,
    )
    from musicue.analysis.onsets import detect_onsets
    from musicue.analysis.pipeline import compute_run_dir, run_analysis
    from musicue.analysis.separation import separate
    from musicue.analysis.structure import detect_structure
    from musicue.analysis.transcription import transcribe_stem
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig

    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cfg = MusiCueConfig()
    cfg.runs_dir = cache_root / "runs"

    # One-time pre-warm: write stems to the path run_analysis expects.
    run_dir = compute_run_dir(song_path, cfg)
    print(f"Pre-warming Demucs into {run_dir / 'stems'} (one-time)...")
    pre_t0 = time.perf_counter()
    stems = separate(song_path, run_dir / "stems", model=cfg.analysis.demucs_model)
    print(f"  prewarm separation: {time.perf_counter() - pre_t0:.2f}s")

    all_results: list[dict[str, float]] = []
    for run in range(1, runs + 1):
        print(f"\n--- Run {run}/{runs} ---")
        # Fresh per-run cache_dir so analysis.json cache does NOT short-circuit
        # the pipeline -- we want every run to re-execute every stage.
        cfg.cache_dir = cache_root / f"cache_run{run}"
        results: dict[str, float] = {}

        with timer("Demucs separation (cache hit)", results):
            separate(song_path, run_dir / "stems", model=cfg.analysis.demucs_model)

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
            analysis = run_analysis(song_path, cfg)

        with timer("Compiler (concert_visuals)", results):
            compile_analysis(analysis, grammar=grammar)

        all_results.append(results)

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
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Persistent cache root (stems live under <root>/runs). "
             "Defaults to a fresh tempdir; pass an explicit path to reuse "
             "stems across invocations.",
    )
    args = parser.parse_args()

    if args.cache_root is None:
        with tempfile.TemporaryDirectory(prefix="musicue_bench_") as tmpdir:
            results = benchmark(args.song, args.grammar, args.runs, Path(tmpdir))
    else:
        results = benchmark(args.song, args.grammar, args.runs, args.cache_root)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.out}")
