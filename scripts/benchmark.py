"""
MusiCue per-stage latency benchmark.

Usage:
  python scripts/benchmark.py --song path/to/song.wav [--grammar concert_visuals] [--runs 3]

Measures wall-clock time for each Layer 1 stage and the full pipeline.
Outputs a table to stdout and a JSON report to benchmark_results.json.

Note: real measurements require all M1 ML deps (allin1, basic-pitch, demucs, etc.)
to be installed. The script imports them lazily inside `benchmark()` so the file
itself loads cleanly without those packages.
"""
from __future__ import annotations

import argparse
import json
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


def benchmark(song_path: Path, grammar: str, runs: int) -> dict:
    import tempfile

    from musicue.analysis.curves import (
        compute_lufs_curve,
        compute_spectral_flux_curve,
    )
    from musicue.analysis.onsets import detect_onsets
    from musicue.analysis.separation import separate
    from musicue.analysis.structure import detect_structure
    from musicue.analysis.transcription import transcribe_stem
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig

    all_results: list[dict[str, float]] = []
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
                try:
                    transcribe_stem(stems["vocals"])
                except Exception as exc:
                    print(f"    transcribe_stem skipped ({type(exc).__name__})")

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
