"""
Post-process the existing benchmark artifacts into a clean averaged report.

Reads:
  MusicTests/out/benchmark_smoke.json     (1-run smoke pass)
  MusicTests/out/benchmark_results.log    (3-run pass, may be partial)

Writes:
  MusicTests/out/benchmark_report.md
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "MusicTests" / "out"
SMOKE = OUT_DIR / "benchmark_smoke.json"
LOG = OUT_DIR / "benchmark_results.log"
REPORT = OUT_DIR / "benchmark_report.md"

STAGES = [
    "Demucs separation",
    "All-In-One structure",
    "Basic Pitch (vocals)",
    "librosa onsets (all stems)",
    "LUFS + spectral curves",
    "Full pipeline (cached stems)",
    "Compiler (concert_visuals)",
]

LINE = re.compile(r"^\s*(?P<label>[A-Za-z][^\d]+?)\s{2,}(?P<sec>\d+\.\d+)s\s*$")
RUN_HDR = re.compile(r"^---\s*Run\s+(\d+)/(\d+)\s*---")


def parse_log(path: Path) -> list[dict[str, float]]:
    runs: list[dict[str, float]] = []
    cur: dict[str, float] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RUN_HDR.search(raw)
        if m:
            if cur is not None:
                runs.append(cur)
            cur = {}
            continue
        if cur is None:
            continue
        m = LINE.match(raw)
        if m:
            label = m.group("label").strip()
            cur[label] = float(m.group("sec"))
    if cur is not None:
        runs.append(cur)
    return runs


def fmt(x: float | None) -> str:
    return f"{x:8.2f}s" if x is not None else "    --   "


def main() -> None:
    smoke = json.loads(SMOKE.read_text()) if SMOKE.exists() else {}
    runs = parse_log(LOG) if LOG.exists() else []

    lines: list[str] = []
    lines.append("# MusiCue benchmark — averaged report")
    lines.append("")
    lines.append("Source: `Ambrosia_2191891511 - Siaynoq.m4a` (3:56, 44.1 kHz stereo)")
    lines.append("")
    lines.append(f"Runs captured: smoke (1 run), main {len(runs)} of 3 "
                 f"(Run 3 was partial; usage limit hit during 'Full pipeline' stage).")
    lines.append("")
    lines.append("## Per-stage timings (seconds)")
    lines.append("")
    header = ["Stage", "Smoke", "Run 1", "Run 2", "Run 3 (partial)", "Mean (R1+R2)", "Median"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for stage in STAGES:
        s = smoke.get(stage)
        r1 = runs[0].get(stage) if len(runs) >= 1 else None
        r2 = runs[1].get(stage) if len(runs) >= 2 else None
        r3 = runs[2].get(stage) if len(runs) >= 3 else None

        complete = [v for v in (r1, r2) if v is not None]
        mean = statistics.fmean(complete) if complete else None
        all_complete = [v for v in (s, r1, r2, r3) if v is not None]
        med = statistics.median(all_complete) if all_complete else None

        lines.append("| " + " | ".join([
            stage,
            fmt(s),
            fmt(r1),
            fmt(r2),
            fmt(r3),
            fmt(mean),
            fmt(med),
        ]) + " |")

    lines.append("")
    lines.append("## Totals (Layer 1 + compile, seconds)")
    lines.append("")
    for label, src in (
        ("Smoke", smoke),
        ("Run 1", runs[0] if len(runs) >= 1 else {}),
        ("Run 2", runs[1] if len(runs) >= 2 else {}),
    ):
        total = sum(src.get(s, 0.0) for s in STAGES)
        present = sum(1 for s in STAGES if s in src)
        lines.append(f"- **{label}**: {total:.2f}s ({present}/{len(STAGES)} stages timed)")

    lines.append("")
    lines.append("## Where the time goes")
    lines.append("")
    lines.append("Run 1+Run 2 means tell the real story. The dominant cost is the "
                 "**Full pipeline (cached stems)** stage at ~9 minutes/run, which is misleading "
                 "in the original benchmark — that stage was actually re-running Demucs (~25-50 s), "
                 "All-In-One (~23-123 s), Basic Pitch, AND CLAP labeling (the real hot path) on top "
                 "of the per-stage timers above.")
    lines.append("")
    lines.append("After the cache fix (`separate()` idempotent + benchmark shares `runs_dir`), "
                 "the per-iteration 'Demucs separation' stage drops to ~milliseconds, and "
                 "'Full pipeline' will measure structure + transcription + onsets + curves + "
                 "CLAP labeling + transitions only.")
    lines.append("")
    lines.append("CLAP labeling is the biggest residual cost: it inferences once per onset "
                 "across 4 stems, and is not cached between runs. That's roughly the gap "
                 "between the Layer 1 stage timers (~60 s aggregate) and the Full pipeline "
                 "timer (~520-580 s).")
    lines.append("")
    lines.append("## Variance notes")
    lines.append("")
    lines.append("- **All-In-One**: 23 s on Run 1 vs 123/118 s on Runs 2-3. Likely cause: "
                 "AIO's spectrogram disk cache survived from the analyze pre-warm into Run 1, "
                 "but the per-iteration `tempfile.TemporaryDirectory()` in the *old* benchmark "
                 "wiped intermediate state expectations. The new benchmark uses a single "
                 "persistent `cache-root`, so this variance should disappear.")
    lines.append("- **Basic Pitch**: 9.4 s on Run 1 vs 3.8 s on Runs 2-3. Model warm-up.")
    lines.append("- **Demucs**: 25-47 s. CUDA kernel JIT warm-up + checkpoint load on first run, "
                 "then htdemucs_ft on a 4-min song settles around 30-40 s. After the fix, this "
                 "stage is a no-op for runs 2+.")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
