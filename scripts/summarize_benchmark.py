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
LOG_V2 = OUT_DIR / "benchmark_v2.log"
JSON_V2 = OUT_DIR / "benchmark_results_v2.json"
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

# v2 renamed the first stage to make its cache-hit semantics explicit.
STAGE_ALIAS = {"Demucs separation (cache hit)": "Demucs separation"}

LINE = re.compile(r"^\s*(?P<label>[A-Za-z][^\d]+?)\s{2,}(?P<sec>\d+\.\d+)s\s*$")
RUN_HDR = re.compile(r"^---\s*Run\s+(\d+)/(\d+)\s*---")


def parse_log(path: Path) -> list[dict[str, float]]:
    runs: list[dict[str, float]] = []
    cur: dict[str, float] | None = None
    in_summary = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        # The benchmark prints an "Avg (s)" summary table after the final run;
        # those lines look exactly like per-run stage lines and would otherwise
        # overwrite the last run's data. Detect the summary header and stop
        # consuming stage lines from that point on.
        if raw.lstrip().startswith("Stage") and "Avg" in raw:
            in_summary = True
            if cur is not None:
                runs.append(cur)
                cur = None
            continue
        if in_summary:
            continue
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
            label = STAGE_ALIAS.get(m.group("label").strip(), m.group("label").strip())
            cur[label] = float(m.group("sec"))
    if cur is not None:
        runs.append(cur)
    return runs


def fmt(x: float | None) -> str:
    return f"{x:8.2f}s" if x is not None else "    --   "


def main() -> None:
    smoke = json.loads(SMOKE.read_text()) if SMOKE.exists() else {}
    runs = parse_log(LOG) if LOG.exists() else []
    runs_v2 = parse_log(LOG_V2) if LOG_V2.exists() else []

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

    # ---------------- v2 (post-fix) ---------------------------------------
    if runs_v2:
        lines.append("## v2 — after the cache fix + CLAP perf patch")
        lines.append("")
        lines.append(f"Captured {len(runs_v2)} run(s) on the original `.m4a` with persistent "
                     "`--cache-root`. The benchmark stage label was renamed to "
                     "`Demucs separation (cache hit)` to make the new semantics explicit.")
        lines.append("")
        v2_header = ["Stage"] + [f"v2 Run {i+1}" for i in range(len(runs_v2))] + ["v2 mean"]
        lines.append("| " + " | ".join(v2_header) + " |")
        lines.append("|" + "|".join(["---"] * len(v2_header)) + "|")
        for stage in STAGES:
            row = [stage]
            vals = []
            for r in runs_v2:
                v = r.get(stage)
                row.append(fmt(v))
                if v is not None:
                    vals.append(v)
            row.append(fmt(statistics.fmean(vals)) if vals else fmt(None))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        v2_total = sum(statistics.fmean(
            [r.get(s, 0.0) for r in runs_v2]
        ) for s in STAGES)
        lines.append(f"**v2 total Layer 1 + compile (mean): {v2_total:.2f}s**")
        lines.append("")
        lines.append("### v1 vs v2 comparison")
        lines.append("")
        comp_header = ["Stage", "v1 mean (R1+R2)", "v2 mean", "Δ"]
        lines.append("| " + " | ".join(comp_header) + " |")
        lines.append("|" + "|".join(["---"] * len(comp_header)) + "|")
        for stage in STAGES:
            v1_vals = [r.get(stage) for r in runs[:2] if r.get(stage) is not None]
            v2_vals = [r.get(stage) for r in runs_v2 if r.get(stage) is not None]
            v1m = statistics.fmean(v1_vals) if v1_vals else None
            v2m = statistics.fmean(v2_vals) if v2_vals else None
            if v1m is not None and v2m is not None and v1m > 0:
                delta = (v2m - v1m) / v1m * 100
                delta_str = f"{delta:+.0f}%"
            else:
                delta_str = "--"
            lines.append("| " + " | ".join([stage, fmt(v1m), fmt(v2m), delta_str]) + " |")
        lines.append("")
        lines.append("**Headlines:**")
        lines.append("")
        lines.append("- `separation.py` idempotency makes the per-iteration Demucs stage a "
                     "literal no-op (~0.3 ms vs 25-47 s).")
        lines.append("- The CLAP audio + text embedding caches cut **Full pipeline** roughly "
                     "in half by eliminating ~1,300 redundant full-file decodes and 3 "
                     "redundant text-embedding calls per pipeline run.")
        lines.append("- LUFS got slightly slower (+12%) because the m4a path now goes through "
                     "audioread+ffmpeg instead of the WAV soundfile fast path. Cost of correctness; "
                     "WAV inputs keep the original speed.")
        lines.append("- All-In-One variance is still present (R1 22 s vs R2 113 s). Its disk "
                     "cache invalidation isn't fully understood yet -- left as a follow-up.")
        lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
