"""Human-readable summary and timeline plot for analysis.json.

The `summarize` helper returns a JSON-serialisable dict of headline counts and
config from an `AnalysisResult`, intended for ad-hoc inspection from the CLI.
The `plot_timeline` helper renders a 3-panel matplotlib figure (LUFS, onsets
per stem, and section bands) for visual debugging of the analysis pipeline.

matplotlib is imported lazily inside `plot_timeline` so importing this module
does not require the optional dev dependency to be installed.
"""

from __future__ import annotations

from pathlib import Path

from musicue.schemas import AnalysisResult


def summarize(analysis_path: Path) -> dict:
    result = AnalysisResult.model_validate_json(analysis_path.read_text())
    onset_counts = {stem: len(events) for stem, events in result.onsets.items()}
    return {
        "duration_sec": result.source.duration_sec,
        "sample_rate": result.source.sample_rate,
        "schema_version": result.schema_version,
        "beat_count": len(result.beats),
        "downbeat_count": sum(1 for b in result.beats if b.is_downbeat),
        "section_count": len(result.sections),
        "sections": [{"label": s.label, "start": s.start, "end": s.end} for s in result.sections],
        "onset_counts": onset_counts,
        "curves": {name: len(curve.values) for name, curve in result.curves.items()},
        "phrase_counts": {stem: len(ps) for stem, ps in result.phrases.items()},
        "analysis_config": result.analysis_config.model_dump(),
    }


def plot_timeline(analysis_path: Path, out_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    result = AnalysisResult.model_validate_json(analysis_path.read_text())

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    if "lufs" in result.curves:
        curve = result.curves["lufs"]
        t = np.arange(len(curve.values)) * curve.hop_sec
        axes[0].plot(t, curve.values, color="royalblue", lw=0.8)
        axes[0].set_ylabel("LUFS")

    for stem, events in result.onsets.items():
        times = [e.t for e in events]
        axes[1].vlines(times, 0, 1, label=stem, alpha=0.6, lw=0.6)
    axes[1].set_ylabel("Onsets")
    axes[1].legend(loc="upper right", fontsize=7)

    colors = {"intro": "#4CAF50", "verse": "#2196F3", "chorus": "#FF5722",
              "bridge": "#9C27B0", "outro": "#607D8B"}
    for s in result.sections:
        color = colors.get(s.label, "#999")
        axes[2].axvspan(s.start, s.end, alpha=0.3, color=color, label=s.label)
        axes[2].text((s.start + s.end) / 2, 0.5, s.label, ha="center", fontsize=8)
    axes[2].set_ylabel("Sections")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    if out_path:
        plt.savefig(str(out_path), dpi=120)
    else:
        plt.show()
    plt.close()
