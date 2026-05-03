from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="musicue", help="Convert songs to typed event timelines for DCC tools.")

_EXPORTERS = {
    "csv": ("musicue.exporters.csv", ".csv"),
    "json": ("musicue.exporters.json_export", ".json"),
}


@app.command()
def analyze(
    song: Path = typer.Argument(..., help="Input audio file (.wav, .flac, .mp3)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output dir for analysis.json"),
) -> None:
    """Run Layer 1 analysis - write analysis.json."""
    from musicue.analysis.pipeline import run_analysis
    from musicue.config import MusiCueConfig

    cfg = MusiCueConfig.from_yaml(config) if config else MusiCueConfig()
    if out:
        cfg.runs_dir = out
    result = run_analysis(song, cfg)
    out_path = (out or cfg.runs_dir / song.stem) / "analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2))
    typer.echo(f"Analysis written to {out_path}")


@app.command()
def compile(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    grammar: str = typer.Option("concert_visuals", "--grammar", "-g"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
) -> None:
    """Run Layer 2 compiler: analysis.json - cuesheet.json."""
    from musicue.compile.compiler import compile_analysis
    from musicue.schemas import AnalysisResult

    analysis = AnalysisResult.model_validate_json(analysis_path.read_text())
    cuesheet = compile_analysis(analysis, grammar=grammar)
    out_path = out or analysis_path.parent / "cuesheet.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cuesheet.model_dump_json(indent=2))
    typer.echo(f"Cuesheet written to {out_path}")


@app.command()
def export(
    cuesheet_path: Path = typer.Argument(..., help="Path to cuesheet.json"),
    target: str = typer.Option(..., "--target", "-t", help=f"Target: {','.join(_EXPORTERS)}"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
) -> None:
    """Run Layer 3 exporter: cuesheet.json - target format."""
    import importlib

    from musicue.schemas import CueSheet

    if target not in _EXPORTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(_EXPORTERS)}", err=True)
        raise typer.Exit(code=1)

    cuesheet = CueSheet.model_validate_json(cuesheet_path.read_text())
    module_name, suffix = _EXPORTERS[target]
    out_path = out or cuesheet_path.parent / f"cuesheet{suffix}"
    mod = importlib.import_module(module_name)
    mod.export(cuesheet, out_path)
    typer.echo(f"Exported to {out_path}")


@app.command()
def render(
    song: Path = typer.Argument(..., help="Input audio file"),
    grammar: str = typer.Option("concert_visuals", "--grammar", "-g"),
    target: str = typer.Option("csv", "--target", "-t"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Convenience: analyze - compile - export in one shot."""
    import importlib

    from musicue.analysis.pipeline import run_analysis
    from musicue.compile.compiler import compile_analysis
    from musicue.config import MusiCueConfig

    if target not in _EXPORTERS:
        typer.echo(f"Unknown target '{target}'. Available: {', '.join(_EXPORTERS)}", err=True)
        raise typer.Exit(code=1)

    cfg = MusiCueConfig.from_yaml(config) if config else MusiCueConfig()
    analysis = run_analysis(song, cfg)
    cuesheet = compile_analysis(analysis, grammar=grammar)
    module_name, suffix = _EXPORTERS[target]
    out_path = out or Path(song.stem + suffix)
    importlib.import_module(module_name).export(cuesheet, out_path)
    typer.echo(f"Rendered to {out_path}")


@app.command()
def inspect(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    latent: bool = typer.Option(
        False, "--latent", help="Show Music2Latent correlations (requires m2l in analysis)"
    ),
) -> None:
    """Print a human-readable summary of analysis.json."""
    import json

    from musicue.inspect import summarize

    summary = summarize(analysis_path)
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def plot(
    analysis_path: Path = typer.Argument(..., help="Path to analysis.json"),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Save plot to file instead of showing"
    ),
) -> None:
    """Render a matplotlib timeline of the analysis."""
    from musicue.inspect import plot_timeline

    plot_timeline(analysis_path, out_path=out)
    if out:
        typer.echo(f"Plot saved to {out}")


if __name__ == "__main__":
    app()
