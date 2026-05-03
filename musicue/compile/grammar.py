"""YAML grammar DSL: schema definitions and loader.

A grammar describes how to translate analysis events (onsets, phrases,
sections, curves) into cuesheet tracks. Each track declares a type
(impulse / envelope / step / ramp / continuous), a source path into
``analysis.json``, an optional filter expression, score config, an
ADSR envelope, and optional rarity / cooldown / shape metadata.

This module provides only the schema and loader; the scoring engine,
built-in grammars, and full compiler land in later M2 tasks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class GrammarTrack(BaseModel):
    model_config = {"extra": "allow"}

    name: str
    type: str
    source: str
    filter: str | None = None
    score: dict[str, Any] = Field(default_factory=lambda: {"base": 1.0})
    envelope: dict[str, float] = Field(
        default_factory=lambda: {"a": 0.01, "d": 0.1, "s": 0.0, "r": 0.0}
    )
    rarity: dict[str, float] | None = None
    cooldown_sec: float | None = None
    shape_curve_from: str | None = None
    emit: str | None = None


class Grammar(BaseModel):
    name: str
    hierarchy_weights: dict[str, float] = Field(
        default_factory=lambda: {"macro": 1.0, "meso": 1.0, "micro": 1.0}
    )
    tracks: list[GrammarTrack] = Field(default_factory=list)
    clap_prompts: list[str] | None = None


def load_grammar(
    name_or_path: str | Path, grammars_dir: Path = Path("grammars")
) -> Grammar:
    """Load a grammar from a YAML file path or by name (resolved against ``grammars_dir``)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = grammars_dir / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Grammar file not found: {path}")
    data = yaml.safe_load(path.read_text())
    return Grammar.model_validate(data)
