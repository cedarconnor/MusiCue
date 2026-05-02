from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class AnalysisRunConfig(BaseModel):
    demucs_model: str = "htdemucs_ft"
    beat_backend: str = "allin1"
    phrase_gap_sec: dict[str, float] = Field(default_factory=lambda: {"vocals": 0.6, "other": 0.4})
    clap_top_k: int = 3
    clap_threshold: float = 0.55
    music2latent: bool = False
    curve_hop_sec: float = 0.04


class CompileRunConfig(BaseModel):
    grammar: str = "concert_visuals"
    grammars_dir: Path = Path("grammars")


class MusiCueConfig(BaseModel):
    analysis: AnalysisRunConfig = Field(default_factory=AnalysisRunConfig)
    compile: CompileRunConfig = Field(default_factory=CompileRunConfig)
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".musicue" / "cache")
    runs_dir: Path = Path("runs")

    @classmethod
    def from_yaml(cls, path: Path) -> MusiCueConfig:
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(data)
