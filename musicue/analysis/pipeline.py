from __future__ import annotations
import hashlib
import soundfile as sf
from pathlib import Path

from musicue.analysis.curves import compute_lufs_curve, compute_rms_curve
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.separation import separate, demucs_version
from musicue.cache import Cache, build_audio_cache_key
from musicue.config import MusiCueConfig
from musicue.schemas import (
    AnalysisConfig, AnalysisResult, OnsetEvent, SourceInfo, TimedCurve,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_dict(cfg: MusiCueConfig) -> dict:
    return {
        "demucs_model": cfg.analysis.demucs_model,
        "demucs_version": demucs_version(),
        "beat_backend": cfg.analysis.beat_backend,
        "curve_hop_sec": cfg.analysis.curve_hop_sec,
    }


def run_analysis(audio_path: Path, cfg: MusiCueConfig) -> AnalysisResult:
    audio_path = audio_path.resolve()
    version_dict = _version_dict(cfg)
    cache_key = build_audio_cache_key(audio_path, version_dict)
    cache = Cache(cfg.cache_dir)

    cached = cache.get(cache_key, "analysis.json")
    if cached is not None:
        return AnalysisResult.model_validate_json(cached.read_text())

    sha256 = _sha256(audio_path)
    info = sf.info(str(audio_path))
    duration_sec = info.frames / info.samplerate

    run_dir = cfg.runs_dir / cache_key[:12]
    stems = separate(audio_path, run_dir / "stems", model=cfg.analysis.demucs_model)
    stems_str = {k: str(v) for k, v in stems.items()}

    onsets: dict[str, list[OnsetEvent]] = {}
    for stem_name, stem_path in stems.items():
        onsets[stem_name] = [OnsetEvent.model_validate(o) for o in detect_onsets(stem_path)]

    curves: dict[str, TimedCurve] = {
        "lufs": TimedCurve(**compute_lufs_curve(audio_path, hop_sec=cfg.analysis.curve_hop_sec))
    }
    for stem_name, stem_path in stems.items():
        curves[f"rms_{stem_name}"] = TimedCurve(
            **compute_rms_curve(stem_path, hop_sec=cfg.analysis.curve_hop_sec)
        )

    result = AnalysisResult(
        source=SourceInfo(
            path=str(audio_path),
            sha256=sha256,
            duration_sec=duration_sec,
            sample_rate=info.samplerate,
        ),
        analysis_config=AnalysisConfig(
            demucs_model=cfg.analysis.demucs_model,
            demucs_version=demucs_version(),
            beat_backend=cfg.analysis.beat_backend,
        ),
        stems=stems_str,
        onsets=onsets,
        curves=curves,
    )

    out_json = run_dir / "analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(result.model_dump_json(indent=2))
    cache.put(cache_key, "analysis.json", out_json)
    return result
