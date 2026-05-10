"""End-to-end M1 analysis pipeline.

Wires together the M0 and M1 analysis modules:

* Demucs source separation (drums/bass/vocals/other).
* All-In-One beat/downbeat/section detection (with a librosa fallback).
* Per-stem onset detection.
* Polyphonic transcription via Basic Pitch (optional dependency).
* Phrase grouping over transcribed notes.
* Optional CLAP semantic re-ranking of onset events.
* Mix-level + per-stem curves: LUFS, spectral centroid, spectral flux,
  stereo width / pan, per-stem RMS.
* Section-transition ramps derived from spectral flux and LUFS.

Heavy / optional dependencies (basic-pitch, laion-clap) are wrapped in
try/except blocks so the pipeline degrades gracefully when they are not
installed: it logs a warning and proceeds with empty MIDI / phrase / label
data for the affected stems instead of crashing.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import soundfile as sf

from musicue.analysis.clap_reranker import attach_clap_labels, clap_version
from musicue.analysis.curves import (
    compute_integrated_lufs,
    compute_lufs_curve,
    compute_rms_curve,
    compute_spectral_centroid_curve,
    compute_spectral_flux_curve,
    compute_stereo_width_pan,
)
from musicue.analysis.drum_classifier import classify_onsets_batch, drum_classifier_version
from musicue.analysis.onsets import detect_onsets
from musicue.analysis.phrases import group_into_phrases
from musicue.analysis.separation import demucs_version, separate
from musicue.analysis.structure import allin1_version, detect_structure
from musicue.analysis.transcription import basic_pitch_version, transcribe_stem
from musicue.analysis.transitions import derive_transitions
from musicue.cache import Cache, build_audio_cache_key
from musicue.config import MusiCueConfig
from musicue.schemas import (
    AnalysisConfig,
    AnalysisResult,
    BeatEvent,
    MidiNote,
    OnsetEvent,
    PhraseEvent,
    SectionEvent,
    SectionTransition,
    SourceInfo,
    TempoInfo,
    TimedCurve,
)

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_dict(cfg: MusiCueConfig) -> dict:
    drum_model_path = Path("models/drum_cnn.pt")
    return {
        "demucs_model": cfg.analysis.demucs_model,
        "demucs_version": demucs_version(),
        "allin1_version": allin1_version(),
        "basic_pitch_version": basic_pitch_version(),
        "clap_version": clap_version(),
        "drum_classifier_version": drum_classifier_version(drum_model_path),
        "beat_backend": cfg.analysis.beat_backend,
        "curve_hop_sec": cfg.analysis.curve_hop_sec,
    }


def compute_run_dir(audio_path: Path, cfg: MusiCueConfig) -> Path:
    """Return the deterministic ``run_dir`` ``run_analysis`` will use for this input.

    Exposed so callers (e.g. benchmarks) can pre-populate ``run_dir/stems`` and
    have ``run_analysis`` reuse them via ``separate()``'s idempotent fast path.
    """
    resolved = audio_path.resolve()
    cache_key = build_audio_cache_key(resolved, _version_dict(cfg))
    return cfg.runs_dir / cache_key[:12]


def _write_run_artifacts(
    result: AnalysisResult,
    audio_path: Path,
    run_dir: Path,
) -> None:
    """Materialize the UI-visible artifacts of an analysis into run_dir.

    Called on both the fresh-analysis path and the cache-hit path so that
    consumers reading from `run_dir/*` always find the same layout
    regardless of whether the analysis was just computed or pulled from
    cache. Re-running this is cheap: peaks are derived from existing audio.
    """
    from musicue.analysis.peaks import write_peaks

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis.json").write_text(result.model_dump_json(indent=2))
    write_peaks(audio_path, run_dir / "peaks.mix.json")
    for stem_name, stem_path_str in result.stems.items():
        stem_path = Path(stem_path_str)
        if stem_path.exists():
            write_peaks(stem_path, run_dir / f"peaks.{stem_name}.json")


def run_analysis(audio_path: Path, cfg: MusiCueConfig) -> AnalysisResult:
    audio_path = audio_path.resolve()
    version_dict = _version_dict(cfg)
    cache_key = build_audio_cache_key(audio_path, version_dict)
    cache = Cache(cfg.cache_dir)

    cached = cache.get(cache_key, "analysis.json")
    if cached is not None:
        result = AnalysisResult.model_validate_json(cached.read_text())
        # Always materialize the UI artifacts on cache hit. The cache only
        # holds analysis.json; the run_dir layout (analysis.json + peaks.*)
        # may have been wiped (manual cleanup, fresh checkout, different
        # cfg.runs_dir). Re-derive cheaply from the cached result + audio.
        run_dir = cfg.runs_dir / cache_key[:12]
        try:
            _write_run_artifacts(result, audio_path, run_dir)
        except Exception as exc:
            # Stems may be missing on disk; fall through to a fresh analysis
            # rather than returning a partially-materialized result.
            log.warning(
                "Cache-hit artifact materialization failed (%s: %s); "
                "falling back to a fresh analysis.",
                type(exc).__name__,
                exc,
            )
        else:
            return result

    sha256 = _sha256(audio_path)
    try:
        info = sf.info(str(audio_path))
        duration_sec = info.frames / info.samplerate
        sample_rate = info.samplerate
    except sf.LibsndfileError:
        # libsndfile can't probe compressed inputs (m4a/mp3/aac); use librosa
        # which proxies through audioread + ffmpeg.
        import librosa as _librosa

        duration_sec = float(_librosa.get_duration(path=str(audio_path)))
        sample_rate = int(_librosa.get_samplerate(str(audio_path)))

    run_dir = cfg.runs_dir / cache_key[:12]

    # --- Source separation -------------------------------------------------
    stems = separate(audio_path, run_dir / "stems", model=cfg.analysis.demucs_model)
    stems_str = {k: str(v) for k, v in stems.items()}

    # --- Structure: tempo / beats / sections -------------------------------
    structure = detect_structure(audio_path, backend=cfg.analysis.beat_backend)
    tempo = TempoInfo.model_validate(structure["tempo"])
    beats = [BeatEvent.model_validate(b) for b in structure["beats"]]
    sections = [SectionEvent.model_validate(s) for s in structure["sections"]]
    sections_dicts = [s.model_dump() for s in sections]

    # --- Onsets per stem ---------------------------------------------------
    onsets: dict[str, list[OnsetEvent]] = {}
    for stem_name, stem_path in stems.items():
        onsets[stem_name] = [OnsetEvent.model_validate(o) for o in detect_onsets(stem_path)]

    # --- Drum classification (best-effort) ---------------------------------
    drum_model_path = Path("models/drum_cnn.pt")
    if drum_model_path.exists() and "drums" in stems:
        try:
            import numpy as np

            drum_audio, drum_sr = sf.read(str(stems["drums"]))
            if drum_audio.ndim > 1:
                drum_audio = drum_audio.mean(axis=1)
            drum_onset_dicts = [o.model_dump() for o in onsets.get("drums", [])]
            classified = classify_onsets_batch(
                drum_onset_dicts,
                drum_audio.astype(np.float32),
                sr=drum_sr,
                model_path=drum_model_path,
            )
            onsets["drums"] = [OnsetEvent.model_validate(e) for e in classified]
        except Exception as exc:
            log.warning(
                "Drum classification skipped (%s: %s).",
                type(exc).__name__,
                exc,
            )

    # --- Transcription + phrasing (vocals, other) --------------------------
    midi: dict[str, list[dict]] = {}
    phrases: dict[str, list[PhraseEvent]] = {}
    for stem_name in ("vocals", "other"):
        stem_path = stems.get(stem_name)
        if stem_path is None:
            continue
        try:
            notes = transcribe_stem(stem_path)
            midi[stem_name] = notes
            gap = cfg.analysis.phrase_gap_sec.get(stem_name, 0.5)
            raw_phrases = group_into_phrases(notes, gap_sec=gap)
            phrases[stem_name] = [PhraseEvent.model_validate(p) for p in raw_phrases]
        except Exception as exc:
            log.warning(
                "Transcription/phrasing failed for stem %r (%s: %s); skipping.",
                stem_name,
                type(exc).__name__,
                exc,
            )
            midi.setdefault(stem_name, [])
            phrases.setdefault(stem_name, [])

    # --- Curves: mix-level + per-stem RMS ----------------------------------
    curves: dict[str, TimedCurve] = {
        "lufs": TimedCurve(**compute_lufs_curve(audio_path, hop_sec=cfg.analysis.curve_hop_sec)),
        "spectral_centroid": TimedCurve(
            **compute_spectral_centroid_curve(audio_path, hop_sec=cfg.analysis.curve_hop_sec)
        ),
        "spectral_flux": TimedCurve(
            **compute_spectral_flux_curve(audio_path, hop_sec=cfg.analysis.curve_hop_sec)
        ),
    }
    stereo = compute_stereo_width_pan(audio_path, hop_sec=cfg.analysis.curve_hop_sec)
    curves["stereo_width"] = TimedCurve(**stereo["width"])
    curves["stereo_pan"] = TimedCurve(**stereo["pan"])
    for stem_name, stem_path in stems.items():
        curves[f"rms_{stem_name}"] = TimedCurve(
            **compute_rms_curve(stem_path, hop_sec=cfg.analysis.curve_hop_sec)
        )

    # --- CLAP labeling (best-effort) ---------------------------------------
    if cfg.analysis.clap_top_k > 0:
        try:
            import yaml

            prompts_file = Path("prompt_banks/default_clap_prompts.yaml")
            if prompts_file.exists():
                prompts = yaml.safe_load(prompts_file.read_text()).get("prompts", [])
            else:
                prompts = ["sub bass drop", "vocal stab", "punchy kick", "deep kick"]
            for stem_name in list(onsets.keys()):
                raw_events = [o.model_dump() for o in onsets[stem_name]]
                labeled = attach_clap_labels(
                    raw_events,
                    audio_path=audio_path,
                    prompts=prompts,
                    enabled=True,
                    threshold=cfg.analysis.clap_threshold,
                    top_k=cfg.analysis.clap_top_k,
                )
                onsets[stem_name] = [OnsetEvent.model_validate(e) for e in labeled]
        except Exception as exc:
            log.warning(
                "CLAP labeling skipped (%s: %s).",
                type(exc).__name__,
                exc,
            )

    # --- Section transitions ----------------------------------------------
    flux_dict = {
        "hop_sec": curves["spectral_flux"].hop_sec,
        "values": curves["spectral_flux"].values,
    }
    lufs_dict = {
        "hop_sec": curves["lufs"].hop_sec,
        "values": curves["lufs"].values,
    }
    raw_transitions = derive_transitions(sections_dicts, flux_dict, lufs_dict)
    section_transitions = [SectionTransition.model_validate(t) for t in raw_transitions]

    # --- MIDI typing -------------------------------------------------------
    typed_midi: dict[str, list[MidiNote]] = {
        stem: [MidiNote.model_validate(n) for n in notes]
        for stem, notes in midi.items()
    }

    result = AnalysisResult(
        source=SourceInfo(
            path=str(audio_path),
            sha256=sha256,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
        ),
        analysis_config=AnalysisConfig(
            demucs_model=cfg.analysis.demucs_model,
            demucs_version=demucs_version(),
            allin1_version=allin1_version(),
            basic_pitch_version=basic_pitch_version(),
            clap_version=clap_version(),
            drum_classifier_version=drum_classifier_version(Path("models/drum_cnn.pt")),
            beat_backend=cfg.analysis.beat_backend,
            curve_hop_sec=cfg.analysis.curve_hop_sec,
        ),
        stems=stems_str,
        tempo=tempo,
        beats=beats,
        sections=sections,
        section_transitions=section_transitions,
        onsets=onsets,
        midi=typed_midi,
        phrases=phrases,
        curves=curves,
        lufs_integrated=compute_integrated_lufs(audio_path),
    )

    # Populate frame/timecode on every event using the configured fps.
    from musicue.frame_population import populate_analysis_frames

    result = populate_analysis_frames(
        result,
        fps=result.analysis_config.fps,
        drop_frame=result.analysis_config.drop_frame,
    )

    # Detect beat patterns and stamp phrase / fill / syncopation fields
    # onto every BeatEvent. Cheap heuristic; runs in milliseconds.
    from musicue.analysis.patterns import populate_beat_pattern_fields

    result = populate_beat_pattern_fields(result)

    _write_run_artifacts(result, audio_path, run_dir)
    cache.put(cache_key, "analysis.json", run_dir / "analysis.json")
    return result
