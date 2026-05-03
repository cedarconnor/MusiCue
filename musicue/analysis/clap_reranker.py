"""CLAP-based semantic re-ranker for candidate event windows.

CLAP (Contrastive Language-Audio Pretraining; LAION) maps short audio clips
and natural-language prompts into a shared embedding space. We use it as a
post-hoc re-ranker over the event timeline: for each candidate event, extract
a 2-second mono window centered on the event time, embed it alongside a fixed
prompt bank ("sub bass drop", "punchy kick", "vocal chop", ...), and attach
top-k prompt labels whose cosine similarity exceeds a configurable threshold.

The dependency (``laion-clap``) is optional under the ``clap`` extra and is
heavyweight (model weights are roughly 4 GB), so the model loader and the
audio-window extractor are split into module-level helpers that tests can
patch out. The model is cached in ``_MODEL_CACHE`` so repeated calls within a
single process pay the load cost only once.

Note on the ``audio_path is None`` case: the production function does *not*
early-return when ``audio_path is None`` because tests legitimately call the
function with ``audio_path=None`` while patching ``_extract_window`` to
return synthetic windows. At runtime, calling with ``audio_path=None``
without mocks will raise inside ``_extract_window`` (which is the desired
behaviour -- it's a programming error).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_CACHE: Any = None


def clap_version() -> str:
    """Return the installed laion-clap package version, or 'not installed'."""
    try:
        return _pkg_version("laion-clap")
    except PackageNotFoundError:
        return "not installed"
    except Exception:
        return "not installed"


def _load_model():
    """Lazy-load and cache the CLAP module. Imports laion_clap on first call."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import laion_clap  # type: ignore[import-not-found]

        _MODEL_CACHE = laion_clap.CLAP_Module(enable_fusion=False)
        _MODEL_CACHE.load_ckpt()
    return _MODEL_CACHE


def _extract_window(
    audio_path: Path, t: float, window_sec: float = 2.0, sr: int = 44100
) -> np.ndarray:
    """Extract a mono ``window_sec`` clip centered on time ``t`` at ``sr`` Hz.

    Reads the file via soundfile, downmixes to mono if needed, slices a
    centered window with zero-padding when the event is near a file boundary,
    and resamples to ``sr`` (CLAP's expected rate of 44.1 kHz) when the file's
    native rate differs.
    """
    import soundfile as sf

    data, file_sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    n_samples = int(window_sec * file_sr)
    center = int(t * file_sr)
    start = max(0, center - n_samples // 2)
    end = min(len(data), start + n_samples)
    chunk = data[start:end]
    if len(chunk) < n_samples:
        chunk = np.pad(chunk, (0, n_samples - len(chunk)))
    if file_sr != sr:
        import librosa

        chunk = librosa.resample(chunk.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return chunk.astype(np.float32)


def attach_clap_labels(
    events: list[dict],
    audio_path: Path | None,
    prompts: list[str],
    enabled: bool = True,
    threshold: float = 0.55,
    top_k: int = 3,
) -> list[dict]:
    """Attach top-k CLAP prompt labels above ``threshold`` to each event.

    For each event, a 2-second window centered on ``event["t"]`` is embedded
    via CLAP, cosine similarity is computed against each prompt embedding,
    and the top-k matches whose score is at least ``threshold`` are appended
    to ``event["labels"]`` as dicts of the shape
    ``{"label": str, "score": float, "source": "clap"}``.

    The function mutates and returns ``events``. When ``enabled`` is False or
    ``prompts`` is empty, it is a no-op.
    """
    if not enabled or not prompts:
        return events

    model = _load_model()
    audio_windows = [_extract_window(audio_path, e["t"]) for e in events]  # type: ignore[arg-type]

    audio_embeddings = model.get_audio_embedding_from_data(
        np.stack(audio_windows), use_tensor=False
    )
    text_embeddings = model.get_text_embedding(prompts, use_tensor=False)

    # Cosine similarity: (n_events, n_prompts).
    audio_norm = audio_embeddings / (
        np.linalg.norm(audio_embeddings, axis=1, keepdims=True) + 1e-9
    )
    text_norm = text_embeddings / (
        np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-9
    )
    scores = audio_norm @ text_norm.T  # (n_events, n_prompts)

    for i, event in enumerate(events):
        top_indices = np.argsort(scores[i])[::-1][:top_k]
        labels = []
        for idx in top_indices:
            score = float(scores[i, idx])
            if score >= threshold:
                labels.append({"label": prompts[idx], "score": score, "source": "clap"})
        event["labels"] = labels
    return events
