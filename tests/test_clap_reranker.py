"""Tests for musicue.analysis.clap_reranker (CLAP semantic re-ranker).

``laion-clap`` is an optional dependency under the ``clap`` extra. The
production code lazy-imports it inside ``_load_model`` and reads audio
inside ``_extract_window``; the tests below mock both so the unit tests
focus on the scoring/labelling logic and run without ``laion-clap`` (or
even an audio file) actually present.
"""
from unittest.mock import MagicMock, patch

import numpy as np

from musicue.analysis.clap_reranker import attach_clap_labels, clap_version


def test_clap_version_returns_string():
    v = clap_version()
    assert isinstance(v, str)


def _make_events():
    return [
        {
            "t": 0.5,
            "strength": 0.9,
            "timescale": "micro",
            "drum_class": "kick",
            "drum_class_conf": 0.9,
            "labels": [],
        },
        {
            "t": 1.5,
            "strength": 0.8,
            "timescale": "micro",
            "drum_class": None,
            "drum_class_conf": None,
            "labels": [],
        },
    ]


def test_attach_clap_labels_no_op_when_disabled():
    events = _make_events()
    result = attach_clap_labels(
        events, audio_path=None, prompts=["sub bass drop"], enabled=False
    )
    assert all(e["labels"] == [] for e in result)


def test_attach_clap_labels_adds_labels_above_threshold():
    events = _make_events()
    prompts = ["punchy kick", "sub bass drop"]

    mock_model = MagicMock()
    # After cosine normalization, event 0 aligns with prompt 0 ("punchy kick")
    # and event 1 has positive overlap with both prompts.
    mock_model.get_audio_embedding_from_data = MagicMock(
        return_value=np.array([[0.8, 0.1], [0.2, 0.3]])
    )
    mock_model.get_text_embedding = MagicMock(
        return_value=np.array([[1.0, 0.0], [0.0, 1.0]])
    )

    # Use the real scoring logic but mock out the model loading + audio extraction
    with patch("musicue.analysis.clap_reranker._load_model", return_value=mock_model):
        with patch(
            "musicue.analysis.clap_reranker._extract_window",
            return_value=np.zeros(44100),
        ):
            result = attach_clap_labels(
                events,
                audio_path=None,
                prompts=prompts,
                enabled=True,
                threshold=0.5,
                top_k=2,
            )
    # Event 0 should have at least one label
    assert len(result[0]["labels"]) > 0
    label = result[0]["labels"][0]
    assert "label" in label and "score" in label and "source" in label
    assert label["source"] == "clap"


def test_attach_clap_labels_skips_below_threshold():
    """Cosine similarity is direction-only; orthogonal vectors give score 0.

    We mock audio embeddings whose direction is orthogonal to the text
    embedding so the resulting cosine similarity is 0, which is below the
    0.5 threshold. The function should attach no labels.
    """
    events = _make_events()
    prompts = ["sub bass drop"]
    mock_model = MagicMock()
    # Audio embeddings point along axis 0; text embedding points along axis 1.
    mock_model.get_audio_embedding_from_data = MagicMock(
        return_value=np.array([[0.1, 0.0], [0.1, 0.0]])
    )
    mock_model.get_text_embedding = MagicMock(
        return_value=np.array([[0.0, 1.0]])
    )

    with patch("musicue.analysis.clap_reranker._load_model", return_value=mock_model):
        with patch(
            "musicue.analysis.clap_reranker._extract_window",
            return_value=np.zeros(44100),
        ):
            result = attach_clap_labels(
                events,
                audio_path=None,
                prompts=prompts,
                enabled=True,
                threshold=0.5,
                top_k=3,
            )
    assert all(e["labels"] == [] for e in result)
