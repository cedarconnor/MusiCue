from pathlib import Path

import pytest
import yaml

from musicue.compile.grammar import Grammar, GrammarTrack, load_grammar  # noqa: F401


def _write_grammar(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(data))
    return p


MINIMAL_GRAMMAR = {
    "name": "test_grammar",
    "hierarchy_weights": {"macro": 1.5, "meso": 1.2, "micro": 0.8},
    "tracks": [
        {
            "name": "kick",
            "type": "impulse",
            "source": "onsets.drums",
            "filter": "drum_class == 'kick'",
            "score": {"base": "strength"},
            "envelope": {"a": 0.005, "d": 0.12, "s": 0.0, "r": 0.0},
        }
    ],
}


def test_load_grammar_from_file(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.name == "test_grammar"
    assert len(grammar.tracks) == 1
    assert grammar.tracks[0].name == "kick"


def test_grammar_track_type(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.tracks[0].type == "impulse"
    assert grammar.tracks[0].source == "onsets.drums"


def test_grammar_hierarchy_weights(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    assert grammar.hierarchy_weights["macro"] == pytest.approx(1.5)
    assert grammar.hierarchy_weights["micro"] == pytest.approx(0.8)


def test_grammar_track_envelope(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p)
    env = grammar.tracks[0].envelope
    assert env["a"] == pytest.approx(0.005)
    assert env["s"] == pytest.approx(0.0)


def test_load_grammar_missing_file():
    with pytest.raises(FileNotFoundError):
        load_grammar(Path("nonexistent.yaml"))


def test_load_grammar_by_name(tmp_path):
    p = _write_grammar(tmp_path, MINIMAL_GRAMMAR)
    grammar = load_grammar(p.stem, grammars_dir=tmp_path)
    assert grammar.name == "test_grammar"
