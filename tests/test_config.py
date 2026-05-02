from pathlib import Path
import yaml
import pytest
from musicue.config import MusiCueConfig


def test_default_analysis_config():
    cfg = MusiCueConfig()
    assert cfg.analysis.demucs_model == "htdemucs_ft"
    assert cfg.analysis.beat_backend == "allin1"
    assert cfg.analysis.curve_hop_sec == pytest.approx(0.04)


def test_default_compile_config():
    cfg = MusiCueConfig()
    assert cfg.compile.grammar == "concert_visuals"


def test_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "analysis": {"demucs_model": "htdemucs"},
        "compile": {"grammar": "lighting"},
    }))
    cfg = MusiCueConfig.from_yaml(config_file)
    assert cfg.analysis.demucs_model == "htdemucs"
    assert cfg.compile.grammar == "lighting"


def test_cache_dir_default():
    cfg = MusiCueConfig()
    assert cfg.cache_dir == Path.home() / ".musicue" / "cache"


def test_from_yaml_missing_keys_use_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"analysis": {"demucs_model": "htdemucs"}}))
    cfg = MusiCueConfig.from_yaml(config_file)
    assert cfg.compile.grammar == "concert_visuals"
