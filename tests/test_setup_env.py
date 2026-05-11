from pathlib import Path

import pytest

from scripts import setup_env


def test_writes_env_when_absent(tmp_path):
    example = tmp_path / ".env.example"
    example.write_text("KEY=value\n", encoding="utf-8")
    target = tmp_path / ".env"
    assert setup_env.write_if_missing(example, target) is True
    assert target.read_text(encoding="utf-8") == "KEY=value\n"


def test_skips_when_env_exists(tmp_path):
    example = tmp_path / ".env.example"
    example.write_text("KEY=fresh\n", encoding="utf-8")
    target = tmp_path / ".env"
    target.write_text("KEY=preserved\n", encoding="utf-8")
    assert setup_env.write_if_missing(example, target) is False
    assert target.read_text(encoding="utf-8") == "KEY=preserved\n"


def test_raises_when_example_missing(tmp_path):
    example = tmp_path / "nope.env.example"
    target = tmp_path / ".env"
    with pytest.raises(FileNotFoundError):
        setup_env.write_if_missing(example, target)
