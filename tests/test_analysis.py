import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from musicue.analysis.separation import separate, demucs_version


def test_demucs_version_returns_string():
    v = demucs_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_separate_raises_on_demucs_failure(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        with pytest.raises(RuntimeError, match="Demucs failed"):
            separate(wav, tmp_path / "out")


def test_separate_returns_four_stem_paths(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals", "other"):
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        stems = separate(wav, out_dir, model=model)

    assert set(stems.keys()) == {"drums", "bass", "vocals", "other"}
    for p in stems.values():
        assert p.exists()


def test_separate_raises_when_stem_missing(tmp_path):
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"\x00" * 100)
    out_dir = tmp_path / "out"
    model = "htdemucs_ft"
    stem_dir = out_dir / model / wav.stem
    stem_dir.mkdir(parents=True)
    for s in ("drums", "bass", "vocals"):  # "other" missing
        (stem_dir / f"{s}.wav").write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with pytest.raises(FileNotFoundError, match="other"):
            separate(wav, out_dir, model=model)
