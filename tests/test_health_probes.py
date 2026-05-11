import pytest

from musicue.health import probes
from musicue.health.models import ComponentState


def test_required_registry_lists_all_probes():
    expected = {
        "python_venv",
        "torch",
        "cuda",
        "ffmpeg",
        "demucs",
        "basic_pitch",
        "allin1",
        "clap",
    }
    assert set(probes._REQUIRED.keys()) == expected


def test_required_flags_correct():
    assert probes._REQUIRED["python_venv"] is True
    assert probes._REQUIRED["torch"] is True
    assert probes._REQUIRED["ffmpeg"] is True
    assert probes._REQUIRED["demucs"] is True
    assert probes._REQUIRED["basic_pitch"] is True
    assert probes._REQUIRED["cuda"] is False
    assert probes._REQUIRED["allin1"] is False
    assert probes._REQUIRED["clap"] is False


def test_wrap_returns_error_status_when_inner_raises():
    @probes._wrap("demucs")
    def boom() -> probes.ComponentStatus:
        raise RuntimeError("oh no")

    status = boom()
    assert status.state == ComponentState.ERROR
    assert status.name == "demucs"
    assert status.required is True
    assert "oh no" in (status.detail or "")


def test_wrap_passes_through_successful_status():
    @probes._wrap("clap")
    def ok() -> probes.ComponentStatus:
        return probes.ComponentStatus(
            name="clap", state=ComponentState.READY, required=False
        )

    status = ok()
    assert status.state == ComponentState.READY
    assert status.required is False


def test_wrap_truncates_long_detail():
    @probes._wrap("torch")
    def boom() -> probes.ComponentStatus:
        raise RuntimeError("x" * 500)

    status = boom()
    assert status.detail is not None
    assert len(status.detail) <= 200


def test_probe_python_venv_ready(monkeypatch):
    monkeypatch.setattr(probes.sys, "prefix", r"D:\MusiCue\.venv")
    monkeypatch.setattr(
        probes.sys, "version_info", (3, 11, 9, "final", 0)
    )
    status = probes.probe_python_venv()
    assert status.state == ComponentState.READY
    assert status.version == "3.11.9"
    assert status.required is True


def test_probe_python_venv_missing_when_not_in_venv(monkeypatch):
    monkeypatch.setattr(probes.sys, "prefix", r"C:\Python311")
    monkeypatch.setattr(
        probes.sys, "version_info", (3, 11, 9, "final", 0)
    )
    status = probes.probe_python_venv()
    assert status.state == ComponentState.MISSING
    assert "venv" in (status.detail or "").lower()


def test_probe_python_venv_degraded_on_old_python(monkeypatch):
    monkeypatch.setattr(probes.sys, "prefix", r"D:\MusiCue\.venv")
    monkeypatch.setattr(
        probes.sys, "version_info", (3, 10, 0, "final", 0)
    )
    status = probes.probe_python_venv()
    assert status.state == ComponentState.DEGRADED
    assert "3.11" in (status.detail or "")
