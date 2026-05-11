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


def test_probe_torch_ready(monkeypatch):
    fake_torch = type("M", (), {"__version__": "2.2.1"})()
    monkeypatch.setitem(
        __import__("sys").modules, "torch", fake_torch
    )
    status = probes.probe_torch()
    assert status.state == ComponentState.READY
    assert status.version == "2.2.1"


def test_probe_torch_missing_when_import_fails(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "torch", None)
    status = probes.probe_torch()
    assert status.state == ComponentState.MISSING


def test_probe_torch_degraded_on_old_version(monkeypatch):
    fake_torch = type("M", (), {"__version__": "1.13.0"})()
    monkeypatch.setitem(
        __import__("sys").modules, "torch", fake_torch
    )
    status = probes.probe_torch()
    assert status.state == ComponentState.DEGRADED
    assert "2.2" in (status.detail or "")


def test_probe_cuda_ready(monkeypatch):
    fake_torch = type(
        "M",
        (),
        {
            "cuda": type(
                "C",
                (),
                {
                    "is_available": staticmethod(lambda: True),
                    "get_device_name": staticmethod(lambda i: "NVIDIA RTX 4090"),
                },
            )()
        },
    )()
    monkeypatch.setitem(
        __import__("sys").modules, "torch", fake_torch
    )
    status = probes.probe_cuda()
    assert status.state == ComponentState.READY
    assert "RTX 4090" in (status.detail or "")
    assert status.required is False


def test_probe_ffmpeg_ready(monkeypatch):
    monkeypatch.setattr(
        probes.shutil, "which", lambda cmd: r"C:\tools\ffmpeg.exe"
    )
    status = probes.probe_ffmpeg()
    assert status.state == ComponentState.READY
    assert status.cache_path == r"C:\tools\ffmpeg.exe"


def test_probe_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(probes.shutil, "which", lambda cmd: None)
    status = probes.probe_ffmpeg()
    assert status.state == ComponentState.MISSING
    assert status.required is True


def test_probe_basic_pitch_ready(monkeypatch):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "basic_pitch", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "0.4.0"
    )
    status = probes.probe_basic_pitch()
    assert status.state == ComponentState.READY
    assert status.version == "0.4.0"
    assert status.required is True


def test_probe_basic_pitch_missing(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "basic_pitch", None)
    status = probes.probe_basic_pitch()
    assert status.state == ComponentState.MISSING
    assert "basic-pitch" in (status.remediation or "")


def test_probe_demucs_ready_when_all_checkpoints_present(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "demucs", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "4.0.1"
    )
    for i in range(probes._DEMUCS_FT_EXPECTED_COUNT):
        (tmp_path / f"abc{i}-hash.th").write_bytes(b"x")
    monkeypatch.setattr(
        probes, "_torch_hub_checkpoint_dir", lambda: tmp_path
    )
    status = probes.probe_demucs()
    assert status.state == ComponentState.READY


def test_probe_demucs_degraded_when_partial(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "demucs", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "4.0.1"
    )
    (tmp_path / "abc-only.th").write_bytes(b"x")
    monkeypatch.setattr(
        probes, "_torch_hub_checkpoint_dir", lambda: tmp_path
    )
    status = probes.probe_demucs()
    assert status.state == ComponentState.DEGRADED


def test_probe_demucs_missing_when_no_checkpoints(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "demucs", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "4.0.1"
    )
    monkeypatch.setattr(
        probes, "_torch_hub_checkpoint_dir", lambda: tmp_path
    )
    status = probes.probe_demucs()
    assert status.state == ComponentState.MISSING


def test_probe_demucs_missing_when_import_fails(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "demucs", None)
    status = probes.probe_demucs()
    assert status.state == ComponentState.MISSING


def test_probe_allin1_ready(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "allin1", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "1.1.0"
    )
    (tmp_path / "ckpt.pt").write_bytes(b"x")
    monkeypatch.setattr(probes, "_allin1_cache_dir", lambda: tmp_path)
    status = probes.probe_allin1()
    assert status.state == ComponentState.READY
    assert status.required is False


def test_probe_allin1_missing_when_import_fails(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "allin1", None)
    status = probes.probe_allin1()
    assert status.state == ComponentState.MISSING
    assert "librosa fallback" in (status.detail or "")


def test_probe_allin1_missing_when_no_checkpoint(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "allin1", fake)
    monkeypatch.setattr(probes, "_allin1_cache_dir", lambda: tmp_path)
    status = probes.probe_allin1()
    assert status.state == ComponentState.MISSING


def test_probe_clap_ready(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "laion_clap", fake)
    monkeypatch.setattr(
        probes, "_pkg_version_or_none", lambda name: "1.1.6"
    )
    (tmp_path / "630k-audioset-best.pt").write_bytes(b"x")
    monkeypatch.setattr(probes, "_clap_cache_dirs", lambda: [tmp_path])
    status = probes.probe_clap()
    assert status.state == ComponentState.READY


def test_probe_clap_missing_when_no_weights(monkeypatch, tmp_path):
    import sys as _sys

    fake = type("M", (), {})()
    monkeypatch.setitem(_sys.modules, "laion_clap", fake)
    monkeypatch.setattr(probes, "_clap_cache_dirs", lambda: [tmp_path])
    status = probes.probe_clap()
    assert status.state == ComponentState.MISSING


def test_probe_clap_missing_when_import_fails(monkeypatch):
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "laion_clap", None)
    status = probes.probe_clap()
    assert status.state == ComponentState.MISSING


def test_probe_cuda_missing_when_unavailable(monkeypatch):
    fake_torch = type(
        "M",
        (),
        {
            "cuda": type(
                "C",
                (),
                {"is_available": staticmethod(lambda: False)},
            )()
        },
    )()
    monkeypatch.setitem(
        __import__("sys").modules, "torch", fake_torch
    )
    status = probes.probe_cuda()
    assert status.state == ComponentState.MISSING
    assert "GPU" in (status.detail or "")
