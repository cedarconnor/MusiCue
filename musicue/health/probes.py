from __future__ import annotations

import shutil
import sys
from functools import wraps
from pathlib import Path
from typing import Callable

from musicue.health.models import ComponentState, ComponentStatus

_REQUIRED: dict[str, bool] = {
    "python_venv": True,
    "torch": True,
    "cuda": False,
    "ffmpeg": True,
    "demucs": True,
    "basic_pitch": True,
    "allin1": False,
    "clap": False,
}


def _wrap(name: str) -> Callable[[Callable[[], ComponentStatus]], Callable[[], ComponentStatus]]:
    def decorator(fn: Callable[[], ComponentStatus]) -> Callable[[], ComponentStatus]:
        @wraps(fn)
        def inner() -> ComponentStatus:
            try:
                return fn()
            except Exception as e:
                return ComponentStatus(
                    name=name,
                    state=ComponentState.ERROR,
                    required=_REQUIRED.get(name, False),
                    detail=str(e)[:200],
                )
        return inner
    return decorator


@_wrap("python_venv")
def probe_python_venv() -> ComponentStatus:
    vi = sys.version_info
    version = f"{vi[0]}.{vi[1]}.{vi[2]}"
    prefix = Path(sys.prefix)
    in_venv = prefix.name == ".venv"

    if not in_venv:
        return ComponentStatus(
            name="python_venv",
            state=ComponentState.MISSING,
            required=True,
            version=version,
            detail=f"Python is not running from a .venv (sys.prefix={sys.prefix})",
            remediation="Run install.bat to create the project venv.",
        )

    if sys.version_info < (3, 11):
        return ComponentStatus(
            name="python_venv",
            state=ComponentState.DEGRADED,
            required=True,
            version=version,
            detail=f"Python {version} is older than the required 3.11.",
            cache_path=str(prefix),
        )

    return ComponentStatus(
        name="python_venv",
        state=ComponentState.READY,
        required=True,
        version=version,
        cache_path=str(prefix),
    )


@_wrap("torch")
def probe_torch() -> ComponentStatus:
    import importlib

    try:
        torch = importlib.import_module("torch")
    except ImportError as e:
        return ComponentStatus(
            name="torch",
            state=ComponentState.MISSING,
            required=True,
            detail=f"torch not importable: {e}",
            remediation="Run install.bat (installs torch 2.2+ with CUDA 12.1).",
        )

    version = getattr(torch, "__version__", "unknown")
    major_minor = version.split(".")[:2]
    try:
        if (int(major_minor[0]), int(major_minor[1])) < (2, 2):
            return ComponentStatus(
                name="torch",
                state=ComponentState.DEGRADED,
                required=True,
                version=version,
                detail=f"torch {version} is older than the recommended 2.2+",
            )
    except (ValueError, IndexError):
        pass

    return ComponentStatus(
        name="torch",
        state=ComponentState.READY,
        required=True,
        version=version,
    )


@_wrap("cuda")
def probe_cuda() -> ComponentStatus:
    import importlib

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return ComponentStatus(
            name="cuda",
            state=ComponentState.MISSING,
            required=False,
            detail="torch is not installed, so CUDA cannot be detected.",
        )

    if not torch.cuda.is_available():
        return ComponentStatus(
            name="cuda",
            state=ComponentState.MISSING,
            required=False,
            detail="No CUDA-capable GPU detected. The pipeline will run on CPU (slower).",
            remediation="Install the CUDA-enabled torch wheel and an NVIDIA driver.",
        )

    name = torch.cuda.get_device_name(0)
    return ComponentStatus(
        name="cuda",
        state=ComponentState.READY,
        required=False,
        detail=f"GPU: {name}",
    )


@_wrap("ffmpeg")
def probe_ffmpeg() -> ComponentStatus:
    path = shutil.which("ffmpeg")
    if path is None:
        return ComponentStatus(
            name="ffmpeg",
            state=ComponentState.MISSING,
            required=True,
            detail="ffmpeg.exe not found on PATH. Required for audio decoding.",
            remediation="Run install.bat (downloads a portable ffmpeg into vendor/).",
        )
    return ComponentStatus(
        name="ffmpeg",
        state=ComponentState.READY,
        required=True,
        cache_path=path,
    )


def _pkg_version_or_none(pkg_name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _v

        return _v(pkg_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


@_wrap("basic_pitch")
def probe_basic_pitch() -> ComponentStatus:
    import importlib

    try:
        importlib.import_module("basic_pitch")
    except ImportError as e:
        return ComponentStatus(
            name="basic_pitch",
            state=ComponentState.MISSING,
            required=True,
            detail=f"basic_pitch not importable: {e}",
            remediation="uv pip install basic-pitch",
        )

    return ComponentStatus(
        name="basic_pitch",
        state=ComponentState.READY,
        required=True,
        version=_pkg_version_or_none("basic-pitch"),
    )
