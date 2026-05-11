from __future__ import annotations

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
