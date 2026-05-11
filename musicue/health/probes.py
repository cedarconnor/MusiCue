from __future__ import annotations

from functools import wraps
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
