"""ProcessPoolExecutor-backed analyze pool with cancel support.

Cancellation strategy: each submitted job runs through a small wrapper
that publishes its OS PID into a ``multiprocessing.Manager`` dict keyed by
job_id. On cancel we look up the PID and ``os.kill(SIGTERM)`` it. On
Windows ``os.kill`` with any non-CTRL signal is mapped to TerminateProcess
by Python itself, so this is cross-platform. ProcessPoolExecutor detects
the dead worker and spawns a replacement transparently.

This is best-effort. Mid-Demucs the GPU state may be left partially
initialised; v0.5 hardens it. Sufficient for v0.1a's user-cancel needs.
"""
from __future__ import annotations

import os
import signal
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager
from typing import Any, Callable


def _wrapped(
    pid_table: Any, job_id: str, fn_qual: str, args: tuple, kwargs: dict
) -> Any:
    """Resolve ``fn_qual`` and run it; publish our PID first.

    ``fn_qual`` is a ``module:attr`` string instead of the function object
    so picklability does not depend on top-level placement of every
    callable; the worker resolves at runtime.
    """
    pid_table[job_id] = os.getpid()
    try:
        mod_name, _, attr = fn_qual.partition(":")
        mod = __import__(mod_name, fromlist=[attr])
        # ``attr`` may be qualified (e.g. ``Class.method``); walk dots.
        target: Any = mod
        for piece in attr.split("."):
            target = getattr(target, piece)
        return target(*args, **kwargs)
    finally:
        try:
            del pid_table[job_id]
        except (KeyError, BrokenPipeError, EOFError):
            pass


class AnalyzePool:
    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future] = {}
        self._manager = Manager()
        self._pid_table = self._manager.dict()

    def submit(self, job_id: str, fn: Callable, *args, **kwargs) -> Future:
        fn_qual = f"{fn.__module__}:{fn.__qualname__}"
        fut = self._pool.submit(
            _wrapped, self._pid_table, job_id, fn_qual, args, kwargs
        )
        self._futures[job_id] = fut
        fut.add_done_callback(lambda _f, jid=job_id: self._futures.pop(jid, None))
        return fut

    def cancel(self, job_id: str) -> bool:
        fut = self._futures.get(job_id)
        if fut is None:
            return False
        # Queued (not yet started): future.cancel() is enough.
        if fut.cancel():
            return True
        # Running: kill the child process by PID.
        try:
            pid = self._pid_table.get(job_id)
        except (BrokenPipeError, EOFError):
            return False
        if pid is None:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
        return True

    def is_running(self, job_id: str) -> bool:
        fut = self._futures.get(job_id)
        if fut is None:
            return False
        return not fut.done()

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
        try:
            self._manager.shutdown()
        except Exception:
            pass
