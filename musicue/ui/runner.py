"""ProcessPoolExecutor-backed analyze pool with cancel support.

Lives outside FastAPI so it can be unit-tested standalone. The pool runs
one analyze at a time (``max_workers=1``) -- Demucs already maxes out a
GPU; concurrency would just thrash. Job cancellation is implemented in a
follow-up task.
"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable


class AnalyzePool:
    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        # job_id -> Future. Populated on submit, cleaned on completion.
        self._futures: dict[str, Future] = {}

    def submit(self, job_id: str, fn: Callable, *args, **kwargs) -> Future:
        fut = self._pool.submit(fn, *args, **kwargs)
        self._futures[job_id] = fut
        fut.add_done_callback(lambda _f, jid=job_id: self._futures.pop(jid, None))
        return fut

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
