"""ProcessPool-backed analyze runner with cancel support."""
from __future__ import annotations

import time

import pytest

from musicue.ui.runner import AnalyzePool


def _ok_worker(payload: dict) -> dict:
    return {"echo": payload["x"] * 2}


def _boom_worker(_payload: dict) -> dict:
    raise RuntimeError("worker exploded")


def test_submit_runs_function_in_separate_process_and_returns_result() -> None:
    pool = AnalyzePool(max_workers=1)
    try:
        fut = pool.submit("job-a", _ok_worker, {"x": 21})
        result = fut.result(timeout=10.0)
        assert result == {"echo": 42}
    finally:
        pool.shutdown()


def test_submit_propagates_worker_exception() -> None:
    pool = AnalyzePool(max_workers=1)
    try:
        fut = pool.submit("job-b", _boom_worker, {})
        with pytest.raises(RuntimeError, match="worker exploded"):
            fut.result(timeout=10.0)
    finally:
        pool.shutdown()


def _slow_worker(payload: dict) -> dict:
    time.sleep(payload["sleep_sec"])
    return {"done": True}


def test_cancel_terminates_running_worker_within_3s() -> None:
    pool = AnalyzePool(max_workers=1)
    try:
        pool.submit("job-c", _slow_worker, {"sleep_sec": 30.0})
        # Give the worker a moment to actually start.
        time.sleep(0.8)
        cancelled = pool.cancel("job-c")
        assert cancelled is True

        t0 = time.monotonic()
        # The future will eventually resolve (with a BrokenProcessPool-like
        # error or similar). Wait for the pool to clean up.
        while pool.is_running("job-c"):
            time.sleep(0.05)
            if time.monotonic() - t0 > 3.0:
                pytest.fail("worker did not die within 3s of cancel")
    finally:
        pool.shutdown()


def test_cancel_unknown_job_returns_false() -> None:
    pool = AnalyzePool(max_workers=1)
    try:
        assert pool.cancel("never-submitted") is False
    finally:
        pool.shutdown()
