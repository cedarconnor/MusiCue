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
