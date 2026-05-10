"""Terminal-state guard: late complete()/fail() must not overwrite cancelled."""
from __future__ import annotations

import asyncio

import pytest

from musicue.ui.jobs import JobManager, JobStatus


@pytest.mark.asyncio
async def test_complete_after_cancel_is_dropped():
    jobs = JobManager()
    job = jobs.submit(kind="analyze", payload={"song_id": "a" * 64})
    assert jobs.request_cancel(job.id) is True
    assert job.status is JobStatus.CANCELLED

    # Late complete from a worker that hadn't noticed the cancel yet.
    await jobs.complete(job.id, result={"analysis_id": "abc"})
    # Status stays CANCELLED; the result is NOT recorded.
    assert job.status is JobStatus.CANCELLED
    assert job.result is None


@pytest.mark.asyncio
async def test_fail_after_cancel_is_dropped():
    jobs = JobManager()
    job = jobs.submit(kind="analyze", payload={"song_id": "a" * 64})
    jobs.request_cancel(job.id)

    await jobs.fail(job.id, error="something")
    assert job.status is JobStatus.CANCELLED
    assert job.error is None


@pytest.mark.asyncio
async def test_complete_after_complete_is_dropped():
    """A double-complete must not double-publish either."""
    jobs = JobManager()
    job = jobs.submit(kind="analyze", payload={})
    await jobs.complete(job.id, result={"a": 1})
    await jobs.complete(job.id, result={"a": 2})
    assert job.result == {"a": 1}


@pytest.mark.asyncio
async def test_cancel_invokes_registered_hook():
    jobs = JobManager()
    job = jobs.submit(kind="analyze", payload={})
    calls: list[str] = []
    jobs.register_cancel_hook(calls.append)
    jobs.request_cancel(job.id)
    assert calls == [job.id]


@pytest.mark.asyncio
async def test_cancel_returns_false_for_already_terminal():
    jobs = JobManager()
    job = jobs.submit(kind="analyze", payload={})
    await jobs.complete(job.id, result={})
    assert jobs.request_cancel(job.id) is False
