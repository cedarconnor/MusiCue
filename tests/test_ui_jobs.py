import asyncio

from musicue.ui.jobs import JobManager, JobStatus


async def test_submit_creates_queued_job():
    mgr = JobManager()
    job = mgr.submit(kind="analyze", payload={"song_id": "x"})
    assert job.status is JobStatus.QUEUED
    assert job.kind == "analyze"
    snap = mgr.get(job.id)
    assert snap is not None
    assert snap.id == job.id


async def test_run_to_completion_publishes_progress_and_complete():
    mgr = JobManager()
    job = mgr.submit(kind="analyze", payload={})
    received: list[dict] = []
    ready = asyncio.Event()

    async def collect():
        ready.set()
        async for evt in mgr.subscribe(job.id):
            received.append(evt)

    task = asyncio.create_task(collect())
    await ready.wait()
    await asyncio.sleep(0.01)

    await mgr.publish(job.id, {"type": "progress", "fraction": 0.5})
    await mgr.complete(job.id, result={"analysis_id": "abc"})
    await asyncio.wait_for(task, timeout=1.0)

    types = [e["type"] for e in received]
    assert "progress" in types
    assert "complete" in types
    assert mgr.get(job.id).status is JobStatus.COMPLETE


async def test_failed_job_publishes_error():
    mgr = JobManager()
    job = mgr.submit(kind="analyze", payload={})
    await mgr.fail(job.id, error="boom")
    assert mgr.get(job.id).status is JobStatus.FAILED
    assert mgr.get(job.id).error == "boom"


async def test_request_cancel():
    mgr = JobManager()
    job = mgr.submit(kind="analyze", payload={})
    assert mgr.request_cancel(job.id) is True
    assert mgr.get(job.id).cancel_requested is True
