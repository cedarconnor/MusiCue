"""In-memory job manager for the Web UI MVP.

Trade-off: no persistence, no priority queues. A server restart kills any
in-flight job. Sufficient for MVP shakedown; v0.1 will replace with the
priority-queue + SQLite design from the spec.
"""
from __future__ import annotations

import asyncio
import enum
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable


class JobStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    payload: dict
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queues: dict[str, list[asyncio.Queue]] = {}
        # Optional callback invoked when request_cancel decides a job is
        # cancellable. Lets the route layer plug in AnalyzePool.cancel
        # without JobManager importing it directly.
        self._cancel_hook: "Callable[[str], None] | None" = None

    def register_cancel_hook(self, hook: "Callable[[str], None]") -> None:
        self._cancel_hook = hook

    def submit(self, kind: str, payload: dict) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, kind=kind, payload=payload)
        self._jobs[job_id] = job
        self._queues[job_id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def list_for_song(self, song_id: str) -> list[Job]:
        """Active (queued/running) jobs whose payload references this song."""
        return [
            j
            for j in self._jobs.values()
            if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            and j.payload.get("song_id") == song_id
        ]

    async def subscribe(self, job_id: str) -> AsyncIterator[dict]:
        if job_id not in self._queues:
            return
        q: asyncio.Queue = asyncio.Queue()
        self._queues[job_id].append(q)
        job = self._jobs[job_id]
        # Replay current status as the first event for late subscribers.
        await q.put({
            "type": "status",
            "status": job.status.value,
            "progress": job.progress,
        })
        # If the job is already in a terminal state when we subscribed
        # (race: runner finished before WS connect), emit the matching
        # terminal event synthetically so the subscriber doesn't hang.
        if job.status is JobStatus.COMPLETE:
            await q.put({"type": "complete", "result": job.result or {}})
        elif job.status is JobStatus.FAILED:
            await q.put({"type": "error", "error": job.error or "unknown"})
        elif job.status is JobStatus.CANCELLED:
            await q.put({"type": "cancelled"})
        try:
            while True:
                evt = await q.get()
                yield evt
                if evt.get("type") in ("complete", "error", "cancelled"):
                    break
        finally:
            if q in self._queues.get(job_id, []):
                self._queues[job_id].remove(q)

    async def publish(self, job_id: str, event: dict) -> None:
        job = self._jobs[job_id]
        if event.get("type") == "progress":
            if "fraction" in event:
                job.progress = float(event["fraction"])
            if job.status is JobStatus.QUEUED:
                job.status = JobStatus.RUNNING
        for q in list(self._queues.get(job_id, [])):
            await q.put(event)

    async def complete(self, job_id: str, result: dict) -> None:
        job = self._jobs[job_id]
        # Don't overwrite a terminal state. If the job was already cancelled,
        # any late "complete" from the still-running worker is dropped on
        # the floor.
        if job.status in (JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.COMPLETE):
            return
        job.status = JobStatus.COMPLETE
        job.result = result
        await self.publish(job_id, {"type": "complete", "result": result})

    async def fail(self, job_id: str, error: str) -> None:
        job = self._jobs[job_id]
        if job.status in (JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.COMPLETE):
            return
        job.status = JobStatus.FAILED
        job.error = error
        await self.publish(job_id, {"type": "error", "error": error})

    def request_cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job.cancel_requested = True
        if self._cancel_hook is not None:
            try:
                self._cancel_hook(job_id)
            except Exception:
                # Hook failure must not block cancellation accounting.
                pass
        # Best-effort: emit a synthetic cancelled event so any subscriber
        # already attached doesn't hang waiting for the analyze coroutine
        # to notice. Workers that do finish naturally afterwards will emit
        # complete/error events too -- those go to no-one because the
        # subscriber's iterator broke on the first terminal event.
        job.status = JobStatus.CANCELLED
        for q in list(self._queues.get(job_id, [])):
            try:
                q.put_nowait({"type": "cancelled"})
            except asyncio.QueueFull:
                pass
        return True
