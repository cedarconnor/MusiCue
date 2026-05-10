"""Jobs router: snapshot, SSE stream, cancel."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def job_snapshot(job_id: str, request: Request) -> dict:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status.value,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
    }


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream of job progress / completion / cancellation events."""
    mgr = request.app.state.jobs
    if mgr.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def stream():
        async for evt in mgr.subscribe(job_id):
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("type") in ("complete", "error", "cancelled"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{job_id}/cancel")
def job_cancel(job_id: str, request: Request) -> dict:
    ok = request.app.state.jobs.request_cancel(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="cannot cancel")
    return {"cancelling": True}
