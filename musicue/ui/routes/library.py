"""Library-level routes: index events + empty-trash."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from musicue.index import index as indexer

router = APIRouter(prefix="/api/library", tags=["library"])


@router.get("/index_events")
async def index_events(request: Request) -> StreamingResponse:
    """SSE banner for cold rebuilds.

    v0.1b runs ``ensure_current`` synchronously on startup, so by the time
    anyone subscribes here we are already idle. The route is kept as a hook
    for future async rebuilds; today it emits a single ``idle`` event and
    closes.
    """

    async def stream():
        yield f"data: {json.dumps({'type':'idle'})}\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/empty-trash")
def empty_trash(request: Request) -> dict:
    db = request.app.state.index_db
    root = request.app.state.storage_root
    return indexer.empty_trash(db, root)
