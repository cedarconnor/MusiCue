"""FastAPI app factory for the Web UI MVP."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from musicue.ui.jobs import JobManager
from musicue.ui.storage import UIStorage


def create_app(storage_root: Path | None = None) -> FastAPI:
    app = FastAPI(title="MusiCue Web UI", version="0.1.0-mvp")
    app.state.storage = UIStorage(storage_root or (Path.home() / ".musicue"))
    app.state.jobs = JobManager()

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
