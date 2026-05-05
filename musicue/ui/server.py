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

    from musicue.ui.routes import songs as songs_routes
    from musicue.ui.routes import jobs as jobs_routes
    from musicue.ui.routes import analyses as analyses_routes
    from musicue.ui.routes import click as click_routes
    app.include_router(songs_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(analyses_routes.router)
    app.include_router(click_routes.router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app
