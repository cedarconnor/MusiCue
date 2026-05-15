"""FastAPI app factory for the Web UI MVP."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from musicue.health.readiness import collect_report
from musicue.ui.jobs import JobManager
from musicue.ui.storage import UIStorage


def create_app(storage_root: Path | None = None) -> FastAPI:
    import sqlite3

    from musicue.index import index as indexer
    from musicue.index import schema as index_schema

    app = FastAPI(title="MusiCue Web UI", version="0.1b")
    root = Path(storage_root or (Path.home() / ".musicue"))
    root.mkdir(parents=True, exist_ok=True)
    app.state.storage = UIStorage(root)
    app.state.jobs = JobManager()

    db = sqlite3.connect(root / "index.db", check_same_thread=False)
    db.execute("PRAGMA foreign_keys = ON")
    db.row_factory = sqlite3.Row
    index_schema.create_all(db)
    indexer.ensure_current(db, root)
    app.state.index_db = db
    app.state.storage_root = root

    from musicue.ui.runner import AnalyzePool

    app.state.pool = AnalyzePool(max_workers=1)
    app.state.jobs.register_cancel_hook(app.state.pool.cancel)

    app.state.readiness_report = collect_report()

    @app.on_event("shutdown")
    def _shutdown_pool() -> None:
        app.state.pool.shutdown(wait=False)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    from musicue.ui.routes import analyses as analyses_routes
    from musicue.ui.routes import cedartoy as cedartoy_routes
    from musicue.ui.routes import click as click_routes
    from musicue.ui.routes import export as export_routes
    from musicue.ui.routes import health as health_routes
    from musicue.ui.routes import jobs as jobs_routes
    from musicue.ui.routes import library as library_routes
    from musicue.ui.routes import songs as songs_routes
    app.include_router(songs_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(analyses_routes.router)
    app.include_router(click_routes.router)
    app.include_router(library_routes.router)
    app.include_router(export_routes.router)
    app.include_router(health_routes.router)
    app.include_router(cedartoy_routes.router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        from fastapi.staticfiles import StaticFiles

        # Vite emits hashed assets to <static>/assets/. Mount them at /assets/
        # with long-cache headers via StaticFiles' default behaviour.
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=assets_dir),
                name="assets",
            )

        index_html = static_dir / "index.html"

        static_root = static_dir.resolve()

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str) -> FileResponse:
            """Catch-all so React Router routes (e.g. /editor/<id>/<id>) load
            the SPA instead of returning FastAPI's plain 404.

            Registered last so all real /api/* routes match first. Anything
            with the api/ prefix that reaches here is genuinely unknown -- we
            return a real 404 instead of silently serving index.html (which
            would let API typos look like working pages).
            """
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            # Serve top-level static files like favicon.ico if they exist;
            # otherwise fall back to the SPA index. Resolve and bound-check
            # the candidate so encoded path traversal (e.g. /%2e%2e/cli.py)
            # cannot escape ``static_dir`` and read source files.
            if full_path:
                try:
                    candidate = (static_dir / full_path).resolve()
                    if (
                        candidate.is_relative_to(static_root)
                        and candidate.is_file()
                    ):
                        return FileResponse(candidate)
                except (OSError, ValueError):
                    pass
            return FileResponse(index_html)

    return app
