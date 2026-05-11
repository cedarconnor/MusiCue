from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def test_health(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_readiness_returns_cached_report(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/health/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "components" in body
    assert "overall" in body
    assert body["overall"] in {"green", "amber", "red"}
    names = {c["name"] for c in body["components"]}
    assert "demucs" in names
    assert "ffmpeg" in names


def test_post_refresh_replaces_cached_report(tmp_path):
    from datetime import datetime, timezone

    from musicue.health.models import (
        ComponentState,
        ComponentStatus,
        ReadinessReport,
    )

    app = create_app(storage_root=tmp_path)
    client = TestClient(app)

    sentinel = ReadinessReport(
        components=[
            ComponentStatus(
                name="sentinel", state=ComponentState.READY, required=False
            )
        ],
        overall="green",
        checked_at=datetime.now(timezone.utc),
    )
    app.state.readiness_report = sentinel

    r = client.post("/api/health/readiness/refresh")
    assert r.status_code == 200
    body = r.json()
    names = {c["name"] for c in body["components"]}
    assert "sentinel" not in names
    assert "demucs" in names

    assert "sentinel" not in {c.name for c in app.state.readiness_report.components}
