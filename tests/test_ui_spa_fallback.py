"""Verify the SPA fallback returns index.html for non-API routes so React
Router can handle deep-link URLs like /editor/<id>/<id>."""
from pathlib import Path

from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def _seed_static(tmp_path: Path) -> Path:
    static_dir = Path(__file__).resolve().parent.parent / "musicue" / "ui" / "static"
    if not static_dir.exists():
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "index.html").write_text(
            "<!DOCTYPE html><html><body><div id=root></div></body></html>"
        )
    return static_dir


def test_root_serves_spa(tmp_path):
    static = _seed_static(tmp_path)
    if not (static / "index.html").exists():
        return  # No frontend built; skip silently.
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<div id=" in r.text or "<div id=\"root\"" in r.text


def test_deep_link_serves_spa(tmp_path):
    static = _seed_static(tmp_path)
    if not (static / "index.html").exists():
        return
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/editor/abc/def")
    assert r.status_code == 200, r.text
    assert "<div id=" in r.text or "<div id=\"root\"" in r.text


def test_unknown_api_route_still_404(tmp_path):
    static = _seed_static(tmp_path)
    if not (static / "index.html").exists():
        return
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/this-route-does-not-exist")
    assert r.status_code == 404


def test_known_api_route_returns_json(tmp_path):
    """Sanity check: SPA fallback must NOT shadow real API routes."""
    static = _seed_static(tmp_path)
    if not (static / "index.html").exists():
        return
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/songs")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {"songs": []}
