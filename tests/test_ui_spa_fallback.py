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


def test_encoded_traversal_does_not_escape_static_dir(tmp_path):
    """Encoded ../ in the path must not let a request read source files
    outside static_dir. Ext-review-2026-05-09 reported this as exploitable."""
    static = _seed_static(tmp_path)
    if not (static / "index.html").exists():
        return
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)

    # All of these previously returned the requested source file's contents.
    # Now they must fall through to index.html (the SPA fallback) -- i.e.
    # status 200 but with the SPA HTML, not the source file's bytes.
    suspicious_paths = [
        "/%2e%2e/server.py",
        "/%2e%2e/%2e%2e/cli.py",
        "/%2e%2e/%2e%2e/%2e%2e/README.md",
        "/../server.py",
    ]
    for path in suspicious_paths:
        r = client.get(path)
        # Encoded traversal: status 200 SPA fallback. Plain ../: starlette
        # normalises before the route matches, also winding up at SPA.
        assert r.status_code == 200, f"unexpected status for {path}: {r.status_code}"
        body = r.text
        assert "<div id=" in body or "<!DOCTYPE" in body, (
            f"path {path} returned non-SPA content: {body[:200]!r}"
        )
        assert "import" not in body[:500], (
            f"path {path} appears to leak Python source: {body[:500]!r}"
        )
