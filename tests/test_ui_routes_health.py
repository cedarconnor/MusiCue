from fastapi.testclient import TestClient

from musicue.ui.server import create_app


def test_health(tmp_path):
    app = create_app(storage_root=tmp_path)
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
