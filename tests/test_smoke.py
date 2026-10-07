from fastapi.testclient import TestClient

from embeda_api.main import app


def test_live() -> None:
    assert TestClient(app).get("/health/live").status_code == 200
