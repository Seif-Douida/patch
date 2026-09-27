"""GET /health: liveness only. It must never touch SQL or other dependencies (spec §13)."""

import pytest
from fastapi.testclient import TestClient

from patchpulse.api.app import app


def test_health_reports_ok_and_image_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_VERSION", "abc123")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "abc123"}


def test_health_version_defaults_to_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_VERSION", raising=False)
    assert TestClient(app).get("/health").json()["version"] == "dev"


def test_api_docs_are_not_exposed() -> None:
    client = TestClient(app)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
