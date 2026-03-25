"""Tests for the Prometheus metrics endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes.metrics import router


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with only the metrics router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPrometheusMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_returns_200_status(self, client: TestClient) -> None:
        """Metrics endpoint should always return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_returns_valid_content_type(self, client: TestClient) -> None:
        """Should return text/plain (prometheus) or application/json (fallback)."""
        response = client.get("/metrics")
        content_type = response.headers["content-type"]
        assert "json" in content_type or "text/plain" in content_type

    def test_response_body_not_empty(self, client: TestClient) -> None:
        """Metrics response should have non-empty body."""
        response = client.get("/metrics")
        assert len(response.content) > 0


class TestModelMetricsSummaryEndpoint:
    """Tests for GET /metrics/models."""

    def test_returns_200(self, client: TestClient) -> None:
        """Model metrics summary should return 200."""
        response = client.get("/metrics/models")
        assert response.status_code == 200

    def test_returns_json_with_status(self, client: TestClient) -> None:
        """Response should contain status and models keys."""
        response = client.get("/metrics/models")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "models" in data

    def test_returns_prometheus_availability_flag(
        self, client: TestClient
    ) -> None:
        """Response should indicate whether Prometheus is available."""
        response = client.get("/metrics/models")
        data = response.json()
        assert "prometheus_available" in data
        assert isinstance(data["prometheus_available"], bool)

    def test_models_key_is_dict(self, client: TestClient) -> None:
        """The models key should be a dictionary."""
        response = client.get("/metrics/models")
        data = response.json()
        assert isinstance(data["models"], dict)
