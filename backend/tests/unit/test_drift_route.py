"""Tests for drift monitoring endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes.drift import router


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with only the drift router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestDriftStatusEndpoint:
    """Tests for GET /drift/status."""

    def test_returns_200(self, client: TestClient) -> None:
        """Drift status should always return 200."""
        response = client.get("/drift/status")
        assert response.status_code == 200

    def test_contains_overall_status(self, client: TestClient) -> None:
        """Response should have an overall_status key."""
        data = client.get("/drift/status").json()
        assert "overall_status" in data
        assert data["overall_status"] in {"stable", "warning", "drifted"}

    def test_contains_text_distribution(self, client: TestClient) -> None:
        """Response should have text_distribution metrics."""
        data = client.get("/drift/status").json()
        assert "text_distribution" in data

    def test_contains_prediction_drift(self, client: TestClient) -> None:
        """Response should have prediction_drift metrics."""
        data = client.get("/drift/status").json()
        assert "prediction_drift" in data

    def test_stable_when_no_data(self, client: TestClient) -> None:
        """With no recorded data, status should be stable."""
        data = client.get("/drift/status").json()
        assert data["overall_status"] == "stable"


class TestRecordPredictionEndpoint:
    """Tests for POST /drift/record."""

    def test_record_returns_200(self, client: TestClient) -> None:
        """Recording a prediction should return 200."""
        response = client.post(
            "/drift/record",
            params={
                "model_name": "ner-v1",
                "predicted_label": "disease",
                "confidence": 0.95,
            },
        )
        assert response.status_code == 200

    def test_record_returns_status_recorded(self, client: TestClient) -> None:
        """Response should confirm the record was captured."""
        response = client.post(
            "/drift/record",
            params={
                "model_name": "icd-v1",
                "predicted_label": "E11.9",
                "confidence": 0.87,
            },
        )
        assert response.json() == {"status": "recorded"}

    def test_record_with_text(self, client: TestClient) -> None:
        """Recording with optional text should also succeed."""
        response = client.post(
            "/drift/record",
            params={
                "model_name": "ner-v1",
                "predicted_label": "medication",
                "confidence": 0.92,
                "text": "Patient takes metformin 1000mg daily.",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"

    def test_record_without_text(self, client: TestClient) -> None:
        """Recording without text should work (text is optional)."""
        response = client.post(
            "/drift/record",
            params={
                "model_name": "summarizer-v1",
                "predicted_label": "standard",
                "confidence": 0.88,
            },
        )
        assert response.status_code == 200

    def test_record_low_confidence(self, client: TestClient) -> None:
        """Low confidence values should be accepted."""
        response = client.post(
            "/drift/record",
            params={
                "model_name": "risk-v1",
                "predicted_label": "high",
                "confidence": 0.12,
            },
        )
        assert response.status_code == 200
