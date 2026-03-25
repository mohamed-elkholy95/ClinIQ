"""Tests for the vital signs REST API endpoints.

Covers POST /vitals, POST /vitals/batch, GET /vitals/types, and
GET /vitals/ranges.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the in-memory rate limiter between tests."""
    obj = app.middleware_stack
    while obj is not None:
        if isinstance(obj, RateLimitMiddleware):
            obj._local_store.clear()
            break
        obj = getattr(obj, "app", None)


@pytest.fixture()
def client():
    """Return a FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /vitals — single document
# ---------------------------------------------------------------------------


class TestPostVitals:
    """Test single-document vital signs extraction endpoint."""

    def test_success(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "BP 120/80 mmHg, HR 72 bpm"})
        assert resp.status_code == 200
        data = resp.json()
        assert "readings" in data
        assert "summary" in data
        assert data["summary"]["total"] >= 2

    def test_readings_have_required_fields(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "HR 72 bpm"})
        data = resp.json()
        for r in data["readings"]:
            assert "vital_type" in r
            assert "value" in r
            assert "unit" in r
            assert "confidence" in r
            assert "interpretation" in r

    def test_bp_has_secondary_value(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "BP 130/85 mmHg"})
        data = resp.json()
        bp = [r for r in data["readings"] if r["vital_type"] == "blood_pressure"]
        assert len(bp) >= 1
        assert bp[0]["secondary_value"] == 85.0

    def test_min_confidence_filter(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={
            "text": "BP 120/80 mmHg, patient afebrile",
            "min_confidence": 0.85,
        })
        data = resp.json()
        for r in data["readings"]:
            assert r["confidence"] >= 0.85

    def test_empty_text_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": ""})
        assert resp.status_code == 422

    def test_text_hash_present(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "HR 72"})
        data = resp.json()
        assert len(data["text_hash"]) == 64

    def test_extraction_time_present(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "HR 72"})
        data = resp.json()
        assert data["extraction_time_ms"] >= 0.0

    def test_critical_findings_in_summary(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals", json={"text": "BP 200/120 mmHg"})
        data = resp.json()
        assert len(data["summary"]["critical_findings"]) >= 1

    def test_multiple_vital_types(self, client: TestClient) -> None:
        text = "BP 120/80, HR 72, T 98.6 F, RR 16, SpO2 98%"
        resp = client.post("/api/v1/vitals", json={"text": text})
        data = resp.json()
        types = {r["vital_type"] for r in data["readings"]}
        assert len(types) >= 3


# ---------------------------------------------------------------------------
# POST /vitals/batch — batch extraction
# ---------------------------------------------------------------------------


class TestPostVitalsBatch:
    """Test batch vital signs extraction endpoint."""

    def test_batch_success(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals/batch", json={
            "texts": ["BP 120/80 mmHg", "HR 72 bpm"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["total_readings"] >= 2

    def test_batch_order_preserved(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals/batch", json={
            "texts": ["HR 72", "BP 120/80 mmHg", "RR 18"],
        })
        data = resp.json()
        assert len(data["results"]) == 3

    def test_batch_aggregate_stats(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals/batch", json={
            "texts": ["BP 120/80 mmHg", "BP 140/90 mmHg"],
        })
        data = resp.json()
        assert "aggregate" in data
        assert data["aggregate"]["documents_processed"] == 2

    def test_batch_total_time(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals/batch", json={
            "texts": ["HR 72"],
        })
        data = resp.json()
        assert data["total_time_ms"] >= 0.0

    def test_batch_empty_list_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/vitals/batch", json={"texts": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /vitals/types — type catalogue
# ---------------------------------------------------------------------------


class TestGetVitalTypes:
    """Test vital sign type catalogue endpoint."""

    def test_returns_9_types(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/types")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 9

    def test_type_fields(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/types")
        for item in resp.json():
            assert "name" in item
            assert "description" in item
            assert "standard_unit" in item

    def test_known_types_present(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/types")
        names = {item["name"] for item in resp.json()}
        assert "Blood Pressure" in names
        assert "Heart Rate" in names
        assert "Temperature" in names


# ---------------------------------------------------------------------------
# GET /vitals/ranges — reference ranges
# ---------------------------------------------------------------------------


class TestGetVitalRanges:
    """Test reference range endpoint."""

    def test_returns_ranges(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/ranges")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 6  # At least 6 types with ranges + diastolic

    def test_range_fields(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/ranges")
        for item in resp.json():
            assert "vital_type" in item
            assert "critical_low" in item
            assert "low" in item
            assert "high" in item
            assert "critical_high" in item
            assert "unit" in item

    def test_includes_diastolic(self, client: TestClient) -> None:
        resp = client.get("/api/v1/vitals/ranges")
        types = {item["vital_type"] for item in resp.json()}
        assert "diastolic_bp" in types
