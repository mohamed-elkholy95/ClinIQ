"""Tests for the medication extraction API endpoints.

Covers POST /medications, POST /medications/batch,
GET /medications/lookup/{drug_name}, and GET /medications/dictionary/stats.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes.medications import router


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with the medications router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# =========================================================================
# POST /medications — single extraction
# =========================================================================


class TestExtractMedications:
    """Test POST /medications endpoint."""

    def test_success(self, client: TestClient) -> None:
        """Should extract medications from clinical text."""
        response = client.post(
            "/medications",
            json={"text": "Metformin 500mg PO BID for diabetes"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["medication_count"] >= 1
        assert len(data["medications"]) >= 1
        assert data["medications"][0]["drug_name"].lower() == "metformin"

    def test_multiple_medications(self, client: TestClient) -> None:
        """Should extract multiple medications."""
        response = client.post(
            "/medications",
            json={
                "text": "Medications: lisinopril 10mg daily, metformin 500mg BID, atorvastatin 40mg at bedtime."
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["medication_count"] >= 3
        assert data["unique_drugs"] >= 3

    def test_min_confidence_filtering(self, client: TestClient) -> None:
        """Should respect min_confidence parameter."""
        response = client.post(
            "/medications",
            json={
                "text": "metformin daily",
                "min_confidence": 0.9,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # High threshold filters low-evidence mentions
        for med in data["medications"]:
            assert med["confidence"] >= 0.9

    def test_include_generics_false(self, client: TestClient) -> None:
        """Should suppress generic names when include_generics=False."""
        response = client.post(
            "/medications",
            json={
                "text": "Lipitor 20mg daily",
                "include_generics": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        if data["medications"]:
            assert data["medications"][0]["generic_name"] is None

    def test_empty_text_validation(self, client: TestClient) -> None:
        """Should reject empty text."""
        response = client.post(
            "/medications",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_processing_time_present(self, client: TestClient) -> None:
        """Response should include processing time."""
        response = client.post(
            "/medications",
            json={"text": "metformin 500mg PO BID"},
        )
        assert response.status_code == 200
        assert "processing_time_ms" in response.json()

    def test_extractor_version_present(self, client: TestClient) -> None:
        """Response should include extractor version."""
        response = client.post(
            "/medications",
            json={"text": "metformin 500mg PO BID"},
        )
        assert response.status_code == 200
        assert response.json()["extractor_version"] == "1.0.0"

    def test_dosage_structure(self, client: TestClient) -> None:
        """Dosage should be structured with value and unit."""
        response = client.post(
            "/medications",
            json={"text": "metformin 500mg PO BID"},
        )
        data = response.json()
        if data["medications"] and data["medications"][0]["dosage"]:
            dosage = data["medications"][0]["dosage"]
            assert "value" in dosage
            assert "unit" in dosage


# =========================================================================
# POST /medications/batch
# =========================================================================


class TestBatchExtraction:
    """Test POST /medications/batch endpoint."""

    def test_batch_success(self, client: TestClient) -> None:
        """Should process multiple texts."""
        response = client.post(
            "/medications/batch",
            json={
                "texts": [
                    "metformin 500mg PO BID",
                    "lisinopril 10mg daily",
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total_medications"] >= 2

    def test_batch_preserves_order(self, client: TestClient) -> None:
        """Results should match input order."""
        response = client.post(
            "/medications/batch",
            json={
                "texts": [
                    "No medications.",
                    "metformin 500mg PO BID",
                ],
            },
        )
        data = response.json()
        assert data["results"][0]["medication_count"] == 0
        assert data["results"][1]["medication_count"] >= 1

    def test_batch_empty_texts_validation(self, client: TestClient) -> None:
        """Should reject empty texts list."""
        response = client.post(
            "/medications/batch",
            json={"texts": []},
        )
        assert response.status_code == 422

    def test_batch_processing_time(self, client: TestClient) -> None:
        """Batch response should include total processing time."""
        response = client.post(
            "/medications/batch",
            json={"texts": ["metformin 500mg"]},
        )
        assert response.status_code == 200
        assert "processing_time_ms" in response.json()


# =========================================================================
# GET /medications/lookup/{drug_name}
# =========================================================================


class TestDrugLookup:
    """Test GET /medications/lookup/{drug_name} endpoint."""

    def test_lookup_generic(self, client: TestClient) -> None:
        """Should find a generic drug name."""
        response = client.get("/medications/lookup/metformin")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["generic_name"] == "metformin"

    def test_lookup_brand(self, client: TestClient) -> None:
        """Should find a brand name and return generic."""
        response = client.get("/medications/lookup/Lipitor")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["generic_name"] == "atorvastatin"
        assert "lipitor" in data["brand_names"]

    def test_lookup_not_found(self, client: TestClient) -> None:
        """Should handle unknown drugs gracefully."""
        response = client.get("/medications/lookup/unknowndrug123")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is False
        assert data["generic_name"] is None

    def test_lookup_returns_brand_names(self, client: TestClient) -> None:
        """Should return all brand names for a generic."""
        response = client.get("/medications/lookup/atorvastatin")
        data = response.json()
        assert data["found"] is True
        assert "lipitor" in data["brand_names"]


# =========================================================================
# GET /medications/dictionary/stats
# =========================================================================


class TestDictionaryStats:
    """Test GET /medications/dictionary/stats endpoint."""

    def test_stats_response(self, client: TestClient) -> None:
        """Should return dictionary statistics."""
        response = client.get("/medications/dictionary/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_entries"] >= 200
        assert data["unique_generics"] >= 50
        assert len(data["routes"]) >= 10
        assert len(data["statuses"]) >= 5

    def test_routes_excludes_unknown(self, client: TestClient) -> None:
        """Routes list should not include 'unknown'."""
        response = client.get("/medications/dictionary/stats")
        data = response.json()
        assert "unknown" not in data["routes"]

    def test_statuses_excludes_unknown(self, client: TestClient) -> None:
        """Statuses list should not include 'unknown'."""
        response = client.get("/medications/dictionary/stats")
        data = response.json()
        assert "unknown" not in data["statuses"]
