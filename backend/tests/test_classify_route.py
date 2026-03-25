"""Tests for the document classification API endpoints.

Covers POST /classify, POST /classify/batch, and GET /classify/types
with validation, happy paths, and error edge cases.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes.classify import router


@pytest.fixture
def client() -> TestClient:
    """Create a test client with the classify router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def discharge_text() -> str:
    """Minimal discharge summary text."""
    return (
        "DISCHARGE SUMMARY\n"
        "Admission Date: 03/20/2026\n"
        "Discharge Date: 03/25/2026\n"
        "Discharge Diagnosis: Pneumonia\n"
        "Hospital Course: Patient treated with antibiotics and improved.\n"
        "Discharge Medications: Amoxicillin 500mg PO TID\n"
        "Follow-up Instructions: PCP in 1 week\n"
    )


@pytest.fixture
def operative_text() -> str:
    """Minimal operative note text."""
    return (
        "OPERATIVE NOTE\n"
        "Pre-operative diagnosis: Appendicitis\n"
        "Post-operative diagnosis: Appendicitis\n"
        "Procedure performed: Laparoscopic appendectomy\n"
        "Anesthesia: General\n"
        "Estimated blood loss: 20mL\n"
        "Specimens sent: Appendix to pathology\n"
    )


class TestClassifyEndpoint:
    """Tests for POST /classify."""

    def test_classify_success(self, client: TestClient, discharge_text: str) -> None:
        """Successful classification returns predicted type and scores."""
        resp = client.post("/classify", json={"text": discharge_text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_type"] == "discharge_summary"
        assert len(data["scores"]) > 0
        assert data["processing_time_ms"] >= 0
        assert "classifier_version" in data

    def test_classify_with_top_k(self, client: TestClient, discharge_text: str) -> None:
        """top_k limits the number of returned scores."""
        resp = client.post("/classify", json={"text": discharge_text, "top_k": 2})
        assert resp.status_code == 200
        assert len(resp.json()["scores"]) <= 2

    def test_classify_with_min_confidence(self, client: TestClient, discharge_text: str) -> None:
        """min_confidence filters low-scoring types."""
        resp = client.post(
            "/classify",
            json={"text": discharge_text, "min_confidence": 0.5},
        )
        assert resp.status_code == 200
        for score in resp.json()["scores"]:
            assert score["confidence"] >= 0.5

    def test_classify_validation_short_text(self, client: TestClient) -> None:
        """Text shorter than 10 characters is rejected."""
        resp = client.post("/classify", json={"text": "short"})
        assert resp.status_code == 422

    def test_classify_validation_missing_text(self, client: TestClient) -> None:
        """Missing text field is rejected."""
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_classify_scores_have_evidence(self, client: TestClient, discharge_text: str) -> None:
        """Each score includes evidence strings."""
        resp = client.post("/classify", json={"text": discharge_text})
        data = resp.json()
        top_score = data["scores"][0]
        assert "evidence" in top_score
        assert isinstance(top_score["evidence"], list)


class TestClassifyBatchEndpoint:
    """Tests for POST /classify/batch."""

    def test_batch_success(
        self, client: TestClient, discharge_text: str, operative_text: str,
    ) -> None:
        """Batch endpoint returns results for all documents."""
        resp = client.post(
            "/classify/batch",
            json={"documents": [discharge_text, operative_text]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 2
        assert len(data["results"]) == 2
        assert data["total_processing_time_ms"] >= 0

    def test_batch_preserves_order(
        self, client: TestClient, discharge_text: str, operative_text: str,
    ) -> None:
        """Results are in the same order as input documents."""
        resp = client.post(
            "/classify/batch",
            json={"documents": [discharge_text, operative_text]},
        )
        results = resp.json()["results"]
        assert results[0]["predicted_type"] == "discharge_summary"
        assert results[1]["predicted_type"] == "operative_note"

    def test_batch_validation_empty(self, client: TestClient) -> None:
        """Empty document list is rejected."""
        resp = client.post("/classify/batch", json={"documents": []})
        assert resp.status_code == 422

    def test_batch_with_top_k(
        self, client: TestClient, discharge_text: str,
    ) -> None:
        """Batch respects top_k parameter."""
        resp = client.post(
            "/classify/batch",
            json={"documents": [discharge_text], "top_k": 1},
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"][0]["scores"]) <= 1


class TestDocumentTypesEndpoint:
    """Tests for GET /classify/types."""

    def test_list_types(self, client: TestClient) -> None:
        """Returns catalogue of document types."""
        resp = client.get("/classify/types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 13  # All types except UNKNOWN
        assert len(data["types"]) == 13

    def test_type_has_description(self, client: TestClient) -> None:
        """Each type entry has type and description fields."""
        resp = client.get("/classify/types")
        for t in resp.json()["types"]:
            assert "type" in t
            assert "description" in t
            assert len(t["description"]) > 0

    def test_unknown_not_listed(self, client: TestClient) -> None:
        """UNKNOWN type is not in the catalogue."""
        resp = client.get("/classify/types")
        types = [t["type"] for t in resp.json()["types"]]
        assert "unknown" not in types
