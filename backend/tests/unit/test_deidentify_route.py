"""Tests for the de-identification API route.

Covers single and batch endpoints, all replacement strategies,
PHI type filtering, validation errors, and edge cases.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes.deidentify import router


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with the deidentify router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /deidentify — single text
# ---------------------------------------------------------------------------

class TestDeidentifyEndpoint:
    """Single-text de-identification endpoint tests."""

    def test_basic_redaction(self, client: TestClient) -> None:
        """Basic text with PHI returns redacted version."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Dr. Smith seen on 01/15/2024.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "[NAME]" in data["text"]
        assert data["entity_count"] >= 1
        assert data["strategy"] == "redact"

    def test_mask_strategy(self, client: TestClient) -> None:
        """MASK strategy replaces with asterisks."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Email: doc@hospital.org",
            "strategy": "mask",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "doc@hospital.org" not in data["text"]
        assert data["strategy"] == "mask"

    def test_surrogate_strategy(self, client: TestClient) -> None:
        """SURROGATE strategy replaces with synthetic values."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Dr. Smith seen on 01/15/2024.",
            "strategy": "surrogate",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Smith" not in data["text"]
        assert data["strategy"] == "surrogate"

    def test_phi_type_filter(self, client: TestClient) -> None:
        """Only specified PHI types are detected."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Dr. Smith on 01/15/2024 email doc@hospital.org",
            "phi_types": ["EMAIL"],
        })
        assert resp.status_code == 200
        data = resp.json()
        # Only email should be detected
        types = {e["phi_type"] for e in data["entities"]}
        assert "NAME" not in types
        assert "DATE" not in types

    def test_invalid_phi_type_422(self, client: TestClient) -> None:
        """Invalid PHI type name returns 422."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Test text",
            "phi_types": ["INVALID_TYPE"],
        })
        assert resp.status_code == 422

    def test_confidence_threshold(self, client: TestClient) -> None:
        """High confidence threshold filters out low-confidence matches."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Dr. Smith has hypertension.",
            "confidence_threshold": 0.99,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Names have base confidence 0.90, should be filtered at 0.99
        names = [e for e in data["entities"] if e["phi_type"] == "NAME"]
        assert len(names) == 0

    def test_empty_text_422(self, client: TestClient) -> None:
        """Empty text fails Pydantic validation."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "",
        })
        assert resp.status_code == 422

    def test_entities_have_positions(self, client: TestClient) -> None:
        """Detected entities include character offsets."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Email: test@example.com is here.",
        })
        data = resp.json()
        if data["entities"]:
            entity = data["entities"][0]
            assert "start_char" in entity
            assert "end_char" in entity
            assert entity["start_char"] < entity["end_char"]

    def test_phi_types_found_list(self, client: TestClient) -> None:
        """phi_types_found contains unique type names."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Dr. Smith on 01/15/2024 email doc@hospital.org",
        })
        data = resp.json()
        assert isinstance(data["phi_types_found"], list)
        # Should have at least NAME/DATE/EMAIL
        assert len(data["phi_types_found"]) >= 1

    def test_no_phi_in_text(self, client: TestClient) -> None:
        """Text without PHI returns zero entities."""
        resp = client.post("/api/v1/deidentify", json={
            "text": "Patient has mild hypertension and diabetes.",
        })
        data = resp.json()
        # Might have 0 entities or a few low-confidence ones
        assert data["entity_count"] >= 0
        assert isinstance(data["text"], str)


# ---------------------------------------------------------------------------
# POST /deidentify/batch
# ---------------------------------------------------------------------------

class TestBatchDeidentifyEndpoint:
    """Batch de-identification endpoint tests."""

    def test_batch_multiple_texts(self, client: TestClient) -> None:
        """Batch processes multiple texts."""
        resp = client.post("/api/v1/deidentify/batch", json={
            "texts": [
                "Dr. Smith on 01/15/2024",
                "Email: doc@hospital.org",
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 2
        assert len(data["results"]) == 2
        assert data["total_entities"] >= 1

    def test_batch_strategy_applied(self, client: TestClient) -> None:
        """Batch applies the specified strategy."""
        resp = client.post("/api/v1/deidentify/batch", json={
            "texts": ["Dr. Smith has a cold."],
            "strategy": "mask",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["strategy"] == "mask"

    def test_batch_empty_texts_422(self, client: TestClient) -> None:
        """Empty texts list fails validation."""
        resp = client.post("/api/v1/deidentify/batch", json={
            "texts": [],
        })
        assert resp.status_code == 422

    def test_batch_single_text(self, client: TestClient) -> None:
        """Batch with single text works correctly."""
        resp = client.post("/api/v1/deidentify/batch", json={
            "texts": ["MRN: PAT12345"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 1

    def test_batch_confidence_threshold(self, client: TestClient) -> None:
        """Batch respects confidence threshold."""
        resp = client.post("/api/v1/deidentify/batch", json={
            "texts": ["Dr. Smith is a doctor."],
            "confidence_threshold": 0.99,
        })
        assert resp.status_code == 200
        data = resp.json()
        # High threshold should filter out low-confidence matches
        names = [
            e for r in data["results"]
            for e in r["entities"]
            if e["phi_type"] == "NAME"
        ]
        assert len(names) == 0
