"""Tests for the relations API endpoint.

Covers:
- POST /relations — happy path, relation type filtering, min_confidence,
  validation errors, empty results
- GET /relations/types — catalogue response
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


API_PREFIX = "/api/v1"


def _make_request(
    text: str,
    entities: list[dict],
    **kwargs,
) -> dict:
    """Build a relation extraction request body."""
    body = {"text": text, "entities": entities, **kwargs}
    return body


class TestRelationsEndpoint:
    """Tests for POST /relations."""

    def test_happy_path(self):
        body = _make_request(
            text="Patient is on metformin for diabetes mellitus",
            entities=[
                {"text": "metformin", "entity_type": "MEDICATION", "start_char": 14, "end_char": 23},
                {"text": "diabetes mellitus", "entity_type": "DISEASE", "start_char": 28, "end_char": 45},
            ],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] == 2
        assert data["pair_count"] >= 1
        assert data["relation_count"] >= 1
        assert len(data["relations"]) >= 1
        assert data["model_name"] == "rule-based-relations"

    def test_relation_type_filtering(self):
        body = _make_request(
            text="Metformin for diabetes management",
            entities=[
                {"text": "Metformin", "entity_type": "MEDICATION", "start_char": 0, "end_char": 9},
                {"text": "diabetes", "entity_type": "DISEASE", "start_char": 14, "end_char": 22},
            ],
            relation_types=["treats"],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        data = resp.json()
        for rel in data["relations"]:
            assert rel["relation_type"] == "treats"

    def test_filter_nonexistent_type_returns_empty(self):
        body = _make_request(
            text="Metformin for diabetes",
            entities=[
                {"text": "Metformin", "entity_type": "MEDICATION", "start_char": 0, "end_char": 9},
                {"text": "diabetes", "entity_type": "DISEASE", "start_char": 14, "end_char": 22},
            ],
            relation_types=["nonexistent_type"],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        assert resp.json()["relation_count"] == 0

    def test_min_confidence_high_filters_all(self):
        body = _make_request(
            text="Metformin was started several months ago and then used to manage the patient's diabetes",
            entities=[
                {"text": "Metformin", "entity_type": "MEDICATION", "start_char": 0, "end_char": 9},
                {"text": "diabetes", "entity_type": "DISEASE", "start_char": 78, "end_char": 86},
            ],
            min_confidence=0.99,
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        assert resp.json()["relation_count"] == 0

    def test_incompatible_entities_no_relations(self):
        body = _make_request(
            text="500mg was increased to 750mg daily",
            entities=[
                {"text": "500mg", "entity_type": "DOSAGE", "start_char": 0, "end_char": 5},
                {"text": "750mg", "entity_type": "DOSAGE", "start_char": 22, "end_char": 27},
            ],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        assert resp.json()["relation_count"] == 0

    def test_validation_empty_text(self):
        body = _make_request(
            text="",
            entities=[
                {"text": "x", "entity_type": "DISEASE", "start_char": 0, "end_char": 1},
                {"text": "y", "entity_type": "DISEASE", "start_char": 2, "end_char": 3},
            ],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 422

    def test_validation_single_entity(self):
        body = _make_request(
            text="Diabetes is present",
            entities=[
                {"text": "Diabetes", "entity_type": "DISEASE", "start_char": 0, "end_char": 8},
            ],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 422  # min 2 entities required

    def test_processing_time_present(self):
        body = _make_request(
            text="Lisinopril for hypertension control",
            entities=[
                {"text": "Lisinopril", "entity_type": "MEDICATION", "start_char": 0, "end_char": 10},
                {"text": "hypertension", "entity_type": "DISEASE", "start_char": 15, "end_char": 27},
            ],
        )
        resp = client.post(f"{API_PREFIX}/relations", json=body)
        assert resp.status_code == 200
        assert resp.json()["processing_time_ms"] >= 0


class TestRelationTypesEndpoint:
    """Tests for GET /relations/types."""

    def test_list_types(self):
        resp = client.get(f"{API_PREFIX}/relations/types")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 12
        assert "treats" in data["relation_types"]
        assert "causes" in data["relation_types"]
        assert "diagnoses" in data["relation_types"]
