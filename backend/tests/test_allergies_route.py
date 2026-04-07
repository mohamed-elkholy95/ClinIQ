"""Tests for the allergy extraction API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

API_PREFIX = "/api/v1"


class TestPostAllergies:
    """Tests for POST /allergies."""

    def test_extract_single_allergy(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Allergies: Penicillin - anaphylaxis"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["allergy_count"] >= 1
        assert any(a["allergen"] == "penicillin" for a in data["allergies"])

    def test_extract_multiple_allergies(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Allergies: PCN, sulfa, codeine, latex"},
        )
        data = resp.json()
        assert data["allergy_count"] >= 4

    def test_nkda_detection(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Allergies: NKDA"},
        )
        data = resp.json()
        assert data["no_known_allergies"] is True

    def test_min_confidence_filtering(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Patient mentioned aspirin.", "min_confidence": 0.99},
        )
        data = resp.json()
        assert data["allergy_count"] == 0

    def test_reaction_included(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Penicillin causes rash."},
        )
        data = resp.json()
        pcn = next(a for a in data["allergies"] if a["allergen"] == "penicillin")
        assert len(pcn["reactions"]) >= 1

    def test_severity_included(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Penicillin: anaphylaxis"},
        )
        data = resp.json()
        pcn = next(a for a in data["allergies"] if a["allergen"] == "penicillin")
        assert pcn["severity"] == "life_threatening"

    def test_category_in_response(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": "Allergic to peanuts."},
        )
        data = resp.json()
        assert "food" in data["categories"]

    def test_empty_text_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies",
            json={"text": ""},
        )
        assert resp.status_code == 422


class TestPostAllergiesBatch:
    """Tests for POST /allergies/batch."""

    def test_batch_extraction(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies/batch",
            json={
                "texts": [
                    "Allergies: PCN",
                    "NKDA",
                    "Allergic to peanuts and shellfish.",
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 3
        assert data["total_allergies"] >= 3
        assert data["nkda_count"] >= 1

    def test_batch_category_breakdown(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies/batch",
            json={
                "texts": [
                    "Allergic to penicillin.",
                    "Allergic to peanuts.",
                ],
            },
        )
        data = resp.json()
        assert "drug" in data["category_breakdown"]
        assert "food" in data["category_breakdown"]

    def test_batch_preserves_order(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/allergies/batch",
            json={"texts": ["Allergic to aspirin.", "NKDA"]},
        )
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["allergy_count"] >= 1
        assert data["results"][1]["no_known_allergies"] is True


class TestGetDictionaryStats:
    """Tests for GET /allergies/dictionary/stats."""

    def test_returns_stats(self) -> None:
        resp = client.get(f"{API_PREFIX}/allergies/dictionary/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_allergens"] >= 90
        assert "drug" in data["by_category"]
        assert data["reaction_count"] >= 30


class TestGetAllergyCategories:
    """Tests for GET /allergies/categories."""

    def test_returns_three_categories(self) -> None:
        resp = client.get(f"{API_PREFIX}/allergies/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        names = [c["category"] for c in data["categories"]]
        assert "drug" in names
        assert "food" in names
        assert "environmental" in names

    def test_category_has_description(self) -> None:
        resp = client.get(f"{API_PREFIX}/allergies/categories")
        data = resp.json()
        for cat in data["categories"]:
            assert "description" in cat
            assert len(cat["description"]) > 10
