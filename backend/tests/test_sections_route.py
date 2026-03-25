"""Tests for the section parsing API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

API_PREFIX = "/api/v1"


class TestPostSections:
    """Tests for POST /sections."""

    def test_parse_basic_note(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections",
            json={"text": "CHIEF COMPLAINT:\nChest pain\n\nHPI:\nAcute onset"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["section_count"] >= 2
        assert len(data["sections"]) >= 2

    def test_returns_categories(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections",
            json={"text": "MEDICATIONS:\nAspirin\n\nALLERGIES:\nNKDA"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "medications" in data["categories_found"]
        assert "allergies" in data["categories_found"]

    def test_min_confidence_filtering(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections",
            json={
                "text": "ASSESSMENT\nHTN",  # ALL-CAPS no colon = 0.85
                "min_confidence": 0.90,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["section_count"] == 0

    def test_empty_text_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections",
            json={"text": ""},
        )
        assert resp.status_code == 422

    def test_section_offsets(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections",
            json={"text": "PLAN:\nStart aspirin"},
        )
        data = resp.json()
        section = data["sections"][0]
        assert "header_start" in section
        assert "header_end" in section
        assert "body_end" in section
        assert section["header_start"] < section["header_end"] <= section["body_end"]


class TestPostSectionsBatch:
    """Tests for POST /sections/batch."""

    def test_batch_parsing(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections/batch",
            json={
                "texts": [
                    "CHIEF COMPLAINT:\nPain",
                    "MEDICATIONS:\nAspirin\n\nPLAN:\nContinue meds",
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 2
        assert data["total_sections"] >= 3
        assert len(data["results"]) == 2

    def test_batch_aggregate_stats(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections/batch",
            json={"texts": ["PLAN:\nX", "PLAN:\nY"]},
        )
        data = resp.json()
        assert data["avg_sections"] >= 1.0
        assert "plan" in data["all_categories_found"]


class TestPostSectionsQuery:
    """Tests for POST /sections/query."""

    def test_position_in_section(self) -> None:
        text = "MEDICATIONS:\nAspirin 81mg"
        pos = text.index("Aspirin")
        resp = client.post(
            f"{API_PREFIX}/sections/query",
            json={"text": text, "position": pos},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["in_section"] is True
        assert data["section"]["category"] == "medications"

    def test_position_out_of_bounds(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sections/query",
            json={"text": "Hello", "position": 100},
        )
        assert resp.status_code == 422

    def test_position_in_preamble(self) -> None:
        text = "Preamble text\n\nCHIEF COMPLAINT:\nPain"
        resp = client.post(
            f"{API_PREFIX}/sections/query",
            json={"text": text, "position": 3},
        )
        data = resp.json()
        assert data["in_section"] is False
        assert data["section"] is None


class TestGetCategories:
    """Tests for GET /sections/categories."""

    def test_returns_categories(self) -> None:
        resp = client.get(f"{API_PREFIX}/sections/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 30
        assert len(data["categories"]) >= 30

    def test_category_has_description(self) -> None:
        resp = client.get(f"{API_PREFIX}/sections/categories")
        data = resp.json()
        for cat in data["categories"]:
            assert "category" in cat
            assert "description" in cat
            assert len(cat["description"]) > 5

    def test_unknown_excluded(self) -> None:
        resp = client.get(f"{API_PREFIX}/sections/categories")
        data = resp.json()
        categories = [c["category"] for c in data["categories"]]
        assert "unknown" not in categories
