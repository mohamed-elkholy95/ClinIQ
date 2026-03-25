"""Tests for the SDoH API route endpoints.

Covers:
* POST /sdoh — single document extraction
* POST /sdoh/batch — batch extraction
* GET /sdoh/domains — domain catalogue
* GET /sdoh/domains/{name} — domain detail
* GET /sdoh/z-codes — Z-code catalogue
* Error handling (empty text, unknown domain, validation)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

API_PREFIX = "/api/v1"


@pytest.fixture(autouse=True)
def _setup_client():
    global client
    client = TestClient(app)
    yield


# ---------------------------------------------------------------------------
# POST /sdoh
# ---------------------------------------------------------------------------


class TestPostSDoH:
    """POST /sdoh endpoint tests."""

    def test_single_extraction_success(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Patient is currently homeless."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "extractions" in data
        assert "domain_summary" in data
        assert "adverse_count" in data
        assert len(data["extractions"]) >= 1

    def test_extraction_fields(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Currently unemployed and uninsured."},
        )
        data = resp.json()
        ext = data["extractions"][0]
        assert "domain" in ext
        assert "text" in ext
        assert "sentiment" in ext
        assert "confidence" in ext
        assert "z_codes" in ext
        assert "trigger" in ext
        assert "negated" in ext

    def test_min_confidence_filtering(self) -> None:
        resp_low = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Patient is currently homeless.", "min_confidence": 0.1},
        )
        resp_high = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Patient is currently homeless.", "min_confidence": 0.99},
        )
        assert len(resp_low.json()["extractions"]) >= len(
            resp_high.json()["extractions"]
        )

    def test_z_codes_in_response(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Currently homeless, living in a shelter."},
        )
        data = resp.json()
        housing_exts = [
            e for e in data["extractions"] if e["domain"] == "housing"
        ]
        assert any(len(e["z_codes"]) > 0 for e in housing_exts)

    def test_substance_use_detection(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Current smoker, 1 PPD. Heavy drinking."},
        )
        data = resp.json()
        subs = [e for e in data["extractions"] if e["domain"] == "substance_use"]
        assert len(subs) >= 1

    def test_processing_time_present(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": "Patient is homeless."},
        )
        data = resp.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_empty_text_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh",
            json={"text": ""},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /sdoh/batch
# ---------------------------------------------------------------------------


class TestPostSDoHBatch:
    """POST /sdoh/batch endpoint tests."""

    def test_batch_success(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh/batch",
            json={
                "documents": [
                    {"text": "Currently homeless."},
                    {"text": "Unemployed and uninsured."},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 2
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_aggregate_counts(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh/batch",
            json={
                "documents": [
                    {"text": "Homeless."},
                    {"text": "Uninsured."},
                    {"text": "Non-smoker."},
                ]
            },
        )
        data = resp.json()
        assert data["total_extractions"] >= 3
        assert "aggregate_adverse" in data
        assert "aggregate_protective" in data

    def test_batch_preserves_order(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh/batch",
            json={
                "documents": [
                    {"text": "Currently employed full-time."},
                    {"text": "Currently homeless."},
                ]
            },
        )
        data = resp.json()
        # First result should have employment, second housing
        r0_domains = [e["domain"] for e in data["results"][0]["extractions"]]
        r1_domains = [e["domain"] for e in data["results"][1]["extractions"]]
        assert "employment" in r0_domains
        assert "housing" in r1_domains

    def test_batch_empty_list_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/sdoh/batch",
            json={"documents": []},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /sdoh/domains
# ---------------------------------------------------------------------------


class TestGetSDoHDomains:
    """GET /sdoh/domains endpoint tests."""

    def test_list_domains(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 8

    def test_domain_fields(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains")
        data = resp.json()
        for domain in data:
            assert "domain" in domain
            assert "description" in domain
            assert "z_codes" in domain
            assert "trigger_count" in domain
            assert "adverse_triggers" in domain
            assert "protective_triggers" in domain

    def test_domain_names(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains")
        data = resp.json()
        names = {d["domain"] for d in data}
        expected = {
            "housing", "employment", "education", "food_security",
            "transportation", "social_support", "substance_use", "financial",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# GET /sdoh/domains/{domain_name}
# ---------------------------------------------------------------------------


class TestGetSDoHDomainDetail:
    """GET /sdoh/domains/{domain_name} endpoint tests."""

    def test_valid_domain(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains/housing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "housing"
        assert data["trigger_count"] > 0

    def test_unknown_domain_404(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains/nonexistent")
        assert resp.status_code == 404

    def test_substance_use_domain(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/domains/substance_use")
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "substance_use"
        assert data["adverse_triggers"] > 0
        assert data["protective_triggers"] > 0


# ---------------------------------------------------------------------------
# GET /sdoh/z-codes
# ---------------------------------------------------------------------------


class TestGetZCodes:
    """GET /sdoh/z-codes endpoint tests."""

    def test_z_codes_list(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/z-codes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 10  # At least 10 Z-codes across domains

    def test_z_code_fields(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/z-codes")
        data = resp.json()
        for entry in data:
            assert "code" in entry
            assert "description" in entry
            assert "domain" in entry
            assert entry["code"].startswith("Z")

    def test_z_codes_cover_all_domains(self) -> None:
        resp = client.get(f"{API_PREFIX}/sdoh/z-codes")
        data = resp.json()
        domains_covered = {entry["domain"] for entry in data}
        assert len(domains_covered) == 8
