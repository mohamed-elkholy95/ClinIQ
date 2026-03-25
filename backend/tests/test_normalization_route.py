"""Tests for the concept normalization API endpoints.

Covers:
- POST /normalize — single entity normalization
- POST /normalize/batch — batch normalization
- GET  /normalize/lookup/{cui} — CUI reverse lookup
- GET  /normalize/dictionary/stats — dictionary statistics
- Validation, error handling, edge cases
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# POST /normalize — single entity
# ---------------------------------------------------------------------------


class TestNormalizeSingle:
    """Test single-entity normalization endpoint."""

    def test_exact_match(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": "Hypertension"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["cui"] == "C0020538"
        assert data["preferred_term"] == "Hypertension"
        assert data["match_type"] == "exact"
        assert data["confidence"] == 1.0
        assert data["codes"]["snomed_ct"] == "38341003"

    def test_alias_match(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": "HTN"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["match_type"] == "alias"
        assert data["preferred_term"] == "Hypertension"

    def test_fuzzy_match(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": "hypertensoin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["match_type"] == "fuzzy"

    def test_no_match(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": "xyzzy_not_a_medical_term"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is False
        assert data["match_type"] == "none"

    def test_with_entity_type(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize",
            json={"text": "metformin", "entity_type": "MEDICATION"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["codes"]["rxnorm"] == "6809"

    def test_custom_min_similarity(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize",
            json={"text": "hypertensoin", "min_similarity": 0.99},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is False

    def test_fuzzy_disabled(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize",
            json={"text": "hypertensoin", "enable_fuzzy": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is False

    def test_response_has_codes_object(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": "Asthma"})
        assert resp.status_code == 200
        data = resp.json()
        assert "codes" in data
        assert "umls_cui" in data["codes"]
        assert "snomed_ct" in data["codes"]
        assert "rxnorm" in data["codes"]
        assert "icd10_cm" in data["codes"]
        assert "loinc" in data["codes"]

    def test_empty_text_rejected(self) -> None:
        resp = client.post(f"{API_PREFIX}/normalize", json={"text": ""})
        assert resp.status_code == 422

    def test_brand_name_normalization(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize",
            json={"text": "Lipitor", "entity_type": "MEDICATION"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["matched"] is True
        assert data["preferred_term"] == "Atorvastatin"


# ---------------------------------------------------------------------------
# POST /normalize/batch
# ---------------------------------------------------------------------------


class TestNormalizeBatch:
    """Test batch normalization endpoint."""

    def test_batch_success(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={
                "entities": [
                    {"text": "HTN", "entity_type": "DISEASE"},
                    {"text": "metformin", "entity_type": "MEDICATION"},
                    {"text": "unknown_xyz"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["summary"]["total"] == 3
        assert data["summary"]["matched"] == 2
        assert data["summary"]["unmatched"] == 1

    def test_batch_preserves_order(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={
                "entities": [
                    {"text": "Asthma"},
                    {"text": "COPD"},
                    {"text": "CHF"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        terms = [r["preferred_term"] for r in data["results"]]
        assert terms[0] == "Asthma"
        assert "Pulmonary" in terms[1]
        assert "Heart Failure" in terms[2]

    def test_batch_processing_time(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={"entities": [{"text": "HTN"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["processing_time_ms"] >= 0

    def test_batch_empty_text_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={"entities": [{"text": ""}]},
        )
        assert resp.status_code == 422

    def test_batch_missing_text_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={"entities": [{"entity_type": "DISEASE"}]},
        )
        assert resp.status_code == 422

    def test_batch_empty_list_rejected(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={"entities": []},
        )
        assert resp.status_code == 422

    def test_batch_summary_match_rate(self) -> None:
        resp = client.post(
            f"{API_PREFIX}/normalize/batch",
            json={
                "entities": [
                    {"text": "HTN"},
                    {"text": "DM"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["match_rate"] == 1.0


# ---------------------------------------------------------------------------
# GET /normalize/lookup/{cui}
# ---------------------------------------------------------------------------


class TestCUILookup:
    """Test CUI reverse-lookup endpoint."""

    def test_lookup_found(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/lookup/C0020538")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cui"] == "C0020538"
        assert data["preferred_term"] == "Hypertension"
        assert isinstance(data["aliases"], list)
        assert "htn" in data["aliases"]
        assert data["type_group"] == "CONDITION"

    def test_lookup_not_found(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/lookup/C9999999")
        assert resp.status_code == 404

    def test_lookup_has_codes(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/lookup/C0020538")
        assert resp.status_code == 200
        data = resp.json()
        assert data["codes"]["snomed_ct"] == "38341003"
        assert data["codes"]["icd10_cm"] == "I10"

    def test_lookup_medication(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/lookup/C0025598")
        assert resp.status_code == 200
        data = resp.json()
        assert data["preferred_term"] == "Metformin"
        assert data["codes"]["rxnorm"] == "6809"


# ---------------------------------------------------------------------------
# GET /normalize/dictionary/stats
# ---------------------------------------------------------------------------


class TestDictionaryStats:
    """Test dictionary statistics endpoint."""

    def test_stats_response(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/dictionary/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_concepts"] > 100
        assert data["total_aliases"] > 100
        assert "CONDITION" in data["by_type_group"]
        assert "MEDICATION" in data["by_type_group"]

    def test_stats_ontology_coverage(self) -> None:
        resp = client.get(f"{API_PREFIX}/normalize/dictionary/stats")
        assert resp.status_code == 200
        data = resp.json()
        coverage = data["ontology_coverage"]
        assert coverage["snomed_ct"] > 0
        assert coverage["rxnorm"] > 0
        assert coverage["icd10_cm"] > 0
        assert coverage["loinc"] > 0
