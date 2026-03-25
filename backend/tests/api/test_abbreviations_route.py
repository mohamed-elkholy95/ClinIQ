"""Tests for clinical abbreviation expansion API endpoints.

Covers POST /abbreviations, POST /abbreviations/batch,
GET /abbreviations/lookup/{abbreviation},
GET /abbreviations/dictionary/stats, and GET /abbreviations/domains.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture()
def client():
    """Create a test client."""
    from starlette.testclient import TestClient
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────
# POST /abbreviations
# ─────────────────────────────────────────────────────────────────────


class TestExpandAbbreviations:
    """Test POST /abbreviations endpoint."""

    def test_expand_success(self, client):
        """Successful abbreviation expansion."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "PMH: HTN, DM2, CAD."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_found"] >= 3
        assert data["original_text"] == "PMH: HTN, DM2, CAD."
        assert "expanded_text" in data

    def test_expand_with_ambiguous(self, client):
        """Ambiguous abbreviation is detected."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "PE: Heart regular, lungs clear, abdomen soft."},
        )
        assert resp.status_code == 200
        data = resp.json()
        pe_matches = [m for m in data["matches"] if m["abbreviation"].upper() == "PE"]
        assert len(pe_matches) >= 1

    def test_expand_min_confidence(self, client):
        """Confidence threshold filters low-confidence matches."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "PMH: HTN, OR, US.", "min_confidence": 0.95},
        )
        assert resp.status_code == 200
        for m in resp.json()["matches"]:
            assert m["confidence"] >= 0.95

    def test_expand_no_expand_in_place(self, client):
        """expand_in_place=False returns original text."""
        text = "PMH: HTN."
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": text, "expand_in_place": False},
        )
        assert resp.status_code == 200
        assert resp.json()["expanded_text"] == text

    def test_expand_domain_filter(self, client):
        """Domain filter restricts results."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "SRP, BOP, HTN, DM2.", "domains": ["dental"]},
        )
        assert resp.status_code == 200
        for m in resp.json()["matches"]:
            assert m["domain"] == "dental"

    def test_expand_empty_text_rejected(self, client):
        """Empty text is rejected by validation."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": ""},
        )
        assert resp.status_code == 422

    def test_expand_match_has_offsets(self, client):
        """Each match includes start and end offsets."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "HTN noted."},
        )
        assert resp.status_code == 200
        for m in resp.json()["matches"]:
            assert "start" in m
            assert "end" in m
            assert m["end"] > m["start"]

    def test_expand_processing_time(self, client):
        """Response includes processing time."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "HTN, DM2."},
        )
        assert resp.status_code == 200
        assert resp.json()["processing_time_ms"] >= 0

    def test_expand_ambiguous_count(self, client):
        """Ambiguous count reflects actual ambiguous matches."""
        resp = client.post(
            "/api/v1/abbreviations",
            json={"text": "PE noted. Concern for PE with DVT and clot."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ambiguous_count"] >= 0
        ambiguous_matches = [m for m in data["matches"] if m["is_ambiguous"]]
        assert len(ambiguous_matches) == data["ambiguous_count"]


# ─────────────────────────────────────────────────────────────────────
# POST /abbreviations/batch
# ─────────────────────────────────────────────────────────────────────


class TestBatchExpand:
    """Test POST /abbreviations/batch endpoint."""

    def test_batch_success(self, client):
        """Batch expansion with multiple documents."""
        resp = client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": ["HTN noted.", "DM2 stable.", "COPD exacerbation."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_documents"] == 3
        assert len(data["results"]) == 3

    def test_batch_order_preserved(self, client):
        """Results maintain input order."""
        resp = client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": ["HTN", "DM2", "COPD"]},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results[0]["original_text"] == "HTN"
        assert results[1]["original_text"] == "DM2"
        assert results[2]["original_text"] == "COPD"

    def test_batch_aggregate_stats(self, client):
        """Batch response includes aggregate abbreviation count."""
        resp = client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": ["HTN, DM2.", "CAD, COPD."]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_abbreviations"] >= 4

    def test_batch_empty_list_rejected(self, client):
        """Empty list is rejected."""
        resp = client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": []},
        )
        assert resp.status_code == 422

    def test_batch_with_confidence(self, client):
        """Batch respects min_confidence."""
        resp = client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": ["HTN, OR."], "min_confidence": 0.95},
        )
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            for m in result["matches"]:
                assert m["confidence"] >= 0.95


# ─────────────────────────────────────────────────────────────────────
# GET /abbreviations/lookup/{abbreviation}
# ─────────────────────────────────────────────────────────────────────


class TestLookup:
    """Test GET /abbreviations/lookup/{abbreviation} endpoint."""

    def test_lookup_found(self, client):
        """Known abbreviation returns details."""
        resp = client.get("/api/v1/abbreviations/lookup/HTN")
        assert resp.status_code == 200
        data = resp.json()
        assert data["abbreviation"] == "HTN"
        assert data["expansion"] == "hypertension"
        assert data["is_ambiguous"] is False

    def test_lookup_ambiguous(self, client):
        """Ambiguous abbreviation returns senses."""
        resp = client.get("/api/v1/abbreviations/lookup/PE")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_ambiguous"] is True
        assert len(data["senses"]) >= 2

    def test_lookup_not_found(self, client):
        """Unknown abbreviation returns null fields."""
        resp = client.get("/api/v1/abbreviations/lookup/ZZZZZ")
        assert resp.status_code == 200
        data = resp.json()
        assert data["expansion"] is None

    def test_lookup_case_insensitive(self, client):
        """Lookup works case-insensitively."""
        resp = client.get("/api/v1/abbreviations/lookup/htn")
        assert resp.status_code == 200
        assert resp.json()["expansion"] == "hypertension"


# ─────────────────────────────────────────────────────────────────────
# GET /abbreviations/dictionary/stats
# ─────────────────────────────────────────────────────────────────────


class TestDictionaryStats:
    """Test GET /abbreviations/dictionary/stats endpoint."""

    def test_stats_success(self, client):
        """Stats endpoint returns dictionary coverage."""
        resp = client.get("/api/v1/abbreviations/dictionary/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entries"] > 200
        assert data["total_unambiguous"] > 0
        assert data["total_ambiguous"] > 0

    def test_stats_has_domains(self, client):
        """Stats include domain breakdown."""
        resp = client.get("/api/v1/abbreviations/dictionary/stats")
        assert resp.status_code == 200
        assert len(resp.json()["domains"]) > 0

    def test_stats_domain_structure(self, client):
        """Each domain has required fields."""
        resp = client.get("/api/v1/abbreviations/dictionary/stats")
        assert resp.status_code == 200
        for domain in resp.json()["domains"]:
            assert "name" in domain
            assert "unambiguous_count" in domain
            assert "ambiguous_sense_count" in domain


# ─────────────────────────────────────────────────────────────────────
# GET /abbreviations/domains
# ─────────────────────────────────────────────────────────────────────


class TestDomains:
    """Test GET /abbreviations/domains endpoint."""

    def test_domains_list(self, client):
        """Returns all 12 clinical domains."""
        resp = client.get("/api/v1/abbreviations/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 12

    def test_domains_have_descriptions(self, client):
        """Each domain has a description."""
        resp = client.get("/api/v1/abbreviations/domains")
        assert resp.status_code == 200
        for domain in resp.json()["domains"]:
            assert "name" in domain
            assert "description" in domain
            assert len(domain["description"]) > 0
