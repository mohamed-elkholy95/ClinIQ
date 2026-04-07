"""Tests for the enhanced analysis REST API endpoints.

Validates ``POST /analyze/enhanced``, ``POST /analyze/enhanced/batch``,
and ``GET /analyze/enhanced/modules`` endpoints.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

SAMPLE_NOTE = """
CHIEF COMPLAINT:
Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
65-year-old male with hypertension and diabetes presents with acute chest pain.
Currently taking metoprolol 50 mg BID and lisinopril 20 mg daily.

ALLERGIES:
Penicillin — anaphylaxis

VITAL SIGNS:
BP: 165/95 mmHg, HR: 92 bpm, Temp: 98.6°F, SpO2: 96% on RA

SOCIAL HISTORY:
Former smoker, quit 5 years ago. Lives alone.

ASSESSMENT AND PLAN:
Acute chest pain — rule out ACS. Start heparin drip.
"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===================================================================
# POST /analyze/enhanced — single document
# ===================================================================


class TestEnhancedAnalyzeEndpoint:
    """Test the single-document enhanced analysis endpoint."""

    @pytest.mark.anyio
    async def test_successful_analysis(self, client):
        """Should return 200 with comprehensive results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert "component_errors" in data

    @pytest.mark.anyio
    async def test_sections_in_response(self, client):
        """Response should include section parsing results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["sections"] is not None
        assert data["sections"]["section_count"] > 0

    @pytest.mark.anyio
    async def test_medications_in_response(self, client):
        """Response should include medication extraction results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["medications"] is not None
        assert data["medications"]["medication_count"] > 0

    @pytest.mark.anyio
    async def test_allergies_in_response(self, client):
        """Response should include allergy extraction results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["allergies"] is not None
        assert data["allergies"]["allergy_count"] > 0

    @pytest.mark.anyio
    async def test_vitals_in_response(self, client):
        """Response should include vital signs extraction results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["vitals"] is not None
        assert data["vitals"]["vital_count"] > 0

    @pytest.mark.anyio
    async def test_quality_in_response(self, client):
        """Response should include quality analysis results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["quality"] is not None
        assert "overall_score" in data["quality"]
        assert "grade" in data["quality"]

    @pytest.mark.anyio
    async def test_classification_in_response(self, client):
        """Response should include document classification."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["classification"] is not None
        assert "predicted_type" in data["classification"]

    @pytest.mark.anyio
    async def test_deidentification_off_by_default(self, client):
        """De-identification should be None by default."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE},
        )
        data = response.json()
        assert data["deidentification"] is None

    @pytest.mark.anyio
    async def test_deidentification_when_enabled(self, client):
        """De-identification should run when explicitly enabled."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE, "enable_deidentification": True},
        )
        data = response.json()
        assert data["deidentification"] is not None

    @pytest.mark.anyio
    async def test_module_toggles(self, client):
        """Disabling modules should exclude their results."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={
                "text": SAMPLE_NOTE,
                "enable_sections": False,
                "enable_medications": False,
                "enable_quality": False,
            },
        )
        data = response.json()
        assert data["sections"] is None
        assert data["medications"] is None
        assert data["quality"] is None

    @pytest.mark.anyio
    async def test_document_id_propagated(self, client):
        """Document ID should appear in the response."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": SAMPLE_NOTE, "document_id": "test-123"},
        )
        data = response.json()
        if data["base_result"]:
            assert data["base_result"]["document_id"] == "test-123"

    @pytest.mark.anyio
    async def test_empty_text_rejected(self, client):
        """Empty text should be rejected with 422."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={"text": ""},
        )
        assert response.status_code == 422

    @pytest.mark.anyio
    async def test_missing_text_rejected(self, client):
        """Missing text field should be rejected with 422."""
        response = await client.post(
            "/api/v1/analyze/enhanced",
            json={},
        )
        assert response.status_code == 422


# ===================================================================
# POST /analyze/enhanced/batch
# ===================================================================


class TestEnhancedBatchEndpoint:
    """Test the batch enhanced analysis endpoint."""

    @pytest.mark.anyio
    async def test_batch_returns_results(self, client):
        """Batch should return results for all documents."""
        response = await client.post(
            "/api/v1/analyze/enhanced/batch",
            json={
                "documents": [
                    {"text": SAMPLE_NOTE},
                    {"text": "Patient denies chest pain. NKDA."},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 2
        assert len(data["results"]) == 2

    @pytest.mark.anyio
    async def test_batch_per_document_toggles(self, client):
        """Each document in batch can have different module toggles."""
        response = await client.post(
            "/api/v1/analyze/enhanced/batch",
            json={
                "documents": [
                    {"text": SAMPLE_NOTE, "enable_sections": True, "enable_vitals": False},
                    {"text": "BP: 120/80", "enable_sections": False, "enable_vitals": True},
                ],
            },
        )
        data = response.json()
        assert data["results"][0]["sections"] is not None
        assert data["results"][0]["vitals"] is None
        assert data["results"][1]["sections"] is None
        assert data["results"][1]["vitals"] is not None

    @pytest.mark.anyio
    async def test_batch_empty_rejected(self, client):
        """Empty batch should be rejected."""
        response = await client.post(
            "/api/v1/analyze/enhanced/batch",
            json={"documents": []},
        )
        assert response.status_code == 422


# ===================================================================
# GET /analyze/enhanced/modules
# ===================================================================


class TestModulesCatalogue:
    """Test the module catalogue endpoint."""

    @pytest.mark.anyio
    async def test_modules_endpoint(self, client):
        """Should return the module catalogue."""
        response = await client.get("/api/v1/analyze/enhanced/modules")
        assert response.status_code == 200
        data = response.json()
        assert data["total_modules"] == 14

    @pytest.mark.anyio
    async def test_module_structure(self, client):
        """Each module should have name, description, default_enabled."""
        response = await client.get("/api/v1/analyze/enhanced/modules")
        data = response.json()
        for module in data["modules"]:
            assert "name" in module
            assert "description" in module
            assert "default_enabled" in module

    @pytest.mark.anyio
    async def test_deidentification_default_disabled(self, client):
        """De-identification module should show as disabled by default."""
        response = await client.get("/api/v1/analyze/enhanced/modules")
        data = response.json()
        deid = next(m for m in data["modules"] if m["name"] == "deidentification")
        assert deid["default_enabled"] is False
