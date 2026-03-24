"""Integration tests for analysis endpoints.

These tests exercise the full FastAPI application stack with an in-memory
SQLite database.  ML model singletons are mocked via the model registry
so that inference runs without real model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from app.ml.ner.model import Entity
from app.ml.icd.model import ICDCodePrediction
from app.ml.summarization.model import SummarizationResult
from app.ml.risk.model import RiskAssessment, RiskFactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ner_model() -> MagicMock:
    """Return a mock NER model that produces deterministic entities."""
    model = MagicMock()
    model.predict.return_value = [
        Entity(text="hypertension", entity_type="CONDITION", start_char=0, end_char=12, confidence=0.95),
        Entity(text="metformin", entity_type="MEDICATION", start_char=20, end_char=29, confidence=0.92),
    ]
    model.is_loaded.return_value = True
    return model


def _mock_icd_model() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = [
        ICDCodePrediction(code="I10", description="Essential hypertension", confidence=0.88),
        ICDCodePrediction(code="E11.9", description="Type 2 diabetes", confidence=0.76),
    ]
    model.is_loaded.return_value = True
    return model


def _mock_summarizer() -> MagicMock:
    model = MagicMock()
    model.summarize.return_value = SummarizationResult(
        summary="Patient presents with hypertension and diabetes.",
        key_findings=["hypertension", "diabetes"],
        detail_level="standard",
        processing_time_ms=25.0,
        model_name="extractive-summarizer",
        model_version="1.0.0",
        sentence_count_original=5,
        sentence_count_summary=1,
    )
    model.is_loaded.return_value = True
    return model


def _mock_risk_scorer() -> MagicMock:
    model = MagicMock()
    model.assess_risk.return_value = RiskAssessment(
        overall_score=55.0,
        risk_level="moderate",
        factors=[RiskFactor(name="hypertension", score=0.7, weight=1.0, category="cardiovascular", description="Elevated blood pressure")],
        recommendations=["Monitor blood pressure"],
        processing_time_ms=30.0,
        category_scores={"cardiovascular": 70.0},
        model_name="rule-based-risk",
        model_version="1.0.0",
    )
    model.is_loaded.return_value = True
    return model


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    """Tests for the /api/v1/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_full_pipeline(self, async_client: AsyncClient) -> None:
        """POST /analyze with all stages enabled returns aggregated result."""
        with patch("app.services.model_registry.get_ner_model", return_value=_mock_ner_model()), \
             patch("app.services.model_registry.get_icd_model", return_value=_mock_icd_model()), \
             patch("app.services.model_registry.get_summarizer", return_value=_mock_summarizer()), \
             patch("app.services.model_registry.get_risk_scorer", return_value=_mock_risk_scorer()):
            response = await async_client.post(
                "/api/v1/analyze",
                json={
                    "text": "Patient presents with hypertension managed with metformin. "
                            "Blood pressure 140/90. Follow-up in 3 months.",
                    "enable_ner": True,
                    "enable_icd": True,
                    "enable_summarization": True,
                    "enable_risk": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data or "processing_time_ms" in data

    @pytest.mark.asyncio
    async def test_analyze_empty_request(self, async_client: AsyncClient) -> None:
        """POST /analyze with empty body returns 422."""
        response = await async_client.post("/api/v1/analyze", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_ner_only(self, async_client: AsyncClient) -> None:
        """POST /analyze with only NER enabled."""
        with patch("app.services.model_registry.get_ner_model", return_value=_mock_ner_model()):
            response = await async_client.post(
                "/api/v1/analyze",
                json={
                    "text": "Patient presents with hypertension managed with metformin. "
                            "Blood pressure 140/90. Follow-up scheduled.",
                    "enable_ner": True,
                    "enable_icd": False,
                    "enable_summarization": False,
                    "enable_risk": False,
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_invalid_text(self, async_client: AsyncClient) -> None:
        """POST /analyze with missing required text field returns 422."""
        response = await async_client.post(
            "/api/v1/analyze",
            json={"enable_ner": True},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# NER endpoint
# ---------------------------------------------------------------------------

class TestNEREndpoint:
    """Tests for the /api/v1/ner endpoint."""

    @pytest.mark.asyncio
    async def test_extract_entities(self, async_client: AsyncClient) -> None:
        with patch("app.services.model_registry.get_ner_model", return_value=_mock_ner_model()):
            response = await async_client.post(
                "/api/v1/ner",
                json={"text": "Patient has hypertension and takes metformin daily. Follow-up in two weeks."},
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data


# ---------------------------------------------------------------------------
# ICD endpoint
# ---------------------------------------------------------------------------

class TestICDEndpoint:
    """Tests for the /api/v1/icd endpoint."""

    @pytest.mark.asyncio
    async def test_predict_icd_codes(self, async_client: AsyncClient) -> None:
        with patch("app.services.model_registry.get_icd_model", return_value=_mock_icd_model()):
            response = await async_client.post(
                "/api/v1/icd-predict",
                json={"text": "Patient diagnosed with essential hypertension and type 2 diabetes mellitus."},
            )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data


# ---------------------------------------------------------------------------
# Summarize endpoint
# ---------------------------------------------------------------------------

class TestSummarizeEndpoint:
    """Tests for the /api/v1/summarize endpoint."""

    @pytest.mark.asyncio
    async def test_summarize_text(self, async_client: AsyncClient) -> None:
        with patch("app.services.model_registry.get_summarizer", return_value=_mock_summarizer()):
            response = await async_client.post(
                "/api/v1/summarize",
                json={
                    "text": "Patient presents with hypertension. Blood pressure 140/90 mmHg. "
                            "Currently on metformin. Follow-up in 3 months.",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data


# ---------------------------------------------------------------------------
# Risk endpoint
# ---------------------------------------------------------------------------

class TestRiskEndpoint:
    """Tests for the /api/v1/risk endpoint."""

    @pytest.mark.asyncio
    async def test_calculate_risk_score(self, async_client: AsyncClient) -> None:
        """Test risk scoring endpoint — uses real rule-based model (no heavy deps)."""
        response = await async_client.post(
            "/api/v1/risk-score",
            json={"text": "Patient with hypertension and type 2 diabetes. Currently on metformin."},
        )

        assert response.status_code == 200
        data = response.json()
        assert "score" in data or "category" in data


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "version" in data

    @pytest.mark.asyncio
    async def test_liveness(self, async_client: AsyncClient) -> None:
        response = await async_client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
