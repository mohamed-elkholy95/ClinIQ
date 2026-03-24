"""Integration tests for analysis endpoints."""

import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock, patch

from app.ml.ner.model import Entity
from app.ml.icd.model import ICDCodePrediction, ICDPredictionResult
from app.ml.summarization.model import SummarizationResult as SummaryResult


class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_full_pipeline(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test full analysis pipeline."""
        with patch("app.api.v1.routes.analyze.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/analyze",
                json={
                    "text": sample_clinical_text,
                    "enable_ner": True,
                    "enable_icd": True,
                    "enable_summarization": True,
                    "enable_risk": True,
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        assert "icd_predictions" in data
        assert "processing_time_ms" in data

    @pytest.mark.asyncio
    async def test_analyze_ner_only(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test analysis with NER only."""
        with patch("app.api.v1.routes.analyze.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/analyze",
                json={
                    "text": sample_clinical_text,
                    "enable_ner": True,
                    "enable_icd": False,
                    "enable_summarization": False,
                    "enable_risk": False,
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        assert len(data["entities"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_invalid_text(self, async_client: AsyncClient):
        """Test analysis with invalid input."""
        response = await async_client.post(
            "/api/v1/analyze",
            json={
                "text": "too short",
                "enable_ner": True,
            },
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_analyze_empty_request(self, async_client: AsyncClient):
        """Test analysis with empty request."""
        response = await async_client.post(
            "/api/v1/analyze",
            json={},
        )

        assert response.status_code == 422


class TestNEREndpoint:
    """Tests for the /ner endpoint."""

    @pytest.mark.asyncio
    async def test_extract_entities(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test entity extraction."""
        with patch("app.api.v1.routes.ner.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/ner",
                json={"text": sample_clinical_text},
            )

        assert response.status_code == 200
        data = response.json()

        assert "entities" in data
        assert "entity_count" in data
        assert "entity_type_counts" in data
        assert data["entity_count"] > 0


class TestICDEndpoint:
    """Tests for the /icd endpoint."""

    @pytest.mark.asyncio
    async def test_predict_icd_codes(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test ICD-10 code prediction."""
        with patch("app.api.v1.routes.icd.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/icd/predict",
                json={"text": sample_clinical_text, "top_k": 5},
            )

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data


class TestSummarizeEndpoint:
    """Tests for the /summarize endpoint."""

    @pytest.mark.asyncio
    async def test_summarize_text(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test text summarization."""
        with patch("app.api.v1.routes.summarize.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/summarize",
                json={"text": sample_clinical_text, "max_length": 100},
            )

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data
        assert "summary" in data["summary"]


class TestRiskEndpoint:
    """Tests for the /risk endpoint."""

    @pytest.mark.asyncio
    async def test_calculate_risk_score(
        self,
        async_client: AsyncClient,
        sample_clinical_text: str,
        mock_pipeline: MagicMock,
    ):
        """Test risk score calculation."""
        from app.ml.risk.scorer import RiskScore

        mock_pipeline.calculate_risk.return_value = RiskScore(
            overall_score=0.65,
            risk_level="moderate",
            category_scores={"cardiovascular": 0.7},
            risk_factors=[],
            protective_factors=[],
            recommendations=["Follow up in 3 months"],
            processing_time_ms=50.0,
            model_name="test-risk",
            model_version="1.0.0",
        )

        with patch("app.api.v1.routes.risk.get_pipeline", return_value=mock_pipeline):
            response = await async_client.post(
                "/api/v1/risk/score",
                json={"text": sample_clinical_text},
            )

        assert response.status_code == 200
        data = response.json()

        assert "risk_score" in data
        assert data["risk_score"]["overall_score"] == 0.65


class TestHealthEndpoint:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_liveness(self, async_client: AsyncClient):
        """Test liveness probe."""
        response = await async_client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
