"""Unit tests for API dependency injection module.

Tests the FastAPI dependency functions that provide authentication,
database sessions, and ML pipeline instances to route handlers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.deps import get_ml_pipeline, get_superuser

# ---------------------------------------------------------------------------
# get_ml_pipeline
# ---------------------------------------------------------------------------


class TestGetMLPipeline:
    """Tests for the get_ml_pipeline dependency."""

    @patch("app.services.model_registry.get_summarizer")
    @patch("app.services.model_registry.get_risk_scorer")
    @patch("app.services.model_registry.get_icd_model")
    @patch("app.services.model_registry.get_ner_model")
    def test_returns_clinical_pipeline(
        self,
        mock_ner: MagicMock,
        mock_icd: MagicMock,
        mock_risk: MagicMock,
        mock_summarizer: MagicMock,
    ):
        """Should construct a ClinicalPipeline with registry models."""
        from app.ml.pipeline import ClinicalPipeline

        pipeline = get_ml_pipeline()
        assert isinstance(pipeline, ClinicalPipeline)
        mock_ner.assert_called_once()
        mock_icd.assert_called_once()
        mock_risk.assert_called_once()
        mock_summarizer.assert_called_once()


# ---------------------------------------------------------------------------
# get_superuser
# ---------------------------------------------------------------------------


class TestGetSuperuser:
    """Tests for the get_superuser dependency."""

    @pytest.mark.asyncio
    async def test_allows_superuser(self):
        user = MagicMock()
        user.is_superuser = True
        result = await get_superuser(user)
        assert result is user

    @pytest.mark.asyncio
    async def test_rejects_non_superuser(self):
        user = MagicMock()
        user.is_superuser = False
        with pytest.raises(HTTPException) as exc_info:
            await get_superuser(user)
        assert exc_info.value.status_code == 403
