"""Extended tests for document_service — covering _ensure_pipeline and
generic exception wrapping (lines 56, 120)."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.services.document_service import AnalysisService
from app.core.exceptions import InferenceError


class TestAnalysisServicePipeline:
    """Cover _ensure_pipeline lazy creation (line 56) and exception wrapping (line 120)."""

    def test_ensure_pipeline_creates_on_first_call(self) -> None:
        """_ensure_pipeline lazily creates and loads the ClinicalPipeline."""
        svc = AnalysisService.__new__(AnalysisService)
        svc._pipeline = None
        svc._is_loaded = False

        with patch("app.services.document_service.ClinicalPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            pipeline = svc._ensure_pipeline()

            MockPipeline.assert_called_once()
            mock_instance.load.assert_called_once()
            assert svc._is_loaded is True
            assert pipeline is mock_instance

    def test_ensure_pipeline_returns_cached(self) -> None:
        """Second call returns the cached pipeline without re-loading."""
        svc = AnalysisService.__new__(AnalysisService)
        mock_pipeline = MagicMock()
        svc._pipeline = mock_pipeline
        svc._is_loaded = True

        result = svc._ensure_pipeline()
        assert result is mock_pipeline
        mock_pipeline.load.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_wraps_generic_exception(self) -> None:
        """Non-InferenceError exceptions are wrapped in InferenceError (line 120)."""
        svc = AnalysisService.__new__(AnalysisService)
        mock_pipeline = MagicMock()
        mock_pipeline.process.side_effect = ValueError("Unexpected shape")
        svc._pipeline = mock_pipeline
        svc._is_loaded = True

        with patch("app.services.document_service.preprocess_clinical_text", return_value="cleaned"):
            with pytest.raises(InferenceError):
                await svc.analyze("Some clinical text")
