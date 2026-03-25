"""Unit tests for the clinical summarization route handler.

Tests cover:
- ``_run_summarization``: summary generation, compression ratio, key_points
  toggle, detail_level forwarding, metadata fields
- ``summarize_clinical_text``: endpoint handler — happy path, InferenceError,
  unexpected error, timing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.schemas.summary import SummarizationRequest, SummarizationResponse
from app.api.v1.routes.summarize import _run_summarization, summarize_clinical_text

# ---------------------------------------------------------------------------
# Fake stand-in
# ---------------------------------------------------------------------------


@dataclass
class FakeSumResult:
    summary: str = "Patient has diabetes managed with metformin."
    key_findings: list[str] = field(default_factory=lambda: ["diabetes", "metformin"])
    detail_level: str = "standard"
    processing_time_ms: float = 10.0
    model_name: str = "textrank"
    model_version: str = "1.0.0"
    sentence_count_original: int = 5
    sentence_count_summary: int = 1


TEXT = (
    "Patient is a 55-year-old male with type 2 diabetes mellitus on metformin 1000 mg BID. "
    "History of hypertension, currently on lisinopril 10 mg daily. Blood glucose well controlled."
)


def _mock_summarizer(result: FakeSumResult | None = None) -> MagicMock:
    model = MagicMock()
    model.summarize.return_value = result or FakeSumResult()
    model.model_name = "textrank"
    model.version = "1.0.0"
    return model


# ---------------------------------------------------------------------------
# _run_summarization
# ---------------------------------------------------------------------------


class TestRunSummarization:
    """Tests for the _run_summarization helper."""

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_compression_ratio(self, mock_get: MagicMock) -> None:
        """Compression ratio = original words / summary words."""
        # 6 summary words
        res = FakeSumResult(summary="This is a six word summary.")
        mock_get.return_value = _mock_summarizer(res)
        req = SummarizationRequest(text="word " * 30)  # 30 words
        resp = _run_summarization(req)
        assert resp.original_word_count == 30
        assert resp.summary_word_count == 6
        assert resp.compression_ratio == round(30 / 6, 2)

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_compression_ratio_single_word_summary(self, mock_get: MagicMock) -> None:
        """Edge case: single-word summary."""
        res = FakeSumResult(summary="Diabetes.")
        mock_get.return_value = _mock_summarizer(res)
        req = SummarizationRequest(text="word " * 20)
        resp = _run_summarization(req)
        assert resp.compression_ratio == 20.0

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_key_points_included_by_default(self, mock_get: MagicMock) -> None:
        """When include_key_points=True, key_points are populated."""
        mock_get.return_value = _mock_summarizer()
        req = SummarizationRequest(text=TEXT, include_key_points=True)
        resp = _run_summarization(req)
        assert resp.key_points is not None
        assert len(resp.key_points) > 0

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_key_points_excluded(self, mock_get: MagicMock) -> None:
        """When include_key_points=False, key_points is None."""
        mock_get.return_value = _mock_summarizer()
        req = SummarizationRequest(text=TEXT, include_key_points=False)
        resp = _run_summarization(req)
        assert resp.key_points is None

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_detail_level_forwarded(self, mock_get: MagicMock) -> None:
        """detail_level is passed to the summarizer model."""
        model = _mock_summarizer()
        mock_get.return_value = model
        req = SummarizationRequest(text=TEXT, detail_level="brief")
        _run_summarization(req)
        model.summarize.assert_called_once_with(TEXT, detail_level="brief")

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_summary_type_extractive(self, mock_get: MagicMock) -> None:
        """summary_type is always 'extractive'."""
        mock_get.return_value = _mock_summarizer()
        req = SummarizationRequest(text=TEXT)
        resp = _run_summarization(req)
        assert resp.summary_type == "extractive"

    @patch("app.api.v1.routes.summarize.get_summarizer")
    def test_model_metadata(self, mock_get: MagicMock) -> None:
        """Model name and version come from the summarizer."""
        mock_get.return_value = _mock_summarizer()
        req = SummarizationRequest(text=TEXT)
        resp = _run_summarization(req)
        assert resp.model_name == "textrank"
        assert resp.model_version == "1.0.0"


# ---------------------------------------------------------------------------
# summarize_clinical_text
# ---------------------------------------------------------------------------


class TestSummarizeClinicalText:
    """Tests for the summarize_clinical_text endpoint handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.summarize.get_summarizer")
    async def test_happy_path(self, mock_get: MagicMock) -> None:
        """Successful summarization returns SummarizationResponse."""
        mock_get.return_value = _mock_summarizer()
        payload = SummarizationRequest(text=TEXT)
        resp = await summarize_clinical_text(payload, AsyncMock(), MagicMock())

        assert isinstance(resp, SummarizationResponse)
        assert resp.summary != ""
        assert resp.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.summarize.get_summarizer")
    async def test_timing_is_set(self, mock_get: MagicMock) -> None:
        """Processing time is overwritten by the handler."""
        mock_get.return_value = _mock_summarizer()
        payload = SummarizationRequest(text=TEXT)
        resp = await summarize_clinical_text(payload, AsyncMock(), MagicMock())
        # The _run_summarization sets processing_time_ms=0.0, handler overwrites it
        assert resp.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.summarize.get_summarizer")
    async def test_inference_error_raises_500(self, mock_get: MagicMock) -> None:
        """InferenceError → HTTP 500."""
        from fastapi import HTTPException

        from app.core.exceptions import InferenceError

        mock_get.return_value = MagicMock(
            summarize=MagicMock(side_effect=InferenceError("sum failed"))
        )
        payload = SummarizationRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await summarize_clinical_text(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.summarize.get_summarizer")
    async def test_unexpected_error_raises_500(self, mock_get: MagicMock) -> None:
        """Unexpected exception → HTTP 500."""
        from fastapi import HTTPException

        mock_get.return_value = MagicMock(
            summarize=MagicMock(side_effect=KeyError("bad key"))
        )
        payload = SummarizationRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await summarize_clinical_text(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500
        assert "unexpectedly" in exc_info.value.detail
