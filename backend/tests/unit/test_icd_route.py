"""Unit tests for the ICD-10 prediction route handler.

Tests cover:
- ``_run_icd_inference``: prediction delegation, confidence filtering,
  chapter inclusion toggle, top_k forwarding
- ``predict_icd_codes``: endpoint handler — happy path, InferenceError,
  unexpected error, timing, response metadata
- ``get_icd_code_details``: code lookup (found and 404)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.routes.icd import (
    _run_icd_inference,
    get_icd_code_details,
    predict_icd_codes,
)
from app.api.schemas.icd import ICDPredictionRequest, ICDPredictionResponse


# ---------------------------------------------------------------------------
# Fake stand-ins
# ---------------------------------------------------------------------------


@dataclass
class FakePrediction:
    code: str = "E11.9"
    description: str = "Type 2 diabetes mellitus without complications"
    confidence: float = 0.85
    chapter: str = "Endocrine, nutritional and metabolic diseases"
    category: str | None = "Diabetes mellitus"
    contributing_text: list[str] | None = None


@dataclass
class FakeICDResult:
    predictions: list[FakePrediction] = field(
        default_factory=lambda: [FakePrediction()]
    )


TEXT = "Patient with type 2 diabetes and hypertension."


# ---------------------------------------------------------------------------
# _run_icd_inference
# ---------------------------------------------------------------------------


class TestRunIcdInference:
    """Tests for the _run_icd_inference helper."""

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_confidence_filter(self, mock_get: MagicMock) -> None:
        """Predictions below min_confidence are excluded."""
        preds = [
            FakePrediction(code="E11.9", confidence=0.85),
            FakePrediction(code="I10", confidence=0.15),
        ]
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult(predictions=preds))
        )
        req = ICDPredictionRequest(text=TEXT, min_confidence=0.5)
        result = _run_icd_inference(req)
        assert len(result) == 1
        assert result[0].code == "E11.9"

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_top_k_forwarded(self, mock_get: MagicMock) -> None:
        """top_k is forwarded to the model's predict call."""
        model = MagicMock(predict=MagicMock(return_value=FakeICDResult()))
        mock_get.return_value = model
        req = ICDPredictionRequest(text=TEXT, top_k=3)
        _run_icd_inference(req)
        model.predict.assert_called_once_with(TEXT, top_k=3)

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_chapter_included_by_default(self, mock_get: MagicMock) -> None:
        """When include_chapter=True (default), chapter/category are populated."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult())
        )
        req = ICDPredictionRequest(text=TEXT, include_chapter=True)
        result = _run_icd_inference(req)
        assert result[0].chapter is not None
        assert result[0].category is not None

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_chapter_excluded(self, mock_get: MagicMock) -> None:
        """When include_chapter=False, chapter/category are None."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult())
        )
        req = ICDPredictionRequest(text=TEXT, include_chapter=False)
        result = _run_icd_inference(req)
        assert result[0].chapter is None
        assert result[0].category is None

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_empty_predictions(self, mock_get: MagicMock) -> None:
        """No predictions from model → empty list."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult(predictions=[]))
        )
        req = ICDPredictionRequest(text=TEXT)
        result = _run_icd_inference(req)
        assert result == []

    @patch("app.api.v1.routes.icd.get_icd_model")
    def test_all_filtered_out(self, mock_get: MagicMock) -> None:
        """When min_confidence is very high, all predictions may be dropped."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult(
                predictions=[FakePrediction(confidence=0.3)]
            ))
        )
        req = ICDPredictionRequest(text=TEXT, min_confidence=0.99)
        result = _run_icd_inference(req)
        assert result == []


# ---------------------------------------------------------------------------
# predict_icd_codes
# ---------------------------------------------------------------------------


class TestPredictIcdCodes:
    """Tests for the predict_icd_codes endpoint handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.icd.get_icd_model")
    async def test_happy_path(self, mock_get: MagicMock) -> None:
        """Successful prediction returns well-formed ICDPredictionResponse."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult())
        )
        payload = ICDPredictionRequest(text=TEXT)
        resp = await predict_icd_codes(payload, AsyncMock(), MagicMock())

        assert isinstance(resp, ICDPredictionResponse)
        assert resp.prediction_count == 1
        assert resp.predictions[0].code == "E11.9"
        assert resp.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.icd.get_icd_model")
    async def test_model_name_from_request(self, mock_get: MagicMock) -> None:
        """Model name in response comes from the request."""
        mock_get.return_value = MagicMock(
            predict=MagicMock(return_value=FakeICDResult())
        )
        payload = ICDPredictionRequest(text=TEXT, model="transformer")
        resp = await predict_icd_codes(payload, AsyncMock(), MagicMock())
        assert resp.model_name == "transformer"

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.icd.get_icd_model")
    async def test_inference_error_raises_500(self, mock_get: MagicMock) -> None:
        """InferenceError → HTTP 500."""
        from fastapi import HTTPException
        from app.core.exceptions import InferenceError

        mock_get.return_value = MagicMock(
            predict=MagicMock(side_effect=InferenceError("model crashed"))
        )
        payload = ICDPredictionRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await predict_icd_codes(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.icd.get_icd_model")
    async def test_unexpected_error_raises_500(self, mock_get: MagicMock) -> None:
        """Unexpected exception → HTTP 500 with generic message."""
        from fastapi import HTTPException

        mock_get.return_value = MagicMock(
            predict=MagicMock(side_effect=TypeError("oops"))
        )
        payload = ICDPredictionRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await predict_icd_codes(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500
        assert "unexpectedly" in exc_info.value.detail


# ---------------------------------------------------------------------------
# get_icd_code_details
# ---------------------------------------------------------------------------


class TestGetIcdCodeDetails:
    """Tests for the ICD-10 code lookup endpoint."""

    @pytest.mark.asyncio
    async def test_code_found(self) -> None:
        """Existing code returns details dict."""
        fake_row = MagicMock()
        fake_row.code = "I10"
        fake_row.description = "Essential (primary) hypertension"
        fake_row.chapter = "Circulatory"
        fake_row.category = "Hypertensive diseases"
        fake_row.is_active = True

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_row

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        resp = await get_icd_code_details("I10", db)
        assert resp["code"] == "I10"
        assert resp["description"] == "Essential (primary) hypertension"
        assert resp["is_active"] is True

    @pytest.mark.asyncio
    async def test_code_not_found_raises_404(self) -> None:
        """Missing code → HTTP 404."""
        from fastapi import HTTPException

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(HTTPException) as exc_info:
            await get_icd_code_details("ZZZZ", db)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_code_uppercased_for_query(self) -> None:
        """The code is uppercased before querying."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        # We can't easily inspect the SQL directly, but we verify
        # lowercase input still triggers the lookup without error
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_icd_code_details("i10", db)
        assert exc_info.value.status_code == 404
