"""Unit tests for the NER route handler.

Tests cover:
- ``_run_ner_inference``: entity extraction, type filtering, negation/uncertainty
  filtering, confidence threshold, and field mapping
- ``extract_entities``: endpoint handler — happy path, InferenceError, unexpected
  exceptions, timing, sorting, response metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.api.v1.routes.ner import _run_ner_inference, extract_entities
from app.api.schemas.ner import NERRequest, NERResponse


# ---------------------------------------------------------------------------
# Fake Entity stand-in
# ---------------------------------------------------------------------------


@dataclass
class FakeEntity:
    """Lightweight Entity stand-in matching the ML-layer interface."""

    text: str = "metformin"
    entity_type: str = "MEDICATION"
    start_char: int = 10
    end_char: int = 19
    confidence: float = 0.95
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


TEXT = "Patient takes metformin 1000 mg for type 2 diabetes."


# ---------------------------------------------------------------------------
# _run_ner_inference
# ---------------------------------------------------------------------------


class TestRunNerInference:
    """Tests for the _run_ner_inference helper."""

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_returns_all_entities_defaults(self, mock_get: MagicMock) -> None:
        """With default settings, all entities are returned."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[FakeEntity(), FakeEntity(text="diabetes", entity_type="DISEASE")])
        )
        req = NERRequest(text=TEXT)
        result = _run_ner_inference(req)
        assert len(result) == 2

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_entity_type_filter(self, mock_get: MagicMock) -> None:
        """Only entities matching requested entity_types are returned."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(text="metformin", entity_type="MEDICATION"),
                FakeEntity(text="diabetes", entity_type="DISEASE"),
            ])
        )
        req = NERRequest(text=TEXT, entity_types=["MEDICATION"])
        result = _run_ner_inference(req)
        assert len(result) == 1
        assert result[0].entity_type == "MEDICATION"

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_negation_filter(self, mock_get: MagicMock) -> None:
        """When include_negated=False, negated entities are dropped."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(text="chest pain", is_negated=True),
                FakeEntity(text="diabetes", is_negated=False),
            ])
        )
        req = NERRequest(text=TEXT, include_negated=False)
        result = _run_ner_inference(req)
        assert len(result) == 1
        assert result[0].text == "diabetes"

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_uncertainty_filter(self, mock_get: MagicMock) -> None:
        """When include_uncertain=False, uncertain entities are dropped."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(text="possible PE", is_uncertain=True, confidence=0.9),
                FakeEntity(text="diabetes", is_uncertain=False, confidence=0.9),
            ])
        )
        req = NERRequest(text=TEXT, include_uncertain=False)
        result = _run_ner_inference(req)
        assert len(result) == 1
        assert result[0].text == "diabetes"

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_confidence_threshold(self, mock_get: MagicMock) -> None:
        """Entities below min_confidence are excluded."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(text="high_conf", confidence=0.9),
                FakeEntity(text="low_conf", confidence=0.2),
            ])
        )
        req = NERRequest(text=TEXT, min_confidence=0.5)
        result = _run_ner_inference(req)
        assert len(result) == 1
        assert result[0].text == "high_conf"

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_combined_filters(self, mock_get: MagicMock) -> None:
        """Type, negation, uncertainty, and confidence filters compose correctly."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(text="metformin", entity_type="MEDICATION", confidence=0.95, is_negated=False, is_uncertain=False),
                FakeEntity(text="chest pain", entity_type="SYMPTOM", confidence=0.8, is_negated=True, is_uncertain=False),
                FakeEntity(text="possible PE", entity_type="DISEASE", confidence=0.6, is_negated=False, is_uncertain=True),
                FakeEntity(text="weak signal", entity_type="MEDICATION", confidence=0.1, is_negated=False, is_uncertain=False),
            ])
        )
        req = NERRequest(
            text=TEXT,
            entity_types=["MEDICATION"],
            include_negated=False,
            include_uncertain=False,
            min_confidence=0.5,
        )
        result = _run_ner_inference(req)
        assert len(result) == 1
        assert result[0].text == "metformin"

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_empty_model_output(self, mock_get: MagicMock) -> None:
        """Model returning no entities → empty list."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[])
        )
        req = NERRequest(text=TEXT)
        result = _run_ner_inference(req)
        assert result == []

    @patch("app.api.v1.routes.ner.get_ner_model")
    def test_field_mapping(self, mock_get: MagicMock) -> None:
        """Entity fields are mapped correctly to EntityResponse."""
        entity = FakeEntity(
            text="lisinopril",
            entity_type="MEDICATION",
            start_char=100,
            end_char=110,
            confidence=0.88,
            normalized_text="Lisinopril",
            umls_cui="C0065374",
            is_negated=False,
            is_uncertain=True,
        )
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[entity])
        )
        req = NERRequest(text=TEXT)
        result = _run_ner_inference(req)
        r = result[0]
        assert r.text == "lisinopril"
        assert r.normalized_text == "Lisinopril"
        assert r.umls_cui == "C0065374"
        assert r.is_uncertain is True


# ---------------------------------------------------------------------------
# extract_entities (endpoint handler)
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """Tests for the extract_entities endpoint handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.ner.get_ner_model")
    async def test_happy_path(self, mock_get: MagicMock) -> None:
        """Successful extraction returns well-formed NERResponse."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[
                FakeEntity(start_char=50),
                FakeEntity(start_char=10, text="first"),
            ])
        )
        payload = NERRequest(text=TEXT)
        resp = await extract_entities(payload, AsyncMock(), MagicMock())

        assert isinstance(resp, NERResponse)
        assert resp.text_length == len(TEXT)
        assert resp.entity_count == 2
        # Sorted by start_char
        assert resp.entities[0].start_char <= resp.entities[1].start_char

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.ner.get_ner_model")
    async def test_processing_time_positive(self, mock_get: MagicMock) -> None:
        """Processing time should be a non-negative float."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[])
        )
        payload = NERRequest(text=TEXT)
        resp = await extract_entities(payload, AsyncMock(), MagicMock())
        assert resp.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.ner.get_ner_model")
    async def test_model_name_from_request(self, mock_get: MagicMock) -> None:
        """The model name in the response comes from the request."""
        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(return_value=[])
        )
        payload = NERRequest(text=TEXT, model="spacy")
        resp = await extract_entities(payload, AsyncMock(), MagicMock())
        assert resp.model_name == "spacy"

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.ner.get_ner_model")
    async def test_inference_error_raises_500(self, mock_get: MagicMock) -> None:
        """InferenceError from model → HTTP 500."""
        from fastapi import HTTPException
        from app.core.exceptions import InferenceError

        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(side_effect=InferenceError("NER failed"))
        )
        payload = NERRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await extract_entities(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.ner.get_ner_model")
    async def test_unexpected_error_raises_500(self, mock_get: MagicMock) -> None:
        """Unexpected exception → HTTP 500 with generic message."""
        from fastapi import HTTPException

        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(side_effect=ValueError("unexpected"))
        )
        payload = NERRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await extract_entities(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500
        assert "unexpectedly" in exc_info.value.detail
