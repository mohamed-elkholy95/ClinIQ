"""Tests for the streaming analysis SSE endpoint.

Covers SSE event formatting, stage-by-stage streaming, partial failure
handling, and disabled-stage skipping.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.api.v1.routes.stream import _sse_event, _stream_analysis
from app.api.schemas.analysis import AnalysisRequest

# Mock target base — stream.py imports these from app.services.model_registry
_MOCK_BASE = "app.api.v1.routes.stream"


# ---------------------------------------------------------------------------
# SSE formatter tests
# ---------------------------------------------------------------------------


class TestSSEEvent:
    """Tests for the _sse_event helper."""

    def test_basic_format(self) -> None:
        """Produces correctly formatted SSE event."""
        result = _sse_event("test", {"key": "value"})
        assert result.startswith("event: test\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_json_payload(self) -> None:
        """Data field contains valid JSON."""
        result = _sse_event("test", {"count": 42, "items": [1, 2, 3]})
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["count"] == 42
        assert payload["items"] == [1, 2, 3]

    def test_special_characters(self) -> None:
        """Handles special characters in payload."""
        result = _sse_event("test", {"text": "line1\nline2"})
        data_line = [l for l in result.split("\n") if l.startswith("data: ")][0]
        payload = json.loads(data_line.removeprefix("data: "))
        assert "line1" in payload["text"]


# ---------------------------------------------------------------------------
# Stream analysis generator tests
# ---------------------------------------------------------------------------


def _make_request(
    text: str = "Patient has diabetes mellitus type 2",
    *,
    ner: bool = True,
    icd: bool = True,
    summary: bool = True,
    risk: bool = True,
) -> AnalysisRequest:
    """Create an AnalysisRequest with configurable stages."""
    return AnalysisRequest(
        text=text,
        config={
            "ner": {"enabled": ner},
            "icd": {"enabled": icd},
            "summary": {"enabled": summary},
            "risk": {"enabled": risk},
        },
    )


def _parse_events(raw_events: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Parse SSE event strings into (event_name, data) tuples."""
    parsed: list[tuple[str, dict[str, Any]]] = []
    for raw in raw_events:
        lines = raw.strip().split("\n")
        event_name = ""
        data = {}
        for line in lines:
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event_name:
            parsed.append((event_name, data))
    return parsed


def _mock_ner_model() -> MagicMock:
    """Create a mock NER model that returns a fake entity."""
    model = MagicMock()
    entity = MagicMock()
    entity.text = "diabetes"
    entity.entity_type = "CONDITION"
    entity.start_char = 12
    entity.end_char = 20
    entity.confidence = 0.95
    entity.normalized_text = "diabetes mellitus"
    entity.umls_cui = "C0011849"
    entity.is_negated = False
    entity.is_uncertain = False
    model.extract_entities.return_value = [entity]
    return model


def _mock_icd_model() -> MagicMock:
    """Create a mock ICD model."""
    model = MagicMock()
    prediction = MagicMock()
    prediction.code = "E11.9"
    prediction.description = "Type 2 diabetes mellitus without complications"
    prediction.confidence = 0.88
    prediction.chapter = "IV"
    prediction.category = "Endocrine"
    prediction.contributing_text = ["diabetes mellitus"]
    result = MagicMock()
    result.predictions = [prediction]
    model.predict.return_value = result
    return model


def _mock_summarizer_model() -> MagicMock:
    """Create a mock summarizer."""
    model = MagicMock()
    model.model_name = "extractive-textrank"
    model.version = "1.0.0"
    result = MagicMock()
    result.summary = "Patient has diabetes."
    result.key_findings = ["diabetes"]
    model.summarize.return_value = result
    return model


def _mock_risk_model() -> MagicMock:
    """Create a mock risk scorer."""
    model = MagicMock()
    factor = MagicMock()
    factor.name = "chronic_disease"
    factor.description = "Chronic disease present"
    factor.weight = 0.8
    factor.score = 65
    assessment = MagicMock()
    assessment.overall_score = 55
    assessment.factors = [factor]
    assessment.recommendations = ["Monitor HbA1c quarterly"]
    model.assess_risk.return_value = assessment
    return model


def _patch_all_models():
    """Context manager that patches all four model registry getters."""
    return (
        patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
        patch(f"{_MOCK_BASE}.get_icd_model", return_value=_mock_icd_model()),
        patch(f"{_MOCK_BASE}.get_summarizer", return_value=_mock_summarizer_model()),
        patch(f"{_MOCK_BASE}.get_risk_scorer", return_value=_mock_risk_model()),
    )


class TestStreamAnalysis:
    """Tests for the _stream_analysis async generator."""

    @pytest.mark.asyncio
    async def test_all_stages_emit_events(self) -> None:
        """All enabled stages emit their events plus started/complete."""
        request = _make_request()
        events: list[str] = []

        p1, p2, p3, p4 = _patch_all_models()
        with p1, p2, p3, p4:
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert event_names[0] == "started"
        assert "ner" in event_names
        assert "icd" in event_names
        assert "summary" in event_names
        assert "risk" in event_names
        assert event_names[-1] == "complete"

    @pytest.mark.asyncio
    async def test_started_event_contains_metadata(self) -> None:
        """Started event includes text length and stage config."""
        request = _make_request(text="Test clinical note")
        events: list[str] = []

        p1, p2, p3, p4 = _patch_all_models()
        with p1, p2, p3, p4:
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        started = parsed[0][1]
        assert started["text_length"] == len("Test clinical note")
        assert started["stages_enabled"]["ner"] is True

    @pytest.mark.asyncio
    async def test_disabled_stages_skipped(self) -> None:
        """Disabled stages do not emit events."""
        request = _make_request(icd=False, summary=False)
        events: list[str] = []

        with (
            patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
            patch(f"{_MOCK_BASE}.get_risk_scorer", return_value=_mock_risk_model()),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert "icd" not in event_names
        assert "summary" not in event_names
        assert "ner" in event_names
        assert "risk" in event_names

    @pytest.mark.asyncio
    async def test_ner_stage_error_continues(self) -> None:
        """NER failure emits stage_error but doesn't abort remaining stages."""
        request = _make_request(icd=False, summary=False, risk=False)
        events: list[str] = []

        with patch(
            f"{_MOCK_BASE}.get_ner_model",
            side_effect=RuntimeError("NER model failed"),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert "stage_error" in event_names
        error_event = next(e for e in parsed if e[0] == "stage_error")
        assert error_event[1]["stage"] == "ner"
        assert "complete" in event_names

    @pytest.mark.asyncio
    async def test_icd_stage_error_continues(self) -> None:
        """ICD failure emits stage_error but remaining stages still run."""
        request = _make_request()
        events: list[str] = []

        with (
            patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
            patch(f"{_MOCK_BASE}.get_icd_model", side_effect=RuntimeError("ICD failure")),
            patch(f"{_MOCK_BASE}.get_summarizer", return_value=_mock_summarizer_model()),
            patch(f"{_MOCK_BASE}.get_risk_scorer", return_value=_mock_risk_model()),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert "stage_error" in event_names
        assert "summary" in event_names  # Continues despite ICD failure
        assert "risk" in event_names

    @pytest.mark.asyncio
    async def test_complete_event_has_timing(self) -> None:
        """Complete event includes total timing and stage count."""
        request = _make_request()
        events: list[str] = []

        p1, p2, p3, p4 = _patch_all_models()
        with p1, p2, p3, p4:
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        complete = next(e for e in parsed if e[0] == "complete")
        assert complete[1]["stages_completed"] == 4
        assert complete[1]["total_processing_time_ms"] > 0
        assert "stage_timings" in complete[1]

    @pytest.mark.asyncio
    async def test_ner_event_payload(self) -> None:
        """NER event contains entity list and count."""
        request = _make_request(icd=False, summary=False, risk=False)
        events: list[str] = []

        with patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        ner_event = next(e for e in parsed if e[0] == "ner")
        assert ner_event[1]["count"] == 1
        assert len(ner_event[1]["entities"]) == 1
        assert ner_event[1]["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_single_stage_only_emits_that_stage(self) -> None:
        """When only one stage enabled, only that stage's event is emitted."""
        request = _make_request(ner=True, icd=False, summary=False, risk=False)
        events: list[str] = []

        with patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert event_names == ["started", "ner", "complete"]
        complete = parsed[-1][1]
        assert complete["stages_completed"] == 1

    @pytest.mark.asyncio
    async def test_document_hash_in_started(self) -> None:
        """Started event includes SHA-256 document hash."""
        # Use only one enabled stage to satisfy the validator
        request = _make_request(
            text="test document for hashing",
            ner=True, icd=False, summary=False, risk=False,
        )
        events: list[str] = []

        with patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        started = parsed[0][1]
        assert "document_hash" in started
        assert len(started["document_hash"]) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_summary_stage_error_continues(self) -> None:
        """Summary failure emits stage_error but risk still runs."""
        request = _make_request()
        events: list[str] = []

        with (
            patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
            patch(f"{_MOCK_BASE}.get_icd_model", return_value=_mock_icd_model()),
            patch(f"{_MOCK_BASE}.get_summarizer", side_effect=RuntimeError("Summary failure")),
            patch(f"{_MOCK_BASE}.get_risk_scorer", return_value=_mock_risk_model()),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert "stage_error" in event_names
        assert "risk" in event_names  # Continues past summary failure

    @pytest.mark.asyncio
    async def test_risk_stage_error(self) -> None:
        """Risk stage failure is captured as stage_error."""
        request = _make_request()
        events: list[str] = []

        with (
            patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
            patch(f"{_MOCK_BASE}.get_icd_model", return_value=_mock_icd_model()),
            patch(f"{_MOCK_BASE}.get_summarizer", return_value=_mock_summarizer_model()),
            patch(f"{_MOCK_BASE}.get_risk_scorer", side_effect=RuntimeError("Risk failure")),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        event_names = [e[0] for e in parsed]
        assert "stage_error" in event_names
        error_event = next(e for e in parsed if e[0] == "stage_error")
        assert error_event[1]["stage"] == "risk"
        assert "complete" in event_names

    @pytest.mark.asyncio
    async def test_partial_failure_counts_successful_stages(self) -> None:
        """stages_completed only counts stages that succeeded."""
        request = _make_request()
        events: list[str] = []

        with (
            patch(f"{_MOCK_BASE}.get_ner_model", return_value=_mock_ner_model()),
            patch(f"{_MOCK_BASE}.get_icd_model", side_effect=RuntimeError("fail")),
            patch(f"{_MOCK_BASE}.get_summarizer", return_value=_mock_summarizer_model()),
            patch(f"{_MOCK_BASE}.get_risk_scorer", side_effect=RuntimeError("fail")),
        ):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        complete = next(e for e in parsed if e[0] == "complete")
        assert complete[1]["stages_completed"] == 2  # NER + summary succeeded

    @pytest.mark.asyncio
    async def test_icd_event_payload(self) -> None:
        """ICD event contains predictions list and count."""
        request = _make_request(ner=False, summary=False, risk=False)
        events: list[str] = []

        with patch(f"{_MOCK_BASE}.get_icd_model", return_value=_mock_icd_model()):
            async for event in _stream_analysis(request):
                events.append(event)

        parsed = _parse_events(events)
        icd_event = next(e for e in parsed if e[0] == "icd")
        assert icd_event[1]["count"] == 1
        assert len(icd_event[1]["predictions"]) == 1
        assert icd_event[1]["predictions"][0]["code"] == "E11.9"
