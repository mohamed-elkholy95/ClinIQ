"""Unit tests for the full-pipeline analysis route.

Tests cover:
- ``_run_ner``: entity extraction delegation with confidence filtering
- ``_run_icd``: ICD-10 prediction delegation with top_k and confidence
- ``_run_summary``: summarisation delegation with compression ratio maths
- ``_run_risk``: risk scoring delegation with category mapping
- ``run_analysis``: full pipeline orchestration (all enabled, selective disable,
  error handling, timing, audit trail, document_id passthrough)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.routes.analyze import (
    _run_icd,
    _run_ner,
    _run_risk,
    _run_summary,
    run_analysis,
)
from app.api.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    ICDConfig,
    NERConfig,
    PipelineConfig,
    RiskConfig,
    SummaryConfig,
)


# ---------------------------------------------------------------------------
# Lightweight stand-ins for ML-layer dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FakeEntity:
    text: str = "metformin"
    entity_type: str = "MEDICATION"
    start_char: int = 10
    end_char: int = 19
    confidence: float = 0.95
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


@dataclass
class FakeICDPrediction:
    code: str = "E11.9"
    description: str = "Type 2 diabetes mellitus without complications"
    confidence: float = 0.85
    chapter: str = "Endocrine, nutritional and metabolic diseases"
    category: str | None = None
    contributing_text: list[str] | None = None


@dataclass
class FakeICDResult:
    predictions: list[FakeICDPrediction] = field(default_factory=lambda: [FakeICDPrediction()])


@dataclass
class FakeSummarizationResult:
    summary: str = "Patient has diabetes managed with metformin."
    key_findings: list[str] = field(default_factory=lambda: ["diabetes", "metformin"])
    detail_level: str = "standard"
    processing_time_ms: float = 10.0
    model_name: str = "textrank"
    model_version: str = "1.0.0"
    sentence_count_original: int = 5
    sentence_count_summary: int = 1


@dataclass
class FakeRiskFactor:
    name: str = "polypharmacy"
    description: str = "Multiple medications detected"
    weight: float = 0.5
    score: float = 0.6
    category: str = "medication_risk"


@dataclass
class FakeRiskAssessment:
    overall_score: float = 45.0
    factors: list[FakeRiskFactor] = field(default_factory=lambda: [FakeRiskFactor()])
    category_scores: dict[str, float] = field(
        default_factory=lambda: {"medication_risk": 45.0}
    )
    recommendations: list[str] = field(
        default_factory=lambda: ["Medication review recommended"]
    )


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _mock_ner_model(entities: list[FakeEntity] | None = None) -> MagicMock:
    model = MagicMock()
    model.extract_entities.return_value = entities or [FakeEntity()]
    return model


def _mock_icd_model(result: FakeICDResult | None = None) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = result or FakeICDResult()
    return model


def _mock_summarizer(result: FakeSummarizationResult | None = None) -> MagicMock:
    model = MagicMock()
    model.summarize.return_value = result or FakeSummarizationResult()
    model.model_name = "textrank"
    model.version = "1.0.0"
    return model


def _mock_risk_scorer(assessment: FakeRiskAssessment | None = None) -> MagicMock:
    model = MagicMock()
    model.assess_risk.return_value = assessment or FakeRiskAssessment()
    model.model_name = "rule-based-risk"
    model.version = "1.0.0"
    return model


CLINICAL_TEXT = (
    "Patient is a 55-year-old male with type 2 diabetes mellitus on metformin 1000 mg BID. "
    "History of hypertension, currently on lisinopril 10 mg daily."
)


# ---------------------------------------------------------------------------
# _run_ner
# ---------------------------------------------------------------------------


class TestRunNer:
    """Tests for the _run_ner helper."""

    @patch("app.api.v1.routes.analyze.get_ner_model")
    def test_returns_entities_above_threshold(self, mock_get: MagicMock) -> None:
        """Entities with confidence >= min_confidence are returned."""
        mock_get.return_value = _mock_ner_model([
            FakeEntity(text="metformin", confidence=0.9),
            FakeEntity(text="aspirin", confidence=0.3),
        ])
        result = _run_ner(CLINICAL_TEXT, min_confidence=0.5)
        assert len(result) == 1
        assert result[0].text == "metformin"

    @patch("app.api.v1.routes.analyze.get_ner_model")
    def test_returns_all_when_threshold_zero(self, mock_get: MagicMock) -> None:
        """With min_confidence=0.0, all entities pass."""
        mock_get.return_value = _mock_ner_model([
            FakeEntity(confidence=0.1),
            FakeEntity(confidence=0.9),
        ])
        result = _run_ner(CLINICAL_TEXT, min_confidence=0.0)
        assert len(result) == 2

    @patch("app.api.v1.routes.analyze.get_ner_model")
    def test_empty_entities_from_model(self, mock_get: MagicMock) -> None:
        """Empty entity list yields empty response."""
        model = MagicMock()
        model.extract_entities.return_value = []
        mock_get.return_value = model
        result = _run_ner(CLINICAL_TEXT, min_confidence=0.0)
        assert result == []

    @patch("app.api.v1.routes.analyze.get_ner_model")
    def test_entity_fields_mapped_correctly(self, mock_get: MagicMock) -> None:
        """All Entity fields are transferred to EntityResponse."""
        entity = FakeEntity(
            text="diabetes",
            entity_type="DISEASE",
            start_char=30,
            end_char=38,
            confidence=0.92,
            normalized_text="Diabetes",
            umls_cui="C0011849",
            is_negated=True,
            is_uncertain=False,
        )
        mock_get.return_value = _mock_ner_model([entity])
        result = _run_ner(CLINICAL_TEXT, min_confidence=0.0)
        r = result[0]
        assert r.text == "diabetes"
        assert r.entity_type == "DISEASE"
        assert r.start_char == 30
        assert r.end_char == 38
        assert r.confidence == 0.92
        assert r.normalized_text == "Diabetes"
        assert r.umls_cui == "C0011849"
        assert r.is_negated is True
        assert r.is_uncertain is False


# ---------------------------------------------------------------------------
# _run_icd
# ---------------------------------------------------------------------------


class TestRunIcd:
    """Tests for the _run_icd helper."""

    @patch("app.api.v1.routes.analyze.get_icd_model")
    def test_predictions_returned_above_threshold(self, mock_get: MagicMock) -> None:
        """Only predictions meeting min_confidence are kept."""
        preds = [
            FakeICDPrediction(code="E11.9", confidence=0.85),
            FakeICDPrediction(code="I10", confidence=0.2),
        ]
        mock_get.return_value = _mock_icd_model(FakeICDResult(predictions=preds))
        result = _run_icd(CLINICAL_TEXT, top_k=10, min_confidence=0.5)
        assert len(result) == 1
        assert result[0].code == "E11.9"

    @patch("app.api.v1.routes.analyze.get_icd_model")
    def test_top_k_forwarded_to_model(self, mock_get: MagicMock) -> None:
        """The top_k parameter is forwarded to the underlying model."""
        model = _mock_icd_model()
        mock_get.return_value = model
        _run_icd(CLINICAL_TEXT, top_k=3, min_confidence=0.0)
        model.predict.assert_called_once_with(CLINICAL_TEXT, top_k=3)

    @patch("app.api.v1.routes.analyze.get_icd_model")
    def test_empty_predictions(self, mock_get: MagicMock) -> None:
        """No predictions from model → empty list."""
        mock_get.return_value = _mock_icd_model(FakeICDResult(predictions=[]))
        result = _run_icd(CLINICAL_TEXT, top_k=10, min_confidence=0.0)
        assert result == []


# ---------------------------------------------------------------------------
# _run_summary
# ---------------------------------------------------------------------------


class TestRunSummary:
    """Tests for the _run_summary helper."""

    @patch("app.api.v1.routes.analyze.get_summarizer")
    def test_compression_ratio_computed(self, mock_get: MagicMock) -> None:
        """Compression ratio = original_wc / summary_wc."""
        res = FakeSummarizationResult(
            summary="Brief summary here.",
            key_findings=["finding"],
        )
        mock_get.return_value = _mock_summarizer(res)
        resp = _run_summary("word " * 30, "standard")
        assert resp.original_word_count == 30
        assert resp.summary_word_count == 3  # "Brief summary here."
        assert resp.compression_ratio == round(30 / 3, 2)

    @patch("app.api.v1.routes.analyze.get_summarizer")
    def test_detail_level_forwarded(self, mock_get: MagicMock) -> None:
        """detail_level is forwarded to the summarizer."""
        model = _mock_summarizer()
        mock_get.return_value = model
        _run_summary(CLINICAL_TEXT, "brief")
        model.summarize.assert_called_once_with(CLINICAL_TEXT, detail_level="brief")

    @patch("app.api.v1.routes.analyze.get_summarizer")
    def test_summary_type_is_extractive(self, mock_get: MagicMock) -> None:
        """Response summary_type should always be 'extractive'."""
        mock_get.return_value = _mock_summarizer()
        resp = _run_summary(CLINICAL_TEXT, "standard")
        assert resp.summary_type == "extractive"


# ---------------------------------------------------------------------------
# _run_risk
# ---------------------------------------------------------------------------


class TestRunRisk:
    """Tests for the _run_risk helper."""

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_low_risk_category(self, mock_get: MagicMock) -> None:
        """Score < 40 → category 'low'."""
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(overall_score=20.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert result.category == "low"

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_moderate_risk_category(self, mock_get: MagicMock) -> None:
        """40 <= score < 60 → 'moderate'."""
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(overall_score=50.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert result.category == "moderate"

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_high_risk_category(self, mock_get: MagicMock) -> None:
        """60 <= score < 80 → 'high'."""
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(overall_score=70.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert result.category == "high"

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_critical_risk_category(self, mock_get: MagicMock) -> None:
        """Score >= 80 → 'critical'."""
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(overall_score=85.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert result.category == "critical"

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_score_normalised_to_0_1(self, mock_get: MagicMock) -> None:
        """Overall score is divided by 100 for the API response."""
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(overall_score=45.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert result.score == round(45.0 / 100.0, 4)

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_top_factors_capped_at_5(self, mock_get: MagicMock) -> None:
        """At most 5 factors appear in the risk summary."""
        factors = [FakeRiskFactor(name=f"f{i}", score=0.8) for i in range(10)]
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(factors=factors, overall_score=60.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        assert len(result.top_factors) <= 5

    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    def test_zero_score_factors_excluded(self, mock_get: MagicMock) -> None:
        """Factors with score == 0 are not included."""
        factors = [
            FakeRiskFactor(name="active", score=0.5),
            FakeRiskFactor(name="inactive", score=0.0),
        ]
        mock_get.return_value = _mock_risk_scorer(
            FakeRiskAssessment(factors=factors, overall_score=30.0)
        )
        result = _run_risk(CLINICAL_TEXT)
        names = [f.name for f in result.top_factors]
        assert "active" in names
        assert "inactive" not in names


# ---------------------------------------------------------------------------
# run_analysis (full pipeline endpoint)
# ---------------------------------------------------------------------------


def _make_request(
    text: str = CLINICAL_TEXT,
    *,
    ner_enabled: bool = True,
    icd_enabled: bool = True,
    summary_enabled: bool = True,
    risk_enabled: bool = True,
    document_id: str | None = None,
) -> AnalysisRequest:
    """Build an AnalysisRequest with selective stages."""
    return AnalysisRequest(
        text=text,
        config=PipelineConfig(
            ner=NERConfig(enabled=ner_enabled),
            icd=ICDConfig(enabled=icd_enabled),
            summary=SummaryConfig(enabled=summary_enabled),
            risk=RiskConfig(enabled=risk_enabled),
        ),
        document_id=document_id,
    )


def _mock_request_obj() -> MagicMock:
    """Build a mock ASGI Request with client info."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {"user-agent": "pytest/1.0"}
    return req


class TestRunAnalysis:
    """Tests for the run_analysis endpoint handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_all_stages_enabled(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """When all stages are enabled, all result fields are populated."""
        mock_ner.return_value = _mock_ner_model()
        mock_icd.return_value = _mock_icd_model()
        mock_sum.return_value = _mock_summarizer()
        mock_risk.return_value = _mock_risk_scorer()

        payload = _make_request()
        db = AsyncMock()
        resp = await run_analysis(payload, _mock_request_obj(), db, MagicMock())

        assert isinstance(resp, AnalysisResponse)
        assert resp.entities is not None
        assert resp.icd_codes is not None
        assert resp.summary is not None
        assert resp.risk_score is not None
        assert resp.timing.ner_ms is not None
        assert resp.timing.icd_ms is not None
        assert resp.timing.summary_ms is not None
        assert resp.timing.risk_ms is not None
        assert resp.timing.total_ms > 0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_ner_only(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """When only NER is enabled, other stages are null."""
        mock_ner.return_value = _mock_ner_model()

        payload = _make_request(
            ner_enabled=True, icd_enabled=False,
            summary_enabled=False, risk_enabled=False,
        )
        db = AsyncMock()
        resp = await run_analysis(payload, _mock_request_obj(), db, MagicMock())

        assert resp.entities is not None
        assert resp.icd_codes is None
        assert resp.summary is None
        assert resp.risk_score is None
        assert resp.timing.ner_ms is not None
        assert resp.timing.icd_ms is None

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_document_id_echoed(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """Client-supplied document_id is echoed in the response."""
        mock_ner.return_value = _mock_ner_model()
        mock_icd.return_value = _mock_icd_model()
        mock_sum.return_value = _mock_summarizer()
        mock_risk.return_value = _mock_risk_scorer()

        payload = _make_request(document_id="pat-001")
        db = AsyncMock()
        resp = await run_analysis(payload, _mock_request_obj(), db, MagicMock())
        assert resp.document_id == "pat-001"

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_text_length_in_response(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """Response text_length equals len(input text)."""
        mock_ner.return_value = _mock_ner_model()
        mock_icd.return_value = _mock_icd_model()
        mock_sum.return_value = _mock_summarizer()
        mock_risk.return_value = _mock_risk_scorer()

        payload = _make_request(text="Hello clinical world.")
        db = AsyncMock()
        resp = await run_analysis(payload, _mock_request_obj(), db, MagicMock())
        assert resp.text_length == len("Hello clinical world.")

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_ner_model_exception_raises_500(self, mock_get: MagicMock) -> None:
        """Unexpected exception during NER → HTTP 500."""
        from fastapi import HTTPException

        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(side_effect=RuntimeError("boom"))
        )
        payload = _make_request(
            icd_enabled=False, summary_enabled=False, risk_enabled=False,
        )
        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await run_analysis(payload, _mock_request_obj(), db, MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_document_processing_error_raises_422(self, mock_get: MagicMock) -> None:
        """DocumentProcessingError during pipeline → HTTP 422."""
        from fastapi import HTTPException
        from app.core.exceptions import DocumentProcessingError

        mock_get.return_value = MagicMock(
            extract_entities=MagicMock(side_effect=DocumentProcessingError("bad doc"))
        )
        payload = _make_request(
            icd_enabled=False, summary_enabled=False, risk_enabled=False,
        )
        db = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await run_analysis(payload, _mock_request_obj(), db, MagicMock())
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_audit_log_written(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """An audit log entry is added to the db session after analysis."""
        mock_ner.return_value = _mock_ner_model()
        mock_icd.return_value = _mock_icd_model()
        mock_sum.return_value = _mock_summarizer()
        mock_risk.return_value = _mock_risk_scorer()

        payload = _make_request()
        db = AsyncMock()
        await run_analysis(payload, _mock_request_obj(), db, MagicMock())

        # _write_audit_log calls db.add()
        db.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.analyze.get_risk_scorer")
    @patch("app.api.v1.routes.analyze.get_summarizer")
    @patch("app.api.v1.routes.analyze.get_icd_model")
    @patch("app.api.v1.routes.analyze.get_ner_model")
    async def test_audit_failure_does_not_break_response(
        self, mock_ner: MagicMock, mock_icd: MagicMock,
        mock_sum: MagicMock, mock_risk: MagicMock,
    ) -> None:
        """If audit logging raises, the response is still returned."""
        mock_ner.return_value = _mock_ner_model()
        mock_icd.return_value = _mock_icd_model()
        mock_sum.return_value = _mock_summarizer()
        mock_risk.return_value = _mock_risk_scorer()

        payload = _make_request()
        db = AsyncMock()
        db.add.side_effect = RuntimeError("audit db exploded")
        resp = await run_analysis(payload, _mock_request_obj(), db, MagicMock())

        # Should still get a valid response despite audit failure
        assert isinstance(resp, AnalysisResponse)
