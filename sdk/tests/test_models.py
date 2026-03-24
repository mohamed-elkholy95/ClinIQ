"""Tests for ClinIQ SDK data models.

Validates dataclass construction, defaults, and the AnalysisResult.from_dict
factory that transforms raw API JSON into typed model instances.
"""

import pytest

from cliniq_client.models import (
    AnalysisResult,
    BatchJob,
    Entity,
    ICDPrediction,
    RiskAssessment,
    RiskFactor,
    Summary,
)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_required_fields(self) -> None:
        """Entity requires text, entity_type, start/end char, and confidence."""
        entity = Entity(
            text="metformin",
            entity_type="MEDICATION",
            start_char=10,
            end_char=19,
            confidence=0.95,
        )
        assert entity.text == "metformin"
        assert entity.entity_type == "MEDICATION"
        assert entity.start_char == 10
        assert entity.end_char == 19
        assert entity.confidence == 0.95

    def test_defaults(self) -> None:
        """Optional fields default to None/False."""
        entity = Entity("x", "T", 0, 1, 0.5)
        assert entity.normalized_text is None
        assert entity.umls_cui is None
        assert entity.is_negated is False
        assert entity.is_uncertain is False

    def test_optional_fields(self) -> None:
        """All optional fields can be set."""
        entity = Entity(
            text="aspirin",
            entity_type="MEDICATION",
            start_char=0,
            end_char=7,
            confidence=0.99,
            normalized_text="acetylsalicylic acid",
            umls_cui="C0004057",
            is_negated=True,
            is_uncertain=True,
        )
        assert entity.normalized_text == "acetylsalicylic acid"
        assert entity.umls_cui == "C0004057"
        assert entity.is_negated is True
        assert entity.is_uncertain is True


# ---------------------------------------------------------------------------
# ICDPrediction
# ---------------------------------------------------------------------------


class TestICDPrediction:
    """Tests for the ICDPrediction dataclass."""

    def test_minimal(self) -> None:
        pred = ICDPrediction(code="I21.9", description="AMI", confidence=0.87)
        assert pred.code == "I21.9"
        assert pred.chapter is None
        assert pred.contributing_text is None

    def test_full(self) -> None:
        pred = ICDPrediction(
            code="E11.9",
            description="Type 2 diabetes",
            confidence=0.74,
            chapter="Endocrine diseases",
            contributing_text=["type 2 diabetes", "metformin"],
        )
        assert pred.chapter == "Endocrine diseases"
        assert len(pred.contributing_text) == 2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    """Tests for the Summary dataclass."""

    def test_defaults(self) -> None:
        summary = Summary(summary="Patient stable.")
        assert summary.key_findings == []
        assert summary.detail_level == "standard"
        assert summary.word_count == 0

    def test_full(self) -> None:
        summary = Summary(
            summary="Pt stable post-op.",
            key_findings=["stable vitals", "no infection"],
            detail_level="detailed",
            word_count=4,
        )
        assert len(summary.key_findings) == 2
        assert summary.word_count == 4


# ---------------------------------------------------------------------------
# RiskFactor / RiskAssessment
# ---------------------------------------------------------------------------


class TestRiskAssessment:
    """Tests for RiskFactor and RiskAssessment dataclasses."""

    def test_risk_factor(self) -> None:
        factor = RiskFactor(
            name="polypharmacy",
            score=0.8,
            weight=0.6,
            category="medication",
            description="Multiple medications",
        )
        assert factor.name == "polypharmacy"
        assert factor.description == "Multiple medications"

    def test_risk_factor_default_description(self) -> None:
        factor = RiskFactor(name="x", score=0.1, weight=0.1, category="c")
        assert factor.description == ""

    def test_risk_assessment_defaults(self) -> None:
        assessment = RiskAssessment(overall_score=0.5, risk_level="moderate")
        assert assessment.factors == []
        assert assessment.recommendations == []


# ---------------------------------------------------------------------------
# AnalysisResult.from_dict
# ---------------------------------------------------------------------------


class TestAnalysisResultFromDict:
    """Tests for the AnalysisResult.from_dict factory method."""

    def test_empty_response(self) -> None:
        """Empty dict produces an AnalysisResult with sensible defaults."""
        result = AnalysisResult.from_dict({})
        assert result.entities == []
        assert result.icd_predictions == []
        assert result.summary is None
        assert result.risk_assessment is None
        assert result.processing_time_ms == 0.0
        assert result.model_versions == {}

    def test_entities_only(self) -> None:
        data = {
            "entities": [
                {
                    "text": "aspirin",
                    "entity_type": "MEDICATION",
                    "start_char": 0,
                    "end_char": 7,
                    "confidence": 0.95,
                }
            ]
        }
        result = AnalysisResult.from_dict(data)
        assert len(result.entities) == 1
        assert result.entities[0].text == "aspirin"

    def test_icd_predictions(self) -> None:
        data = {
            "icd_predictions": [
                {"code": "I21.9", "description": "AMI", "confidence": 0.87}
            ]
        }
        result = AnalysisResult.from_dict(data)
        assert len(result.icd_predictions) == 1
        assert result.icd_predictions[0].code == "I21.9"

    def test_summary(self) -> None:
        data = {"summary": {"summary": "Stable patient.", "key_findings": ["stable"]}}
        result = AnalysisResult.from_dict(data)
        assert result.summary is not None
        assert result.summary.summary == "Stable patient."

    def test_risk_assessment(self) -> None:
        data = {
            "risk_assessment": {
                "overall_score": 0.83,
                "risk_level": "critical",
                "factors": [
                    {
                        "name": "cardiac",
                        "score": 0.9,
                        "weight": 0.8,
                        "category": "cardiovascular",
                    }
                ],
                "recommendations": ["Immediate review"],
            }
        }
        result = AnalysisResult.from_dict(data)
        assert result.risk_assessment is not None
        assert result.risk_assessment.risk_level == "critical"
        assert len(result.risk_assessment.factors) == 1
        assert result.risk_assessment.recommendations == ["Immediate review"]

    def test_full_response(self) -> None:
        """Full pipeline response round-trips correctly."""
        data = {
            "entities": [
                {
                    "text": "diabetes",
                    "entity_type": "CONDITION",
                    "start_char": 20,
                    "end_char": 28,
                    "confidence": 0.92,
                }
            ],
            "icd_predictions": [
                {"code": "E11.9", "description": "T2DM", "confidence": 0.74}
            ],
            "summary": {
                "summary": "Patient with diabetes.",
                "key_findings": ["diabetes"],
                "detail_level": "standard",
                "word_count": 3,
            },
            "risk_assessment": {
                "overall_score": 0.45,
                "risk_level": "moderate",
                "factors": [],
                "recommendations": [],
            },
            "processing_time_ms": 123.4,
            "model_versions": {"ner": "1.0.0", "icd": "1.0.0"},
        }
        result = AnalysisResult.from_dict(data)
        assert len(result.entities) == 1
        assert len(result.icd_predictions) == 1
        assert result.summary is not None
        assert result.risk_assessment is not None
        assert result.processing_time_ms == 123.4
        assert result.model_versions["ner"] == "1.0.0"

    def test_null_entities_and_predictions(self) -> None:
        """None values for entities/predictions are treated as empty lists."""
        data = {"entities": None, "icd_predictions": None}
        result = AnalysisResult.from_dict(data)
        assert result.entities == []
        assert result.icd_predictions == []


# ---------------------------------------------------------------------------
# BatchJob
# ---------------------------------------------------------------------------


class TestBatchJob:
    """Tests for the BatchJob dataclass."""

    def test_minimal(self) -> None:
        job = BatchJob(job_id="abc-123", status="pending", total_documents=10)
        assert job.job_id == "abc-123"
        assert job.processed_documents == 0
        assert job.failed_documents == 0
        assert job.progress == 0.0
        assert job.result_file is None

    def test_completed_job(self) -> None:
        job = BatchJob(
            job_id="xyz-789",
            status="completed",
            total_documents=50,
            processed_documents=50,
            failed_documents=2,
            progress=1.0,
            result_file="/results/xyz-789.json",
        )
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.result_file == "/results/xyz-789.json"
