"""ClinIQ SDK data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """Extracted medical entity."""

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


@dataclass
class ICDPrediction:
    """ICD-10 code prediction."""

    code: str
    description: str | None
    confidence: float
    chapter: str | None = None
    contributing_text: list[str] | None = None


@dataclass
class Summary:
    """Clinical text summary."""

    summary: str
    key_findings: list[str] = field(default_factory=list)
    detail_level: str = "standard"
    word_count: int = 0


@dataclass
class RiskFactor:
    """Risk assessment factor."""

    name: str
    score: float
    weight: float
    category: str
    description: str = ""


@dataclass
class RiskAssessment:
    """Patient risk assessment."""

    overall_score: float
    risk_level: str
    factors: list[RiskFactor] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Full pipeline analysis result."""

    entities: list[Entity] = field(default_factory=list)
    icd_predictions: list[ICDPrediction] = field(default_factory=list)
    summary: Summary | None = None
    risk_assessment: RiskAssessment | None = None
    processing_time_ms: float = 0.0
    model_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        """Create from API response dict."""
        entities = [Entity(**e) for e in (data.get("entities") or [])]
        icd_preds = [ICDPrediction(**p) for p in (data.get("icd_predictions") or [])]
        summary = None
        if data.get("summary"):
            summary = Summary(**data["summary"])
        risk = None
        if data.get("risk_assessment"):
            ra = data["risk_assessment"]
            factors = [RiskFactor(**f) for f in (ra.get("factors") or [])]
            risk = RiskAssessment(
                overall_score=ra["overall_score"],
                risk_level=ra["risk_level"],
                factors=factors,
                recommendations=ra.get("recommendations", []),
            )
        return cls(
            entities=entities,
            icd_predictions=icd_preds,
            summary=summary,
            risk_assessment=risk,
            processing_time_ms=data.get("processing_time_ms", 0),
            model_versions=data.get("model_versions", {}),
        )


@dataclass
class BatchJob:
    """Batch processing job status."""

    job_id: str
    status: str
    total_documents: int
    processed_documents: int = 0
    failed_documents: int = 0
    progress: float = 0.0
    result_file: str | None = None
