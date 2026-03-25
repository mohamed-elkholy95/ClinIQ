"""Clinical note quality analysis endpoints.

Provides quality scoring for clinical notes to assess their suitability
for NLP processing *before* running the full inference pipeline.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.ml.quality.analyzer import (
    ClinicalNoteQualityAnalyzer,
    QualityConfig,
    QualityDimension,
    QualityReport,
)

router = APIRouter(tags=["quality"])

# Module-level singleton — lightweight, no model loading required
_analyzer = ClinicalNoteQualityAnalyzer()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QualityRequest(BaseModel):
    """Request body for single-note quality analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Clinical note text to analyze for quality.",
    )
    expected_sections: list[str] | None = Field(
        default=None,
        description=(
            "Override default expected sections. "
            "If provided, completeness scoring uses these instead of the "
            "default (Chief Complaint, HPI, Assessment, Plan)."
        ),
    )


class QualityBatchRequest(BaseModel):
    """Request body for batch quality analysis."""

    documents: list[QualityRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Up to 100 clinical notes to analyze.",
    )


class FindingResponse(BaseModel):
    """A single quality finding."""

    dimension: str
    severity: str
    message: str
    detail: str | None = None


class DimensionScoreResponse(BaseModel):
    """Score for a single quality dimension."""

    dimension: str
    score: float
    weight: float
    findings: list[FindingResponse]


class QualityResponse(BaseModel):
    """Response for a quality analysis."""

    overall_score: float = Field(description="Overall quality score (0-100).")
    grade: str = Field(description="Letter grade: A/B/C/D/F.")
    dimensions: list[DimensionScoreResponse]
    recommendations: list[str]
    stats: dict[str, Any]
    text_hash: str
    analysis_ms: float


class QualityBatchResponse(BaseModel):
    """Response for batch quality analysis."""

    results: list[QualityResponse]
    summary: dict[str, Any] = Field(
        description="Aggregate statistics across the batch."
    )


class DimensionInfoResponse(BaseModel):
    """Information about a quality dimension."""

    dimension: str
    description: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _report_to_response(report: QualityReport) -> QualityResponse:
    """Convert internal report to API response."""
    return QualityResponse(
        overall_score=round(report.overall_score, 2),
        grade=report.grade,
        dimensions=[
            DimensionScoreResponse(
                dimension=d.dimension.value,
                score=round(d.score, 2),
                weight=round(d.weight, 3),
                findings=[
                    FindingResponse(
                        dimension=f.dimension.value,
                        severity=f.severity.value,
                        message=f.message,
                        detail=f.detail,
                    )
                    for f in d.findings
                ],
            )
            for d in report.dimensions
        ],
        recommendations=report.recommendations,
        stats={
            k: v
            for k, v in report.stats.items()
            if k != "sentences"  # Exclude raw sentence list from API response
        },
        text_hash=report.text_hash,
        analysis_ms=round(report.analysis_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/quality",
    response_model=QualityResponse,
    summary="Analyze clinical note quality",
    description=(
        "Scores a clinical note across five quality dimensions: completeness, "
        "readability, structure, information density, and consistency.  Returns "
        "a composite score (0–100), letter grade, per-dimension breakdowns, "
        "and actionable recommendations."
    ),
    responses={
        200: {"description": "Quality report generated"},
        422: {"description": "Input validation error"},
    },
)
async def analyze_quality(
    payload: QualityRequest,
) -> QualityResponse:
    """Analyze quality of a single clinical note."""
    analyzer = _analyzer
    if payload.expected_sections:
        config = QualityConfig(expected_sections=payload.expected_sections)
        analyzer = ClinicalNoteQualityAnalyzer(config)

    report = analyzer.analyze(payload.text)
    return _report_to_response(report)


@router.post(
    "/quality/batch",
    response_model=QualityBatchResponse,
    summary="Batch quality analysis",
    description=(
        "Analyze up to 100 clinical notes and receive individual quality reports "
        "plus aggregate summary statistics."
    ),
    responses={
        200: {"description": "Batch quality reports generated"},
        422: {"description": "Input validation error"},
    },
)
async def analyze_quality_batch(
    payload: QualityBatchRequest,
) -> QualityBatchResponse:
    """Analyze quality of multiple clinical notes."""
    results: list[QualityResponse] = []
    scores: list[float] = []

    for doc in payload.documents:
        analyzer = _analyzer
        if doc.expected_sections:
            config = QualityConfig(expected_sections=doc.expected_sections)
            analyzer = ClinicalNoteQualityAnalyzer(config)

        report = analyzer.analyze(doc.text)
        results.append(_report_to_response(report))
        scores.append(report.overall_score)

    # Aggregate summary
    grade_counts: dict[str, int] = {}
    for r in results:
        grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1

    summary = {
        "total": len(results),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_score": round(min(scores), 2) if scores else 0,
        "max_score": round(max(scores), 2) if scores else 0,
        "grade_distribution": grade_counts,
    }

    return QualityBatchResponse(results=results, summary=summary)


@router.get(
    "/quality/dimensions",
    response_model=list[DimensionInfoResponse],
    summary="List quality dimensions",
    description="Returns the five quality dimensions with descriptions.",
)
async def list_dimensions() -> list[DimensionInfoResponse]:
    """Return catalogue of quality dimensions."""
    dimension_descriptions = {
        QualityDimension.COMPLETENESS: (
            "Measures note completeness based on word count and "
            "presence of expected clinical sections (Chief Complaint, HPI, Assessment, Plan)."
        ),
        QualityDimension.READABILITY: (
            "Evaluates sentence length, abbreviation density, and "
            "overall readability for NLP processing."
        ),
        QualityDimension.STRUCTURE: (
            "Assesses structural quality including section headers, "
            "whitespace ratio, list usage, and formatting consistency."
        ),
        QualityDimension.INFORMATION_DENSITY: (
            "Measures the concentration of medical terms, numeric "
            "measurements, and clinically relevant content."
        ),
        QualityDimension.CONSISTENCY: (
            "Checks for duplicate paragraphs, contradictory assertion "
            "modifiers, and other internal consistency issues."
        ),
    }

    return [
        DimensionInfoResponse(
            dimension=dim.value,
            description=desc,
        )
        for dim, desc in dimension_descriptions.items()
    ]
