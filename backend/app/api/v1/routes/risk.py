"""Clinical risk scoring endpoint.

Scores patient risk across multiple clinical domains (medication, diagnostic
complexity, follow-up urgency) based on free-text clinical notes using the
rule-based risk scorer from the model registry.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.risk import RiskFactorResponse, RiskScoreRequest, RiskScoreResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session
from app.services.model_registry import get_risk_scorer

router = APIRouter(tags=["risk"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helper — delegates to the real ML model
# ---------------------------------------------------------------------------


def _score_to_category(score: float) -> str:
    """Map a 0-100 numeric score to a qualitative category label.

    Parameters
    ----------
    score:
        Risk score on a 0-100 scale.

    Returns
    -------
    str
        One of 'critical', 'high', 'moderate', or 'low'.
    """
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 40:
        return "moderate"
    return "low"


def _run_risk_scoring(request: RiskScoreRequest) -> RiskScoreResponse:
    """Run risk scoring using the model registry.

    Parameters
    ----------
    request:
        Validated risk score request with text and optional filters.

    Returns
    -------
    RiskScoreResponse
        Risk assessment with per-category scores, factors, and recommendations.
    """
    model = get_risk_scorer()
    assessment = model.assess_risk(request.text)

    # Convert ML-layer RiskFactor objects to API schema.
    # The ML model's category field (e.g. "medication_risk") maps to the API's
    # source Literal; we use "derived" as the default since these are all
    # rule-derived factors.
    risk_factors = [
        RiskFactorResponse(
            name=f.name,
            description=f.description,
            weight=f.weight,
            value=f.score,
            source="derived",
            evidence=None,
        )
        for f in assessment.factors
        if f.score > 0
    ]

    # Protective factors are those with a negative effective weight (i.e. they
    # reduce risk).  For the rule-based scorer all factors are risk-additive,
    # so this list is typically empty.
    protective_factors: list[RiskFactorResponse] = []

    # Build category_scores from the assessment.
    category_scores: dict[str, float] = dict(assessment.category_scores)

    # Filter to requested domains if specified.
    if request.risk_domains:
        category_scores = {k: v for k, v in category_scores.items() if k in request.risk_domains}

    overall = assessment.overall_score
    overall_category = _score_to_category(overall)

    return RiskScoreResponse(
        score=round(overall / 100.0, 4),  # Normalise 0-100 → 0-1 for API.
        category=overall_category,  # type: ignore[arg-type]
        category_scores={k: round(v / 100.0, 4) for k, v in category_scores.items()},
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        recommendations=assessment.recommendations,
        model_name=model.model_name,
        model_version=model.version,
        processing_time_ms=0.0,  # Overwritten by route handler.
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/risk-score",
    response_model=RiskScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Clinical risk scoring",
    description=(
        "Score patient risk across clinical domains — readmission, mortality, sepsis, "
        "falls, pressure injury, and deterioration — from the supplied clinical text. "
        "Domains can be filtered via `risk_domains`. "
        "Optional `patient_context` (age, gender, comorbidities) can be provided to "
        "augment the text-based prediction. "
        "The response includes per-domain scores, contributing risk and protective "
        "factors, and actionable recommendations."
    ),
    responses={
        200: {"description": "Risk scores computed successfully"},
        422: {"description": "Input validation error"},
        500: {"description": "Risk scoring inference error"},
    },
)
async def calculate_risk_score(
    payload: RiskScoreRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RiskScoreResponse:
    """Run the risk scoring model and return per-domain scores."""
    t0 = time.monotonic()

    try:
        result = _run_risk_scoring(payload)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk scoring failed unexpectedly. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - t0) * 1000
    result.processing_time_ms = elapsed_ms
    return result
