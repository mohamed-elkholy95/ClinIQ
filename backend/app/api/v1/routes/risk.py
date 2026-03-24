"""Clinical risk scoring endpoint.

Scores patient risk across multiple clinical domains (readmission, mortality,
sepsis, falls, etc.) based on free-text clinical notes.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.risk import RiskFactorResponse, RiskScoreRequest, RiskScoreResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session

router = APIRouter(tags=["risk"])


# ---------------------------------------------------------------------------
# Placeholder inference helper
# ---------------------------------------------------------------------------

_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "medication": "Medication-related risks including polypharmacy and high-risk drug interactions",
    "cardiovascular": "Cardiovascular risks including acute cardiac conditions and hypertension",
    "infection": "Infection-related risks including immunosuppression and sepsis indicators",
    "surgical": "Surgical and procedural complication risks",
    "follow_up": "Follow-up compliance and care-transition risks",
}


def _score_to_category(score: float) -> str:
    """Map a numeric score to a qualitative category label."""
    if score >= 0.8:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.4:
        return "moderate"
    return "low"


def _run_risk_scoring(request: RiskScoreRequest) -> RiskScoreResponse:
    """Placeholder risk scoring; returns mock scores.

    Replace with a call to the real risk model once the ML layer is wired.
    """
    category_scores: dict[str, float] = {
        "medication": 0.45,
        "cardiovascular": 0.60,
        "infection": 0.25,
        "surgical": 0.10,
        "follow_up": 0.35,
    }

    # Apply any caller-supplied weight overrides (placeholder — weights don't
    # affect the mock scores yet).
    if request.category_weights:
        # Validate keys; unknown keys are ignored for now.
        pass

    # Filter to requested domains.
    if request.risk_domains:
        category_scores = {k: v for k, v in category_scores.items() if k in request.risk_domains}

    overall = round(sum(category_scores.values()) / len(category_scores), 4) if category_scores else 0.0
    overall_category = _score_to_category(overall)

    risk_factors = [
        RiskFactorResponse(
            name="placeholder_cardiovascular_factor",
            description="Placeholder cardiovascular risk factor — wire up real model.",
            weight=0.6,
            value=0.6,
            source="text",
            evidence=None,
        )
    ]
    protective_factors: list[RiskFactorResponse] = []
    recommendations = [
        "Placeholder recommendation — wire up real risk model.",
        "Consider specialist review for elevated cardiovascular category score.",
    ]

    return RiskScoreResponse(
        score=overall,
        category=overall_category,  # type: ignore[arg-type]
        category_scores=category_scores,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        recommendations=recommendations,
        model_name="rule-based-risk",
        model_version="1.0.0",
        processing_time_ms=0.0,
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
