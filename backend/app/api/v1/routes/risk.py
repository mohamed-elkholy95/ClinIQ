"""Risk scoring endpoint routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps import PipelineDep
from app.api.v1.schemas import (
    RiskRequest,
    RiskResponse,
    RiskScoreResponse,
    RiskFactor,
)
from app.ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["risk-scoring"])


@router.post(
    "/score",
    response_model=RiskResponse,
    summary="Calculate risk score",
    description="Calculate patient risk score based on clinical text analysis.",
)
async def calculate_risk_score(
    request: RiskRequest,
    pipeline: PipelineDep,
) -> RiskResponse:
    """Calculate risk score from clinical text."""
    import time

    start_time = time.time()

    try:
        result = pipeline.calculate_risk(request.text)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk scoring service unavailable",
            )

        risk_score = RiskScoreResponse(
            overall_score=result.overall_score,
            risk_level=result.risk_level,
            category_scores=result.category_scores,
            risk_factors=[
                RiskFactor(
                    name=rf.name,
                    description=rf.description,
                    weight=rf.weight,
                    value=rf.value,
                    source=rf.source,
                    evidence=rf.evidence,
                )
                for rf in result.risk_factors
            ],
            protective_factors=[
                RiskFactor(
                    name=pf.name,
                    description=pf.description,
                    weight=pf.weight,
                    value=pf.value,
                    source=pf.source,
                    evidence=pf.evidence,
                )
                for pf in result.protective_factors
            ],
            recommendations=result.recommendations,
        )

        processing_time = (time.time() - start_time) * 1000

        return RiskResponse(
            risk_score=risk_score,
            processing_time_ms=processing_time,
            model_version=result.model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk scoring failed: {str(e)}",
        )


@router.post(
    "/analyze",
    summary="Detailed risk analysis",
    description="Get detailed risk analysis with category breakdowns and recommendations.",
)
async def analyze_risk_detailed(
    request: RiskRequest,
    pipeline: PipelineDep,
) -> dict:
    """Get detailed risk analysis."""
    try:
        # First extract entities for better risk analysis
        entities = pipeline.extract_entities(request.text)

        # Get ICD predictions
        icd_predictions = pipeline.predict_icd_codes(request.text, top_k=10)

        # Calculate risk with full context
        from app.ml.risk.scorer import RiskScorer

        scorer = RiskScorer()
        result = scorer.calculate_risk(
            request.text,
            entities=entities,
            icd_predictions=icd_predictions,
        )

        return {
            "overall_assessment": {
                "score": result.overall_score,
                "level": result.risk_level,
                "interpretation": _interpret_risk_level(result.risk_level),
            },
            "category_breakdown": [
                {
                    "category": cat,
                    "score": score,
                    "level": _score_to_level(score),
                    "description": _get_category_description(cat),
                }
                for cat, score in result.category_scores.items()
            ],
            "risk_factors": [
                {
                    "factor": rf.name,
                    "description": rf.description,
                    "weight": rf.weight,
                    "evidence": rf.evidence,
                }
                for rf in result.risk_factors[:10]
            ],
            "recommendations": result.recommendations,
            "entities_analyzed": len(entities),
            "icd_codes_considered": len(icd_predictions),
        }

    except Exception as e:
        logger.error(f"Detailed risk analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk analysis failed: {str(e)}",
        )


@router.post(
    "/categories",
    summary="Category scores only",
    description="Get risk scores for individual categories.",
)
async def get_category_scores(
    request: RiskRequest,
    pipeline: PipelineDep,
) -> dict:
    """Get risk scores by category."""
    try:
        result = pipeline.calculate_risk(request.text)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk scoring service unavailable",
            )

        return {
            "overall_score": result.overall_score,
            "categories": {
                cat: {
                    "score": score,
                    "level": _score_to_level(score),
                    "weight": pipeline._risk_scorer.category_weights.get(cat, 0)
                    if pipeline._risk_scorer
                    else 0,
                }
                for cat, score in result.category_scores.items()
            },
            "highest_risk_category": max(
                result.category_scores.items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Category scoring failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Category scoring failed: {str(e)}",
        )


def _interpret_risk_level(level: str) -> str:
    """Get interpretation of risk level."""
    interpretations = {
        "low": "Patient presents with minimal risk factors. Standard monitoring and follow-up recommended.",
        "moderate": "Some risk factors identified. Enhanced monitoring and timely follow-up advised.",
        "high": "Significant risk factors present. Close monitoring and intervention may be required.",
        "critical": "Critical risk level detected. Immediate attention and urgent intervention recommended.",
    }
    return interpretations.get(level, "Risk level interpretation unavailable.")


def _score_to_level(score: float) -> str:
    """Convert score to risk level."""
    if score >= 0.8:
        return "critical"
    elif score >= 0.6:
        return "high"
    elif score >= 0.4:
        return "moderate"
    else:
        return "low"


def _get_category_description(category: str) -> str:
    """Get description for risk category."""
    descriptions = {
        "medication": "Medication-related risks including polypharmacy and drug interactions",
        "cardiovascular": "Cardiovascular system risks including heart disease and hypertension",
        "infection": "Infection-related risks including immunosuppression and sepsis",
        "surgical": "Surgical and procedural risks including complications",
        "follow_up": "Follow-up compliance risks including missed appointments",
    }
    return descriptions.get(category, "Risk category")
