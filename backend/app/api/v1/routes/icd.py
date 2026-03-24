"""ICD-10 prediction endpoint routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps import PipelineDep
from app.api.v1.schemas import ICDRequest, ICDResponse, ICDCodePrediction
from app.ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/icd", tags=["icd-10-prediction"])


@router.post(
    "/predict",
    response_model=ICDResponse,
    summary="Predict ICD-10 codes",
    description="Predict ICD-10 diagnosis codes from clinical text with confidence scores.",
)
async def predict_icd_codes(
    request: ICDRequest,
    pipeline: PipelineDep,
) -> ICDResponse:
    """Predict ICD-10 codes from clinical text."""
    import time

    start_time = time.time()

    try:
        predictions = pipeline.predict_icd_codes(request.text, top_k=request.top_k)

        # Build response
        icd_predictions = [
            ICDCodePrediction(
                code=p["code"],
                description=p.get("description"),
                confidence=p["confidence"],
                chapter=p.get("chapter"),
                category=p.get("category"),
            )
            for p in predictions
        ]

        processing_time = (time.time() - start_time) * 1000

        return ICDResponse(
            predictions=icd_predictions,
            processing_time_ms=processing_time,
            model_version=pipeline._icd_model.version if pipeline._icd_model else "unknown",
        )

    except Exception as e:
        logger.error(f"ICD-10 prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ICD-10 prediction failed: {str(e)}",
        )


@router.post(
    "/explain",
    summary="Explain ICD-10 predictions",
    description="Get ICD-10 predictions with explanation of contributing text segments.",
)
async def explain_icd_predictions(
    request: ICDRequest,
    pipeline: PipelineDep,
) -> dict:
    """Get ICD-10 predictions with explanations."""
    try:
        predictions = pipeline.predict_icd_codes(request.text, top_k=request.top_k)

        # Add explanations
        result = {
            "text": request.text[:500] + "..." if len(request.text) > 500 else request.text,
            "predictions": [],
            "chapter_distribution": {},
        }

        # Count by chapter
        from collections import Counter
        chapters = Counter()

        for pred in predictions:
            explanation = {
                "code": pred["code"],
                "description": pred.get("description"),
                "confidence": pred["confidence"],
                "chapter": pred.get("chapter"),
                "category": pred.get("category"),
                "contributing_segments": pred.get("contributing_text", []),
            }
            result["predictions"].append(explanation)

            if pred.get("chapter"):
                chapters[pred["chapter"]] += 1

        result["chapter_distribution"] = dict(chapters)

        return result

    except Exception as e:
        logger.error(f"ICD-10 explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ICD-10 explanation failed: {str(e)}",
        )


@router.get(
    "/codes/{code}",
    summary="Get ICD-10 code details",
    description="Get details about a specific ICD-10 code.",
)
async def get_icd_code_details(code: str) -> dict:
    """Get details for an ICD-10 code."""
    from app.ml.icd.model import get_chapter_for_code

    # This would normally look up in database
    # For now, return basic info
    chapter = get_chapter_for_code(code)

    return {
        "code": code,
        "chapter": chapter,
        "description": "ICD-10 code details (would be fetched from database)",
    }
