"""ICD-10 code prediction endpoint.

Exposes a standalone ICD-10 classification endpoint that predicts diagnosis
codes from free clinical text without running the full analysis pipeline.
Predictions are produced by the real rule-based ICD classifier loaded via
the model registry.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.icd import ICDCodeResponse, ICDPredictionRequest, ICDPredictionResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session
from app.services.model_registry import get_icd_model

router = APIRouter(tags=["icd"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helper — delegates to the real ML model
# ---------------------------------------------------------------------------


def _run_icd_inference(request: ICDPredictionRequest) -> list[ICDCodeResponse]:
    """Run ICD-10 classification using the model registry.

    Parameters
    ----------
    request:
        Validated ICD prediction request with text and optional filters.

    Returns
    -------
    list[ICDCodeResponse]
        Ranked ICD-10 code predictions filtered per request parameters.
    """
    model = get_icd_model()
    result = model.predict(request.text, top_k=request.top_k)

    predictions: list[ICDCodeResponse] = []
    for pred in result.predictions:
        if pred.confidence < request.min_confidence:
            continue
        resp = ICDCodeResponse(
            code=pred.code,
            description=pred.description,
            confidence=pred.confidence,
            chapter=pred.chapter if request.include_chapter else None,
            category=pred.category if request.include_chapter else None,
            contributing_text=pred.contributing_text,
        )
        predictions.append(resp)

    return predictions


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/icd-predict",
    response_model=ICDPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="ICD-10 code prediction",
    description=(
        "Predict ICD-10-CM diagnosis codes from the supplied clinical text. "
        "Predictions are ranked by confidence and can be filtered by a minimum "
        "confidence threshold. The number of returned codes is controlled by `top_k`. "
        "Each prediction optionally includes the ICD-10 chapter / category and the "
        "text spans that contributed most to the prediction."
    ),
    responses={
        200: {"description": "Predictions returned successfully"},
        422: {"description": "Input validation error"},
        500: {"description": "ICD prediction inference error"},
    },
)
async def predict_icd_codes(
    payload: ICDPredictionRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ICDPredictionResponse:
    """Run ICD-10 classification and return ranked code predictions."""
    t0 = time.monotonic()

    try:
        predictions = _run_icd_inference(payload)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ICD-10 prediction failed unexpectedly. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - t0) * 1000

    return ICDPredictionResponse(
        predictions=predictions,
        prediction_count=len(predictions),
        model_name=payload.model,
        model_version="1.0.0",
        processing_time_ms=elapsed_ms,
        document_summary=None,
    )


@router.get(
    "/icd-codes/{code}",
    summary="ICD-10 code lookup",
    description=(
        "Look up the description and chapter information for a specific ICD-10-CM code. "
        "Returns HTTP 404 if the code is not found in the reference table."
    ),
    responses={
        200: {"description": "Code details returned"},
        404: {"description": "ICD-10 code not found"},
    },
)
async def get_icd_code_details(
    code: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict:
    """Return reference information for a single ICD-10-CM code.

    Queries the icd_codes reference table. Returns a 404 when the code is
    not present.
    """
    from sqlalchemy import select

    from app.db.models import ICDCode

    result = await db.execute(select(ICDCode).where(ICDCode.code == code.upper()))
    row = result.scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ICD-10 code '{code}' not found in the reference table.",
        )

    return {
        "code": row.code,
        "description": row.description,
        "chapter": row.chapter,
        "category": row.category,
        "is_active": row.is_active,
    }
