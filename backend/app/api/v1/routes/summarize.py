"""Clinical summarization endpoint.

Generates concise, clinically accurate summaries of free-text clinical notes
using the extractive TextRank summarizer loaded via the model registry.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.summary import SummarizationRequest, SummarizationResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session
from app.services.model_registry import get_summarizer

router = APIRouter(tags=["summarization"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helper — delegates to the real ML model
# ---------------------------------------------------------------------------


def _run_summarization(request: SummarizationRequest) -> SummarizationResponse:
    """Run clinical summarization using the model registry.

    Parameters
    ----------
    request:
        Validated summarization request with text and configuration.

    Returns
    -------
    SummarizationResponse
        Generated summary with metadata.
    """
    model = get_summarizer()
    result = model.summarize(request.text, detail_level=request.detail_level)

    original_wc = len(request.text.split())
    summary_wc = len(result.summary.split())
    compression = round(original_wc / summary_wc, 2) if summary_wc else 1.0

    return SummarizationResponse(
        summary=result.summary,
        key_points=result.key_findings if request.include_key_points else None,
        original_word_count=original_wc,
        summary_word_count=summary_wc,
        compression_ratio=compression,
        summary_type="extractive",
        model_name=model.model_name,
        model_version=model.version,
        processing_time_ms=0.0,  # Overwritten by route handler.
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/summarize",
    response_model=SummarizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Clinical text summarization",
    description=(
        "Generate a clinically accurate summary of the supplied clinical text. "
        "The verbosity is controlled by `detail_level` ('brief', 'standard', 'detailed') "
        "and the summarization strategy is set via `model` ('extractive', 'section-based', "
        "'abstractive', 'hybrid'). "
        "Optional `key_points` returns a bullet-point list alongside the prose summary."
    ),
    responses={
        200: {"description": "Summary generated successfully"},
        422: {"description": "Input validation error"},
        500: {"description": "Summarization inference error"},
    },
)
async def summarize_clinical_text(
    payload: SummarizationRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SummarizationResponse:
    """Run the summarization model and return the generated summary."""
    t0 = time.monotonic()

    try:
        result = _run_summarization(payload)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summarization failed unexpectedly. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - t0) * 1000
    result.processing_time_ms = elapsed_ms
    return result
