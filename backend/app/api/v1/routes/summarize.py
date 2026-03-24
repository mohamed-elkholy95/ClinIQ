"""Clinical summarization endpoint.

Generates concise, clinically accurate summaries of free-text clinical notes
using configurable extractive or abstractive strategies.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.summary import SummarizationRequest, SummarizationResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session

router = APIRouter(tags=["summarization"])


# ---------------------------------------------------------------------------
# Placeholder inference helper
# ---------------------------------------------------------------------------

_DETAIL_WORD_TARGETS: dict[str, int] = {
    "brief": 50,
    "standard": 150,
    "detailed": 300,
}


def _run_summarization(request: SummarizationRequest) -> SummarizationResponse:
    """Placeholder summarization; returns a mock summary.

    Replace with a call to the real summarization model once the ML layer
    is wired up.
    """
    original_words = request.text.split()
    original_word_count = len(original_words)

    # Determine target word count.
    target_words = request.max_length_words or _DETAIL_WORD_TARGETS.get(request.detail_level, 150)
    target_words = max(target_words, request.min_length_words or 5)

    # Naively take the first N words as a placeholder extractive summary.
    summary_words = original_words[:target_words]
    summary_text = " ".join(summary_words)
    if len(original_words) > target_words:
        summary_text += "..."

    summary_word_count = len(summary_words)
    compression_ratio = round(original_word_count / summary_word_count, 2) if summary_word_count else 1.0

    key_points: list[str] | None = None
    if request.include_key_points:
        key_points = [
            "[Placeholder key point 1 — wire up real model]",
            "[Placeholder key point 2 — wire up real model]",
        ]

    return SummarizationResponse(
        summary=f"[Placeholder summary] {summary_text}",
        key_points=key_points,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
        compression_ratio=compression_ratio,
        summary_type="extractive",
        model_name=request.model,
        model_version="1.0.0",
        processing_time_ms=0.0,
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
