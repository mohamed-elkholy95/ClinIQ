"""Temporal information extraction API endpoints.

Exposes date, duration, frequency, and temporal-relation extraction through
REST endpoints for clinical timeline construction.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.temporal.extractor import ClinicalTemporalExtractor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/temporal", tags=["temporal"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TemporalRequest(BaseModel):
    """Request body for temporal extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Clinical text to extract temporal information from",
    )
    reference_date: date | None = Field(
        default=None,
        description="Anchor date for resolving relative expressions (defaults to today)",
    )


class TemporalResponse(BaseModel):
    """Response body for temporal extraction."""

    expressions: list[dict[str, Any]]
    frequencies: list[dict[str, Any]]
    temporal_links: list[dict[str, Any]]
    reference_date: str
    expression_count: int
    frequency_count: int
    link_count: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=TemporalResponse,
    summary="Extract temporal information",
    description=(
        "Extracts dates, durations, frequencies, relative time references, "
        "and temporal relations from clinical text.  Relative expressions "
        "are resolved against the reference_date (defaults to today)."
    ),
)
async def extract_temporal(request: TemporalRequest) -> TemporalResponse:
    """Extract all temporal information from clinical text.

    Parameters
    ----------
    request:
        Clinical text and optional reference date.

    Returns
    -------
    TemporalResponse
        Extracted temporal expressions, frequencies, and links.
    """
    try:
        extractor = ClinicalTemporalExtractor(
            reference_date=request.reference_date,
        )
        result = extractor.extract(request.text)

        return TemporalResponse(
            expressions=[e.to_dict() for e in result.expressions],
            frequencies=[f.to_dict() for f in result.frequencies],
            temporal_links=[t.to_dict() for t in result.temporal_links],
            reference_date=result.reference_date.isoformat(),
            expression_count=len(result.expressions),
            frequency_count=len(result.frequencies),
            link_count=len(result.temporal_links),
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as exc:
        logger.exception("Temporal extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Temporal extraction error: {exc}",
        ) from exc


@router.get(
    "/frequency-map",
    summary="List known frequency abbreviations",
    description=(
        "Returns all recognised clinical frequency abbreviations with their "
        "normalised daily occurrence and interval values."
    ),
)
async def list_frequencies() -> dict[str, Any]:
    """Return the frequency abbreviation catalogue.

    Returns
    -------
    dict
        Mapping of abbreviations to normalised frequency data.
    """
    from app.ml.temporal.extractor import FREQUENCY_MAP

    entries = {}
    for abbrev, (times_per_day, interval_hours, as_needed) in FREQUENCY_MAP.items():
        entries[abbrev] = {
            "times_per_day": times_per_day,
            "interval_hours": interval_hours,
            "as_needed": as_needed,
        }

    return {
        "frequencies": entries,
        "count": len(entries),
    }
