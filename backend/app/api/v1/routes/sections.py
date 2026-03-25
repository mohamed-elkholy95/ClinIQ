"""API endpoints for clinical document section parsing.

Provides endpoints to parse clinical documents into their constituent
sections, identify section boundaries, and query section categories.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.sections.parser import ClinicalSectionParser, SectionCategory

router = APIRouter(tags=["sections"])

# Module-level singleton (stateless, thread-safe).
_parser = ClinicalSectionParser()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SectionParseRequest(BaseModel):
    """Request body for section parsing."""

    text: str = Field(..., min_length=1, description="Clinical note text")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence to include a section",
    )


class SectionBatchRequest(BaseModel):
    """Request body for batch section parsing."""

    texts: list[str] = Field(..., min_length=1, max_length=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PositionQueryRequest(BaseModel):
    """Request body for position-in-section queries."""

    text: str = Field(..., min_length=1)
    position: int = Field(..., ge=0, description="Character offset to query")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sections")
async def parse_sections(request: SectionParseRequest) -> dict[str, Any]:
    """Parse a clinical document into sections.

    Returns detected section headers with character offsets, categories,
    and confidence scores.
    """
    parser = ClinicalSectionParser(min_confidence=request.min_confidence)
    result = parser.parse(request.text)
    return result.to_dict()


@router.post("/sections/batch")
async def parse_sections_batch(request: SectionBatchRequest) -> dict[str, Any]:
    """Parse multiple documents into sections.

    Returns per-document section results plus aggregate statistics.
    """
    parser = ClinicalSectionParser(min_confidence=request.min_confidence)
    results = [parser.parse(t) for t in request.texts]

    # Aggregate stats.
    all_categories: set[str] = set()
    total_sections = 0
    for r in results:
        total_sections += len(r.sections)
        all_categories.update(str(c) for c in r.categories_found)

    return {
        "results": [r.to_dict() for r in results],
        "document_count": len(results),
        "total_sections": total_sections,
        "avg_sections": round(total_sections / len(results), 2) if results else 0,
        "all_categories_found": sorted(all_categories),
    }


@router.post("/sections/query")
async def query_section_at(request: PositionQueryRequest) -> dict[str, Any]:
    """Query which section a character position falls into.

    Returns the section containing the given position, or null if the
    position is in the preamble.
    """
    if request.position > len(request.text):
        raise HTTPException(
            status_code=422,
            detail=f"Position {request.position} exceeds text length {len(request.text)}",
        )

    section = _parser.get_section_at(request.text, request.position)
    return {
        "position": request.position,
        "section": section.to_dict() if section else None,
        "in_section": section is not None,
    }


@router.get("/sections/categories")
async def list_categories() -> dict[str, Any]:
    """List all supported section categories with descriptions."""
    descriptions = ClinicalSectionParser.get_category_descriptions()
    categories = [
        {"category": cat, "description": desc}
        for cat, desc in descriptions.items()
        if cat != SectionCategory.UNKNOWN
    ]
    return {
        "categories": categories,
        "count": len(categories),
    }
