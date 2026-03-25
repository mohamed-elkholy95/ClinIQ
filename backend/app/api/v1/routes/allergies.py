"""API endpoints for clinical allergy extraction.

Provides endpoints to extract drug, food, and environmental allergies from
clinical free text with reaction detection, severity classification, and
NKDA status.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ml.allergies.extractor import ClinicalAllergyExtractor

router = APIRouter(tags=["allergies"])

# Module-level singleton (stateless, thread-safe).
_extractor = ClinicalAllergyExtractor()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AllergyExtractionRequest(BaseModel):
    """Request body for allergy extraction."""

    text: str = Field(..., min_length=1, description="Clinical note text")
    min_confidence: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Minimum confidence to include an allergy",
    )


class AllergyBatchRequest(BaseModel):
    """Request body for batch allergy extraction."""

    texts: list[str] = Field(..., min_length=1, max_length=50)
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/allergies")
async def extract_allergies(request: AllergyExtractionRequest) -> dict[str, Any]:
    """Extract allergies from a clinical document.

    Detects drug, food, and environmental allergens with associated
    reactions, severity classification, and NKDA status.
    """
    extractor = ClinicalAllergyExtractor(min_confidence=request.min_confidence)
    result = extractor.extract(request.text)
    return result.to_dict()


@router.post("/allergies/batch")
async def extract_allergies_batch(request: AllergyBatchRequest) -> dict[str, Any]:
    """Extract allergies from multiple documents.

    Returns per-document results plus aggregate statistics.
    """
    extractor = ClinicalAllergyExtractor(min_confidence=request.min_confidence)
    results = extractor.extract_batch(request.texts)

    total_allergies = sum(len(r.allergies) for r in results)
    nkda_count = sum(1 for r in results if r.no_known_allergies)

    # Category breakdown.
    category_counts: dict[str, int] = {}
    for r in results:
        for a in r.allergies:
            cat = str(a.category)
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "results": [r.to_dict() for r in results],
        "document_count": len(results),
        "total_allergies": total_allergies,
        "nkda_count": nkda_count,
        "category_breakdown": category_counts,
    }


@router.get("/allergies/dictionary/stats")
async def allergen_dictionary_stats() -> dict[str, Any]:
    """Return allergen dictionary coverage statistics."""
    counts = ClinicalAllergyExtractor.get_allergen_count()
    total = sum(counts.values())
    return {
        "total_allergens": total,
        "by_category": counts,
        "reaction_count": ClinicalAllergyExtractor.get_reaction_count(),
    }


@router.get("/allergies/categories")
async def list_allergy_categories() -> dict[str, Any]:
    """List supported allergy categories with descriptions."""
    categories = [
        {
            "category": "drug",
            "description": "Drug and medication allergies (antibiotics, NSAIDs, opioids, etc.)",
            "count": _extractor.get_allergen_count().get("drug", 0),
        },
        {
            "category": "food",
            "description": "Food allergies (peanuts, shellfish, dairy, etc.)",
            "count": _extractor.get_allergen_count().get("food", 0),
        },
        {
            "category": "environmental",
            "description": "Environmental allergies (pollen, dust, latex, etc.)",
            "count": _extractor.get_allergen_count().get("environmental", 0),
        },
    ]
    return {
        "categories": categories,
        "count": len(categories),
    }
