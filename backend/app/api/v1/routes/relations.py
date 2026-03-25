"""Clinical relation extraction API endpoints.

Exposes entity-relation extraction through REST endpoints.  Accepts
pre-extracted entities or raw text (which triggers NER first), and returns
structured semantic relations.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import ClinIQError
from app.ml.ner.model import Entity
from app.ml.relations.extractor import (
    RelationType,
    RuleBasedRelationExtractor,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/relations", tags=["relations"])

# Module-level extractor instance (stateless, safe to share)
_extractor = RuleBasedRelationExtractor()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EntityInput(BaseModel):
    """Minimal entity representation for relation extraction requests."""

    text: str = Field(..., description="Entity surface text")
    entity_type: str = Field(..., description="Entity type (DISEASE, MEDICATION, etc.)")
    start_char: int = Field(..., ge=0, description="Start character offset")
    end_char: int = Field(..., gt=0, description="End character offset")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_negated: bool = Field(default=False)

    def to_entity(self) -> Entity:
        """Convert to internal Entity dataclass."""
        return Entity(
            text=self.text,
            entity_type=self.entity_type,
            start_char=self.start_char,
            end_char=self.end_char,
            confidence=self.confidence,
            is_negated=self.is_negated,
        )


class RelationRequest(BaseModel):
    """Request body for relation extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Clinical text to extract relations from",
    )
    entities: list[EntityInput] = Field(
        ...,
        min_length=2,
        description="Pre-extracted entities with offsets",
    )
    max_distance: int = Field(
        default=150,
        ge=10,
        le=500,
        description="Maximum character distance between entity pairs",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    relation_types: list[str] | None = Field(
        default=None,
        description="Filter to specific relation types (e.g. ['treats', 'causes'])",
    )


class RelationResponse(BaseModel):
    """Response body for relation extraction."""

    relations: list[dict[str, Any]]
    entity_count: int
    pair_count: int
    relation_count: int
    processing_time_ms: float
    model_name: str
    model_version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=RelationResponse,
    summary="Extract relations between entities",
    description=(
        "Given clinical text and pre-extracted entities, identifies semantic "
        "relations (treats, causes, diagnoses, etc.) between entity pairs."
    ),
)
async def extract_relations(request: RelationRequest) -> RelationResponse:
    """Extract semantic relations between medical entities.

    Parameters
    ----------
    request:
        Clinical text and pre-extracted entities.

    Returns
    -------
    RelationResponse
        Extracted relations sorted by descending confidence.
    """
    try:
        entities = [e.to_entity() for e in request.entities]

        result = _extractor.extract(
            text=request.text,
            entities=entities,
            max_distance=request.max_distance,
            min_confidence=request.min_confidence,
        )

        # Apply relation type filter if specified
        relations = result.relations
        if request.relation_types:
            valid_types = set(request.relation_types)
            relations = [r for r in relations if r.relation_type.value in valid_types]

        return RelationResponse(
            relations=[r.to_dict() for r in relations],
            entity_count=result.entity_count,
            pair_count=result.pair_count,
            relation_count=len(relations),
            processing_time_ms=result.processing_time_ms,
            model_name=result.model_name,
            model_version=result.model_version,
        )
    except ClinIQError:
        raise
    except Exception as exc:
        logger.exception("Relation extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Relation extraction error: {exc}",
        ) from exc


@router.get(
    "/types",
    summary="List available relation types",
    description="Returns all supported relation types with descriptions.",
)
async def list_relation_types() -> dict[str, Any]:
    """Return the catalogue of supported relation types.

    Returns
    -------
    dict
        Mapping of relation type values to their metadata.
    """
    return {
        "relation_types": {
            rt.value: {
                "name": rt.name,
                "description": rt.value.replace("_", " ").title(),
            }
            for rt in RelationType
        },
        "count": len(RelationType),
    }
