"""NER endpoint routes."""

import logging
from collections import Counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps import PipelineDep
from app.api.v1.schemas import NERRequest, NERResponse, EntityResponse
from app.ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ner", tags=["named-entity-recognition"])


@router.post(
    "",
    response_model=NERResponse,
    summary="Extract medical entities",
    description="Extract medical named entities from clinical text including diseases, medications, procedures, and more.",
)
async def extract_entities(
    request: NERRequest,
    pipeline: PipelineDep,
) -> NERResponse:
    """Extract medical entities from text."""
    import time

    start_time = time.time()

    try:
        entities = pipeline.extract_entities(request.text)

        # Build response
        entity_responses = [
            EntityResponse(
                text=e.text,
                entity_type=e.entity_type,
                start_char=e.start_char,
                end_char=e.end_char,
                confidence=e.confidence,
                normalized_text=e.normalized_text,
                umls_cui=e.umls_cui,
                is_negated=e.is_negated,
                is_uncertain=e.is_uncertain,
            )
            for e in entities
        ]

        # Count by type
        type_counts = Counter(e.entity_type for e in entities)

        processing_time = (time.time() - start_time) * 1000

        return NERResponse(
            entities=entity_responses,
            entity_count=len(entity_responses),
            entity_type_counts=dict(type_counts),
            processing_time_ms=processing_time,
            model_version=pipeline._ner_model.version if pipeline._ner_model else "unknown",
        )

    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}",
        )


@router.post(
    "/highlight",
    summary="Get entities with highlighted text",
    description="Extract entities and return text with entity markers.",
)
async def extract_with_highlight(
    request: NERRequest,
    pipeline: PipelineDep,
) -> dict:
    """Extract entities and return highlighted text."""
    entities = pipeline.extract_entities(request.text)

    # Sort by position (reverse) for insertion without offset issues
    sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)

    # Build highlighted text
    highlighted = request.text
    for entity in sorted_entities:
        # Insert closing tag
        tag = f"[/{entity.entity_type}]"
        highlighted = highlighted[: entity.end_char] + tag + highlighted[entity.end_char :]

        # Insert opening tag
        tag = f"[{entity.entity_type}]"
        highlighted = highlighted[: entity.start_char] + tag + highlighted[entity.start_char :]

    return {
        "original_text": request.text,
        "highlighted_text": highlighted,
        "entities": [e.to_dict() for e in entities],
    }
