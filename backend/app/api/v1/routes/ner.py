"""NER (Named Entity Recognition) endpoint.

Exposes a standalone NER inference endpoint that extracts clinical entities
from free-text without running the full analysis pipeline.  Entities are
produced by the real rule-based NER model loaded via the model registry.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.ner import EntityResponse, NERRequest, NERResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session
from app.services.model_registry import get_ner_model

router = APIRouter(tags=["ner"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helper — delegates to the real ML model
# ---------------------------------------------------------------------------


def _run_ner_inference(request: NERRequest) -> list[EntityResponse]:
    """Run NER inference using the model registry.

    Parameters
    ----------
    request:
        Validated NER request with text and optional filters.

    Returns
    -------
    list[EntityResponse]
        Extracted entities, filtered per request parameters.
    """
    model = get_ner_model()
    raw_entities = model.extract_entities(request.text)

    # Convert ML-layer Entity dataclasses to API schema objects.
    entities: list[EntityResponse] = [
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
        for e in raw_entities
    ]

    # Apply entity type filter if requested.
    if request.entity_types is not None:
        entities = [e for e in entities if e.entity_type in request.entity_types]

    # Apply negation filter.
    if not request.include_negated:
        entities = [e for e in entities if not e.is_negated]

    # Apply uncertainty filter.
    if not request.include_uncertain:
        entities = [e for e in entities if not e.is_uncertain]

    # Apply confidence threshold.
    entities = [e for e in entities if e.confidence >= request.min_confidence]

    return entities


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/ner",
    response_model=NERResponse,
    status_code=status.HTTP_200_OK,
    summary="Named entity recognition",
    description=(
        "Extract clinical named entities from the supplied text. "
        "Supported entity types include DISEASE, SYMPTOM, MEDICATION, DOSAGE, "
        "PROCEDURE, ANATOMY, LAB_VALUE, TEST, TREATMENT, DEVICE, BODY_PART, "
        "DURATION, FREQUENCY, and TEMPORAL. "
        "Entities can optionally be filtered by type, confidence threshold, "
        "and negation / uncertainty status."
    ),
    responses={
        200: {"description": "Entities extracted successfully"},
        422: {"description": "Input validation error"},
        500: {"description": "NER inference error"},
    },
)
async def extract_entities(
    payload: NERRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> NERResponse:
    """Run NER inference and return extracted clinical entities."""
    t0 = time.monotonic()

    try:
        entities = _run_ner_inference(payload)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="NER inference failed unexpectedly. Please try again.",
        ) from exc

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Sort entities by start character position.
    entities.sort(key=lambda e: e.start_char)

    return NERResponse(
        text_length=len(payload.text),
        entity_count=len(entities),
        entities=entities,
        model_name=payload.model,
        model_version="1.0.0",
        processing_time_ms=elapsed_ms,
    )
