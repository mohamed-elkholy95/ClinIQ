"""NER (Named Entity Recognition) endpoint.

Exposes a standalone NER inference endpoint that extracts clinical entities
from free-text without running the full analysis pipeline.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.ner import EntityResponse, NERRequest, NERResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InferenceError
from app.db.session import get_db_session

router = APIRouter(tags=["ner"])


# ---------------------------------------------------------------------------
# Placeholder inference helper
# ---------------------------------------------------------------------------


def _run_ner_inference(request: NERRequest) -> list[EntityResponse]:
    """Placeholder NER inference; returns mock entities.

    Replace this with a call to the real NER model once the model layer is wired.
    """
    mock_entities: list[EntityResponse] = [
        EntityResponse(
            text="hypertension",
            entity_type="DISEASE",
            start_char=11,
            end_char=23,
            confidence=0.96,
            normalized_text="Hypertension",
            umls_cui="C0020538",
            is_negated=False,
            is_uncertain=False,
        ),
        EntityResponse(
            text="metformin",
            entity_type="MEDICATION",
            start_char=44,
            end_char=53,
            confidence=0.94,
            normalized_text="Metformin",
            umls_cui="C0025598",
            is_negated=False,
            is_uncertain=False,
        ),
        EntityResponse(
            text="chest pain",
            entity_type="SYMPTOM",
            start_char=100,
            end_char=110,
            confidence=0.89,
            normalized_text="chest pain",
            umls_cui="C0008031",
            is_negated=True,
            is_uncertain=False,
        ),
    ]

    # Apply entity type filter if requested.
    if request.entity_types is not None:
        mock_entities = [e for e in mock_entities if e.entity_type in request.entity_types]

    # Apply negation filter.
    if not request.include_negated:
        mock_entities = [e for e in mock_entities if not e.is_negated]

    # Apply uncertainty filter.
    if not request.include_uncertain:
        mock_entities = [e for e in mock_entities if not e.is_uncertain]

    # Apply confidence filter.
    mock_entities = [e for e in mock_entities if e.confidence >= request.min_confidence]

    return mock_entities


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
