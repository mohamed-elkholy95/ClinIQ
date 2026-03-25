"""De-identification API route for PHI redaction in clinical text.

Exposes a ``POST /deidentify`` endpoint that accepts raw clinical text
and returns the de-identified version along with detected PHI entity
metadata.  Supports three replacement strategies (redact, mask,
surrogate) and per-request PHI type filtering.

This endpoint is designed for pre-processing clinical text before
feeding it into downstream NLP pipelines (NER, ICD prediction, etc.)
where PHI is not needed and could pose a compliance risk.
"""

import logging
from enum import StrEnum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.deidentification import (
    DeidentificationConfig,
    Deidentifier,
    PhiType,
    ReplacementStrategy,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/deidentify", tags=["deidentify"])


class DeidentifyStrategy(StrEnum):
    """Replacement strategy for the API request."""

    REDACT = "redact"
    MASK = "mask"
    SURROGATE = "surrogate"


class DeidentifyRequest(BaseModel):
    """Request schema for de-identification.

    Parameters
    ----------
    text : str
        Clinical text to de-identify.  Maximum 100,000 characters.
    strategy : DeidentifyStrategy
        Replacement strategy (default: redact).
    phi_types : list[str] | None
        Optional subset of PHI types to detect.  If omitted, all
        18 HIPAA Safe Harbor categories are scanned.
    confidence_threshold : float
        Minimum detection confidence to include (default: 0.5).
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Clinical text to de-identify",
    )
    strategy: DeidentifyStrategy = Field(
        default=DeidentifyStrategy.REDACT,
        description="PHI replacement strategy",
    )
    phi_types: list[str] | None = Field(
        default=None,
        description="Optional PHI type filter (e.g. ['NAME', 'DATE', 'SSN'])",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence",
    )


class PhiEntityResponse(BaseModel):
    """A single detected PHI entity in the response."""

    text: str
    phi_type: str
    start_char: int
    end_char: int
    confidence: float
    pattern_name: str


class DeidentifyResponse(BaseModel):
    """Response schema for de-identification.

    Parameters
    ----------
    text : str
        De-identified text with PHI replaced.
    entities : list[PhiEntityResponse]
        List of detected PHI entities with positions and types.
    entity_count : int
        Total number of PHI spans detected.
    phi_types_found : list[str]
        Unique PHI categories found in the text.
    strategy : str
        The replacement strategy that was applied.
    """

    text: str
    entities: list[PhiEntityResponse]
    entity_count: int
    phi_types_found: list[str]
    strategy: str


class BatchDeidentifyRequest(BaseModel):
    """Batch de-identification request.

    Parameters
    ----------
    texts : list[str]
        List of clinical texts to de-identify (max 50).
    strategy : DeidentifyStrategy
        Replacement strategy applied to all texts.
    confidence_threshold : float
        Minimum detection confidence.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of clinical texts to de-identify",
    )
    strategy: DeidentifyStrategy = Field(
        default=DeidentifyStrategy.REDACT,
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


class BatchDeidentifyResponse(BaseModel):
    """Batch de-identification response."""

    results: list[DeidentifyResponse]
    total_entities: int
    total_documents: int


@router.post(
    "",
    response_model=DeidentifyResponse,
    summary="De-identify clinical text",
    description=(
        "Detect and replace Protected Health Information (PHI) in clinical "
        "text following HIPAA Safe Harbor guidelines.  Returns the sanitised "
        "text along with metadata about each detected PHI span."
    ),
)
async def deidentify_text(request: DeidentifyRequest) -> DeidentifyResponse:
    """De-identify a single clinical text document.

    Parameters
    ----------
    request : DeidentifyRequest
        The clinical text and configuration options.

    Returns
    -------
    DeidentifyResponse
        De-identified text with PHI entity metadata.

    Raises
    ------
    HTTPException
        422 if PHI type names are invalid; 500 on unexpected errors.
    """
    try:
        # Parse enabled PHI types
        enabled_types: set[PhiType] | None = None
        if request.phi_types:
            try:
                enabled_types = {PhiType(t.upper()) for t in request.phi_types}
            except ValueError as exc:
                valid_types = [t.value for t in PhiType]
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid PHI type. Valid types: {valid_types}",
                ) from exc

        # Map API strategy to internal enum
        strategy = ReplacementStrategy(request.strategy.value)

        config = DeidentificationConfig(
            strategy=strategy,
            enabled_types=enabled_types,
            confidence_threshold=request.confidence_threshold,
        )

        deid = Deidentifier(config)
        result = deid.deidentify(request.text)

        return DeidentifyResponse(
            text=result["text"],
            entities=[PhiEntityResponse(**e) for e in result["entities"]],
            entity_count=result["entity_count"],
            phi_types_found=result["phi_types_found"],
            strategy=result["strategy"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("De-identification failed")
        raise HTTPException(
            status_code=500,
            detail=f"De-identification failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=BatchDeidentifyResponse,
    summary="Batch de-identify clinical texts",
    description=(
        "De-identify multiple clinical texts in a single request. "
        "Maximum 50 documents per batch."
    ),
)
async def deidentify_batch(
    request: BatchDeidentifyRequest,
) -> BatchDeidentifyResponse:
    """De-identify a batch of clinical text documents.

    Parameters
    ----------
    request : BatchDeidentifyRequest
        List of texts and shared configuration.

    Returns
    -------
    BatchDeidentifyResponse
        Results for each input document.

    Raises
    ------
    HTTPException
        500 on unexpected errors.
    """
    try:
        strategy = ReplacementStrategy(request.strategy.value)
        config = DeidentificationConfig(
            strategy=strategy,
            confidence_threshold=request.confidence_threshold,
        )
        deid = Deidentifier(config)

        results: list[DeidentifyResponse] = []
        total_entities = 0

        for text in request.texts:
            result = deid.deidentify(text)
            total_entities += result["entity_count"]
            results.append(
                DeidentifyResponse(
                    text=result["text"],
                    entities=[
                        PhiEntityResponse(**e) for e in result["entities"]
                    ],
                    entity_count=result["entity_count"],
                    phi_types_found=result["phi_types_found"],
                    strategy=result["strategy"],
                )
            )

        return BatchDeidentifyResponse(
            results=results,
            total_entities=total_entities,
            total_documents=len(request.texts),
        )

    except Exception as exc:
        logger.exception("Batch de-identification failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch de-identification failed: {exc}",
        ) from exc
