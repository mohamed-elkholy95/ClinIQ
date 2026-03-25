"""Clinical concept normalization API endpoints.

Provides entity linking from extracted text spans to standardised medical
ontology codes (UMLS CUI, SNOMED-CT, RxNorm, ICD-10-CM, LOINC).

Endpoints
---------
- ``POST /normalize``         — Normalize a single entity mention
- ``POST /normalize/batch``   — Normalize multiple entities in one request
- ``GET  /normalize/lookup/{cui}`` — Reverse-lookup a concept by CUI
- ``GET  /normalize/dictionary/stats`` — Dictionary coverage statistics
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.ml.normalization.normalizer import (
    ClinicalConceptNormalizer,
    NormalizerConfig,
    get_dictionary_stats,
)

router = APIRouter(tags=["normalization"])

# Module-level normalizer — thread-safe for concurrent requests
_normalizer = ClinicalConceptNormalizer()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class NormalizeSingleRequest(BaseModel):
    """Request body for single-entity normalization."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="Entity surface form to normalize (e.g. 'HTN', 'heart attack').",
        ),
    ]
    entity_type: str | None = Field(
        default=None,
        description=(
            "Optional NER entity type for type-aware filtering "
            "(e.g. 'DISEASE', 'MEDICATION', 'PROCEDURE', 'LAB_VALUE')."
        ),
    )
    min_similarity: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy similarity threshold (0.0–1.0).",
    )
    enable_fuzzy: bool = Field(
        default=True,
        description="Whether to attempt fuzzy matching when exact/alias lookups fail.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "HTN",
                "entity_type": "DISEASE",
                "min_similarity": 0.80,
                "enable_fuzzy": True,
            }
        }
    }


class NormalizeBatchRequest(BaseModel):
    """Request body for batch entity normalization."""

    entities: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=500,
        description=(
            "List of entity objects. Each must have a 'text' key; "
            "optional 'entity_type' key for type-aware filtering."
        ),
    )
    min_similarity: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum fuzzy similarity threshold.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "entities": [
                    {"text": "HTN", "entity_type": "DISEASE"},
                    {"text": "metformin", "entity_type": "MEDICATION"},
                    {"text": "chest pain"},
                ],
                "min_similarity": 0.80,
            }
        }
    }


class CodeSet(BaseModel):
    """Ontology code set for a normalised concept."""

    umls_cui: str | None = None
    snomed_ct: str | None = None
    rxnorm: str | None = None
    icd10_cm: str | None = None
    loinc: str | None = None


class AlternativeMatch(BaseModel):
    """An alternative candidate match."""

    cui: str
    preferred_term: str
    confidence: float
    codes: CodeSet


class NormalizeSingleResponse(BaseModel):
    """Response for single-entity normalization."""

    input_text: str
    matched: bool
    cui: str | None = None
    preferred_term: str | None = None
    confidence: float
    match_type: str
    codes: CodeSet
    semantic_type: str
    alternatives: list[AlternativeMatch]


class BatchSummary(BaseModel):
    """Summary statistics for a batch normalization."""

    total: int
    matched: int
    unmatched: int
    match_rate: float
    processing_time_ms: float


class NormalizeBatchResponse(BaseModel):
    """Response for batch entity normalization."""

    results: list[NormalizeSingleResponse]
    summary: BatchSummary


class ConceptLookupResponse(BaseModel):
    """Response for CUI reverse-lookup."""

    cui: str
    preferred_term: str
    aliases: list[str]
    codes: CodeSet
    semantic_type: str
    type_group: str


class DictionaryStatsResponse(BaseModel):
    """Response for dictionary coverage statistics."""

    total_concepts: int
    total_aliases: int
    by_type_group: dict[str, int]
    ontology_coverage: dict[str, int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/normalize",
    response_model=NormalizeSingleResponse,
    summary="Normalize a single entity mention",
    description=(
        "Maps an entity surface form to standardised medical ontology codes. "
        "Tries exact match, then alias match, then fuzzy matching."
    ),
)
async def normalize_entity(request: NormalizeSingleRequest) -> dict:
    """Normalize a single entity mention to standardised codes.

    Parameters
    ----------
    request : NormalizeSingleRequest
        Entity text and optional type constraint.

    Returns
    -------
    dict
        Normalization result with matched codes and confidence.
    """
    config = NormalizerConfig(
        min_similarity=request.min_similarity,
        enable_fuzzy=request.enable_fuzzy,
    )
    normalizer = ClinicalConceptNormalizer(config)
    result = normalizer.normalize(request.text, request.entity_type)
    return result.to_dict()


@router.post(
    "/normalize/batch",
    response_model=NormalizeBatchResponse,
    summary="Normalize a batch of entity mentions",
    description=(
        "Maps multiple entity surface forms to standardised ontology codes "
        "in a single request. Each entity may include an optional entity_type "
        "for type-aware filtering."
    ),
)
async def normalize_batch(request: NormalizeBatchRequest) -> dict:
    """Normalize a batch of entity mentions.

    Parameters
    ----------
    request : NormalizeBatchRequest
        List of entities with text and optional type.

    Returns
    -------
    dict
        Batch results with per-entity normalization and summary statistics.
    """
    # Validate each entity has 'text'
    for i, entity in enumerate(request.entities):
        if "text" not in entity or not entity["text"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Entity at index {i} missing required 'text' field.",
            )

    config = NormalizerConfig(min_similarity=request.min_similarity)
    normalizer = ClinicalConceptNormalizer(config)
    batch_result = normalizer.normalize_batch(request.entities)
    return batch_result.to_dict()


@router.get(
    "/normalize/lookup/{cui}",
    response_model=ConceptLookupResponse,
    summary="Look up a concept by UMLS CUI",
    description="Reverse-lookup a concept entry by its UMLS Concept Unique Identifier.",
)
async def lookup_concept(cui: str) -> dict:
    """Look up a concept by UMLS CUI.

    Parameters
    ----------
    cui : str
        UMLS Concept Unique Identifier (e.g. 'C0020538').

    Returns
    -------
    dict
        Concept details including preferred term, aliases, and codes.

    Raises
    ------
    HTTPException
        404 if the CUI is not found in the dictionary.
    """
    concept = _normalizer.lookup_cui(cui)
    if concept is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CUI '{cui}' not found in concept dictionary.",
        )
    return {
        "cui": concept.cui,
        "preferred_term": concept.preferred_term,
        "aliases": list(concept.aliases),
        "codes": {
            "umls_cui": concept.cui,
            "snomed_ct": concept.snomed_code,
            "rxnorm": concept.rxnorm_code,
            "icd10_cm": concept.icd10_code,
            "loinc": concept.loinc_code,
        },
        "semantic_type": concept.semantic_type,
        "type_group": concept.type_group.value,
    }


@router.get(
    "/normalize/dictionary/stats",
    response_model=DictionaryStatsResponse,
    summary="Get concept dictionary statistics",
    description="Returns coverage statistics for the concept dictionary.",
)
async def dictionary_stats() -> dict:
    """Return concept dictionary coverage statistics.

    Returns
    -------
    dict
        Total concepts, aliases, per-group counts, and ontology coverage.
    """
    return get_dictionary_stats()
