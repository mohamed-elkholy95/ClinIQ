"""REST API routes for clinical abbreviation expansion.

Provides endpoints for detecting and expanding medical abbreviations
in clinical free text, with context-aware disambiguation for ambiguous
abbreviations.
"""

import logging
from contextlib import suppress

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ml.abbreviations import (
    AbbreviationConfig,
    AbbreviationExpander,
    ClinicalDomain,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/abbreviations", tags=["abbreviations"])


# ─────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────


class AbbreviationRequest(BaseModel):
    """Request schema for abbreviation expansion.

    Parameters
    ----------
    text : str
        Clinical text to analyze (1–50,000 characters).
    min_confidence : float
        Minimum confidence threshold (0.0–1.0, default 0.60).
    expand_in_place : bool
        Whether to produce expanded text output (default True).
    domains : list[str] | None
        Filter to specific clinical domains (None = all).
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Clinical text to analyze",
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    expand_in_place: bool = Field(
        default=True,
        description="Whether to produce expanded text output",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Filter to specific clinical domains",
    )


class BatchAbbreviationRequest(BaseModel):
    """Request schema for batch abbreviation expansion.

    Parameters
    ----------
    texts : list[str]
        List of clinical texts (1–50 documents).
    min_confidence : float
        Minimum confidence threshold.
    expand_in_place : bool
        Whether to produce expanded text output.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of clinical texts to analyze",
    )
    min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
    )
    expand_in_place: bool = Field(default=True)


class AbbreviationMatchResponse(BaseModel):
    """Response schema for a single abbreviation match."""

    abbreviation: str
    expansion: str
    start: int
    end: int
    confidence: float
    domain: str
    is_ambiguous: bool
    resolution: str
    alternative_expansions: list[str]


class AbbreviationResponse(BaseModel):
    """Response schema for abbreviation expansion."""

    original_text: str
    expanded_text: str
    matches: list[AbbreviationMatchResponse]
    total_found: int
    ambiguous_count: int
    processing_time_ms: float


class BatchAbbreviationResponse(BaseModel):
    """Response schema for batch abbreviation expansion."""

    results: list[AbbreviationResponse]
    total_documents: int
    total_abbreviations: int
    total_ambiguous: int


class DomainInfo(BaseModel):
    """Information about a clinical domain."""

    name: str
    unambiguous_count: int
    ambiguous_sense_count: int


class DictionaryStatsResponse(BaseModel):
    """Response schema for dictionary statistics."""

    total_unambiguous: int
    total_ambiguous: int
    total_senses: int
    total_entries: int
    domains: list[DomainInfo]


class LookupResponse(BaseModel):
    """Response schema for abbreviation lookup."""

    abbreviation: str
    is_ambiguous: bool
    expansion: str | None = None
    domain: str | None = None
    confidence: float | None = None
    senses: list[dict] | None = None


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────


@router.post("", response_model=AbbreviationResponse)
async def expand_abbreviations(request: AbbreviationRequest) -> dict:
    """Detect and expand clinical abbreviations in text.

    Analyzes clinical free text for medical abbreviations, expands them
    to full forms, and handles ambiguous abbreviations via context-aware
    disambiguation.

    Parameters
    ----------
    request : AbbreviationRequest
        Input text and configuration options.

    Returns
    -------
    dict
        Expansion results with matches and expanded text.
    """
    logger.info(
        "Abbreviation expansion request",
        extra={"text_length": len(request.text)},
    )

    # Parse domain filter
    domains = None
    if request.domains:
        with suppress(ValueError):
            domains = [ClinicalDomain(d) for d in request.domains]

    config = AbbreviationConfig(
        min_confidence=request.min_confidence,
        expand_in_place=request.expand_in_place,
        domains=domains,
    )
    expander = AbbreviationExpander(config)
    result = expander.expand(request.text)
    return result.to_dict()


@router.post("/batch", response_model=BatchAbbreviationResponse)
async def expand_abbreviations_batch(
    request: BatchAbbreviationRequest,
) -> dict:
    """Expand abbreviations in multiple documents.

    Parameters
    ----------
    request : BatchAbbreviationRequest
        List of clinical texts to analyze.

    Returns
    -------
    dict
        Batch results with per-document matches and aggregate statistics.
    """
    logger.info(
        "Batch abbreviation expansion",
        extra={"document_count": len(request.texts)},
    )

    config = AbbreviationConfig(
        min_confidence=request.min_confidence,
        expand_in_place=request.expand_in_place,
    )
    expander = AbbreviationExpander(config)
    results = expander.expand_batch(request.texts)

    return {
        "results": [r.to_dict() for r in results],
        "total_documents": len(results),
        "total_abbreviations": sum(r.total_found for r in results),
        "total_ambiguous": sum(r.ambiguous_count for r in results),
    }


@router.get("/lookup/{abbreviation}", response_model=LookupResponse)
async def lookup_abbreviation(
    abbreviation: str,
) -> dict:
    """Look up a specific abbreviation in the dictionary.

    Parameters
    ----------
    abbreviation : str
        Abbreviation to look up (case-insensitive).

    Returns
    -------
    dict
        Abbreviation details or 404 equivalent (null fields).
    """
    expander = AbbreviationExpander()
    result = expander.lookup(abbreviation)

    if result is None:
        return {
            "abbreviation": abbreviation,
            "is_ambiguous": False,
            "expansion": None,
            "domain": None,
            "confidence": None,
            "senses": None,
        }

    return result


@router.get("/dictionary/stats", response_model=DictionaryStatsResponse)
async def get_dictionary_stats() -> dict:
    """Get abbreviation dictionary coverage statistics.

    Returns
    -------
    dict
        Total counts by domain and ambiguity status.
    """
    expander = AbbreviationExpander()
    stats = expander.get_dictionary_stats()

    return {
        "total_unambiguous": stats["total_unambiguous"],
        "total_ambiguous": stats["total_ambiguous"],
        "total_senses": stats["total_senses"],
        "total_entries": stats["total_entries"],
        "domains": [
            {
                "name": name,
                "unambiguous_count": info["unambiguous"],
                "ambiguous_sense_count": info["ambiguous_senses"],
            }
            for name, info in stats["domains"].items()
        ],
    }


@router.get("/domains")
async def list_domains() -> dict:
    """List all supported clinical domains.

    Returns
    -------
    dict
        List of clinical domain names with descriptions.
    """
    domain_descriptions = {
        ClinicalDomain.CARDIOLOGY: "Heart and vascular system",
        ClinicalDomain.PULMONOLOGY: "Respiratory system",
        ClinicalDomain.ENDOCRINE: "Endocrine and metabolic disorders",
        ClinicalDomain.NEUROLOGY: "Nervous system",
        ClinicalDomain.GASTROENTEROLOGY: "Digestive system",
        ClinicalDomain.RENAL: "Kidneys and urinary tract",
        ClinicalDomain.INFECTIOUS: "Infectious diseases",
        ClinicalDomain.MUSCULOSKELETAL: "Bones, joints, and muscles",
        ClinicalDomain.HEMATOLOGY: "Blood and lab values",
        ClinicalDomain.GENERAL: "General clinical terminology",
        ClinicalDomain.DENTAL: "Dental and oral health",
        ClinicalDomain.PHARMACY: "Medications, dosing, and routes",
    }

    return {
        "domains": [
            {"name": str(d), "description": desc}
            for d, desc in domain_descriptions.items()
        ],
        "total": len(ClinicalDomain),
    }

