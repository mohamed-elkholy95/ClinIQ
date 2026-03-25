"""Social Determinants of Health (SDoH) extraction API endpoints.

Provides REST endpoints for extracting social and behavioural risk
factors from clinical text, with single-document, batch, and domain
catalogue operations.

Endpoints
---------
* ``POST /sdoh`` — Extract SDoH factors from a single clinical note.
* ``POST /sdoh/batch`` — Extract SDoH factors from up to 50 notes.
* ``GET  /sdoh/domains`` — List all 8 SDoH domains with metadata.
* ``GET  /sdoh/domains/{domain_name}`` — Detail for a specific domain.
* ``GET  /sdoh/z-codes`` — Full Z-code catalogue across all domains.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.sdoh.extractor import (
    ClinicalSDoHExtractor,
    SDoHDomain,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Singleton extractor (stateless, thread-safe)
# ---------------------------------------------------------------------------

_extractor = ClinicalSDoHExtractor()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SDoHRequest(BaseModel):
    """Request body for single-document SDoH extraction.

    Attributes
    ----------
    text : str
        Clinical note text to analyse.
    min_confidence : float
        Minimum confidence threshold (0.0–1.0).
    """

    text: str = Field(..., min_length=1, max_length=50000)
    min_confidence: float = Field(0.50, ge=0.0, le=1.0)


class SDoHBatchRequest(BaseModel):
    """Request body for batch SDoH extraction.

    Attributes
    ----------
    documents : list[SDoHRequest]
        Up to 50 documents to process.
    """

    documents: list[SDoHRequest] = Field(..., min_length=1, max_length=50)


class SDoHExtractionResponse(BaseModel):
    """Single extraction item in the response."""

    domain: str
    text: str
    sentiment: str
    confidence: float
    z_codes: list[str]
    trigger: str
    start: int
    end: int
    negated: bool
    section: str


class SDoHResponse(BaseModel):
    """Response body for single-document SDoH extraction."""

    extractions: list[SDoHExtractionResponse]
    domain_summary: dict[str, int]
    adverse_count: int
    protective_count: int
    text_length: int
    processing_time_ms: float


class SDoHBatchResponse(BaseModel):
    """Response body for batch SDoH extraction."""

    results: list[SDoHResponse]
    total_documents: int
    total_extractions: int
    aggregate_adverse: int
    aggregate_protective: int
    processing_time_ms: float


class DomainInfoResponse(BaseModel):
    """Metadata for a single SDoH domain."""

    domain: str
    description: str
    z_codes: list[dict[str, str]]
    trigger_count: int
    adverse_triggers: int
    protective_triggers: int


class ZCodeEntry(BaseModel):
    """A single Z-code entry."""

    code: str
    description: str
    domain: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/sdoh", response_model=SDoHResponse, tags=["sdoh"])
async def extract_sdoh(request: SDoHRequest) -> dict[str, Any]:
    """Extract social determinants of health from clinical text.

    Scans the input text across 8 SDoH domains (housing, employment,
    education, food security, transportation, social support, substance
    use, financial) and returns all detected factors with confidence
    scores and ICD-10-CM Z-codes.

    Parameters
    ----------
    request : SDoHRequest
        Clinical text and optional confidence threshold.

    Returns
    -------
    dict[str, Any]
        SDoH extraction results with domain summary.
    """
    extractor = ClinicalSDoHExtractor(min_confidence=request.min_confidence)
    result = extractor.extract(request.text)
    return result.to_dict()


@router.post("/sdoh/batch", response_model=SDoHBatchResponse, tags=["sdoh"])
async def extract_sdoh_batch(request: SDoHBatchRequest) -> dict[str, Any]:
    """Extract SDoH factors from multiple clinical documents.

    Processes up to 50 documents and returns per-document results with
    aggregate statistics.

    Parameters
    ----------
    request : SDoHBatchRequest
        List of documents to process.

    Returns
    -------
    dict[str, Any]
        Per-document results and aggregate counts.
    """
    start_time = time.perf_counter()
    results: list[dict[str, Any]] = []
    total_extractions = 0
    aggregate_adverse = 0
    aggregate_protective = 0

    for doc in request.documents:
        extractor = ClinicalSDoHExtractor(min_confidence=doc.min_confidence)
        result = extractor.extract(doc.text)
        result_dict = result.to_dict()
        results.append(result_dict)
        total_extractions += len(result.extractions)
        aggregate_adverse += result.adverse_count
        aggregate_protective += result.protective_count

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "results": results,
        "total_documents": len(request.documents),
        "total_extractions": total_extractions,
        "aggregate_adverse": aggregate_adverse,
        "aggregate_protective": aggregate_protective,
        "processing_time_ms": round(elapsed_ms, 2),
    }


@router.get(
    "/sdoh/domains",
    response_model=list[DomainInfoResponse],
    tags=["sdoh"],
)
async def list_sdoh_domains() -> list[dict[str, Any]]:
    """List all 8 SDoH domains with metadata.

    Returns trigger counts, Z-code mappings, and descriptions for each
    domain.

    Returns
    -------
    list[dict[str, Any]]
        Domain metadata list.
    """
    return _extractor.get_all_domains()


@router.get(
    "/sdoh/domains/{domain_name}",
    response_model=DomainInfoResponse,
    tags=["sdoh"],
)
async def get_sdoh_domain(domain_name: str) -> dict[str, Any]:
    """Get detailed metadata for a specific SDoH domain.

    Parameters
    ----------
    domain_name : str
        Domain identifier (e.g., "housing", "substance_use").

    Returns
    -------
    dict[str, Any]
        Domain metadata.

    Raises
    ------
    HTTPException
        404 if domain name is not recognised.
    """
    try:
        domain = SDoHDomain(domain_name.lower())
    except ValueError:
        valid = [d.value for d in SDoHDomain]
        raise HTTPException(
            status_code=404,
            detail=f"Unknown domain '{domain_name}'. Valid domains: {valid}",
        )

    return _extractor.get_domain_info(domain)


@router.get("/sdoh/z-codes", response_model=list[ZCodeEntry], tags=["sdoh"])
async def list_z_codes() -> list[dict[str, str]]:
    """List all ICD-10-CM Z-codes across all SDoH domains.

    Returns a flat list of Z-code entries with their domain association.

    Returns
    -------
    list[dict[str, str]]
        Z-code catalogue.
    """
    from app.ml.sdoh.extractor import DOMAIN_Z_CODES

    entries: list[dict[str, str]] = []
    for domain, codes in DOMAIN_Z_CODES.items():
        for code_entry in codes:
            entries.append(
                {
                    "code": code_entry["code"],
                    "description": code_entry["description"],
                    "domain": domain.value,
                }
            )
    return entries
