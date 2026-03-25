"""Medication extraction API endpoints.

Exposes structured medication extraction from clinical notes through REST
endpoints, parsing drug names, dosages, routes, frequencies, durations,
and indications into normalized, machine-readable components.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.medications.extractor import (
    DRUG_DICTIONARY,
    ClinicalMedicationExtractor,
    MedicationStatus,
    RouteOfAdministration,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/medications", tags=["medications"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class MedicationRequest(BaseModel):
    """Request body for medication extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Clinical text to extract medications from",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for returned medications",
    )
    include_generics: bool = Field(
        default=True,
        description="Include generic name normalization in results",
    )


class MedicationBatchRequest(BaseModel):
    """Request body for batch medication extraction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of clinical texts (max 50)",
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class DosageResponse(BaseModel):
    """Dosage information."""

    value: float
    unit: str
    value_high: float | None = None
    raw_text: str = ""


class MedicationResponse(BaseModel):
    """A single extracted medication."""

    drug_name: str
    generic_name: str | None = None
    dosage: DosageResponse | None = None
    route: str
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    prn: bool = False
    status: str
    start_char: int
    end_char: int
    confidence: float
    raw_text: str


class MedicationExtractionResponse(BaseModel):
    """Response body for medication extraction."""

    medications: list[MedicationResponse]
    medication_count: int
    unique_drugs: int
    processing_time_ms: float
    extractor_version: str


class MedicationBatchResponse(BaseModel):
    """Response body for batch medication extraction."""

    results: list[MedicationExtractionResponse]
    total_medications: int
    processing_time_ms: float


class DrugLookupResponse(BaseModel):
    """Response for drug name lookup."""

    query: str
    found: bool
    generic_name: str | None = None
    brand_names: list[str] = []


class DrugDictionaryStatsResponse(BaseModel):
    """Statistics about the drug dictionary."""

    total_entries: int
    unique_generics: int
    routes: list[str]
    statuses: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=MedicationExtractionResponse,
    summary="Extract medications from clinical text",
    description=(
        "Parse medication mentions into structured components: drug name, "
        "dosage, route, frequency, duration, indication, PRN status, and "
        "clinical status. Supports both free-text narratives and structured "
        "medication lists."
    ),
)
async def extract_medications(request: MedicationRequest) -> dict[str, Any]:
    """Extract structured medication information from clinical text.

    Parameters
    ----------
    request : MedicationRequest
        Clinical text and extraction parameters.

    Returns
    -------
    dict[str, Any]
        Structured medication extraction results.

    Raises
    ------
    HTTPException
        500 if extraction fails unexpectedly.
    """
    try:
        extractor = ClinicalMedicationExtractor(min_confidence=request.min_confidence)
        result = extractor.extract(request.text)

        medications = []
        for med in result.medications:
            med_dict = med.to_dict()
            if not request.include_generics:
                med_dict["generic_name"] = None
            medications.append(med_dict)

        return {
            "medications": medications,
            "medication_count": result.medication_count,
            "unique_drugs": result.unique_drugs,
            "processing_time_ms": result.processing_time_ms,
            "extractor_version": result.extractor_version,
        }
    except Exception as exc:
        logger.exception("Medication extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Medication extraction failed: {exc}",
        ) from exc


@router.post(
    "/batch",
    response_model=MedicationBatchResponse,
    summary="Batch medication extraction",
    description="Extract medications from up to 50 clinical texts in one request.",
)
async def extract_medications_batch(
    request: MedicationBatchRequest,
) -> dict[str, Any]:
    """Extract medications from multiple clinical texts.

    Parameters
    ----------
    request : MedicationBatchRequest
        List of clinical texts.

    Returns
    -------
    dict[str, Any]
        Batch extraction results with per-document and aggregate counts.

    Raises
    ------
    HTTPException
        500 if extraction fails.
    """
    try:
        extractor = ClinicalMedicationExtractor(min_confidence=request.min_confidence)
        all_results = extractor.extract_batch(request.texts)

        import time

        start = time.perf_counter()

        response_results = []
        total_meds = 0
        for result in all_results:
            response_results.append({
                "medications": [m.to_dict() for m in result.medications],
                "medication_count": result.medication_count,
                "unique_drugs": result.unique_drugs,
                "processing_time_ms": result.processing_time_ms,
                "extractor_version": result.extractor_version,
            })
            total_meds += result.medication_count

        elapsed = (time.perf_counter() - start) * 1000
        # Use sum of individual times + overhead
        total_time = sum(r.processing_time_ms for r in all_results) + elapsed

        return {
            "results": response_results,
            "total_medications": total_meds,
            "processing_time_ms": round(total_time, 2),
        }
    except Exception as exc:
        logger.exception("Batch medication extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch extraction failed: {exc}",
        ) from exc


@router.get(
    "/lookup/{drug_name}",
    response_model=DrugLookupResponse,
    summary="Look up a drug name",
    description="Check if a drug name exists in the dictionary and get its generic equivalent.",
)
async def lookup_drug(drug_name: str) -> dict[str, Any]:
    """Look up a drug name in the medication dictionary.

    Parameters
    ----------
    drug_name : str
        Drug name to look up (brand or generic).

    Returns
    -------
    dict[str, Any]
        Lookup result with generic name and brand variants.
    """
    drug_lower = drug_name.lower().strip()
    generic = DRUG_DICTIONARY.get(drug_lower)

    if generic is None:
        return {
            "query": drug_name,
            "found": False,
            "generic_name": None,
            "brand_names": [],
        }

    # Find all brand names that map to the same generic
    brand_names = sorted(
        name
        for name, gen in DRUG_DICTIONARY.items()
        if gen == generic and name != generic
    )

    return {
        "query": drug_name,
        "found": True,
        "generic_name": generic,
        "brand_names": brand_names,
    }


@router.get(
    "/dictionary/stats",
    response_model=DrugDictionaryStatsResponse,
    summary="Drug dictionary statistics",
    description="Get statistics about the medication dictionary coverage.",
)
async def dictionary_stats() -> dict[str, Any]:
    """Return statistics about the drug dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary size, unique generics, supported routes and statuses.
    """
    unique_generics = set(DRUG_DICTIONARY.values())

    return {
        "total_entries": len(DRUG_DICTIONARY),
        "unique_generics": len(unique_generics),
        "routes": [r.value for r in RouteOfAdministration if r != RouteOfAdministration.UNKNOWN],
        "statuses": [s.value for s in MedicationStatus if s != MedicationStatus.UNKNOWN],
    }
