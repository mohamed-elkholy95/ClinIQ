"""Document type classification API endpoints.

Provides endpoints for classifying clinical documents by type
(discharge summary, progress note, operative note, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.classifier.document_classifier import (
    ClassificationResult,
    DocumentType,
    RuleBasedDocumentClassifier,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level classifier instance (lightweight, no ML deps to load)
_classifier = RuleBasedDocumentClassifier()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """Request schema for document classification.

    Attributes
    ----------
    text:
        Clinical document text to classify.
    min_confidence:
        Minimum confidence threshold for returned scores.
    top_k:
        Maximum number of ranked predictions to return.
    """

    text: str = Field(..., min_length=10, max_length=200_000)
    min_confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=14)


class ClassifyBatchRequest(BaseModel):
    """Request schema for batch document classification.

    Attributes
    ----------
    documents:
        List of clinical document texts (max 50).
    min_confidence:
        Minimum confidence threshold for returned scores.
    top_k:
        Maximum number of ranked predictions per document.
    """

    documents: list[str] = Field(..., min_length=1, max_length=50)
    min_confidence: float = Field(default=0.05, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=14)


class ClassificationScoreResponse(BaseModel):
    """Response schema for a single classification score."""

    document_type: str
    confidence: float
    evidence: list[str]


class ClassifyResponse(BaseModel):
    """Response schema for document classification."""

    predicted_type: str
    scores: list[ClassificationScoreResponse]
    processing_time_ms: float
    classifier_version: str


class ClassifyBatchResponse(BaseModel):
    """Response schema for batch classification."""

    results: list[ClassifyResponse]
    total_documents: int
    total_processing_time_ms: float


class DocumentTypeInfo(BaseModel):
    """Information about a single document type."""

    type: str
    description: str


class DocumentTypesResponse(BaseModel):
    """Response listing all supported document types."""

    types: list[DocumentTypeInfo]
    count: int


# ---------------------------------------------------------------------------
# Description map for the catalogue endpoint
# ---------------------------------------------------------------------------

_TYPE_DESCRIPTIONS: dict[DocumentType, str] = {
    DocumentType.DISCHARGE_SUMMARY: "Summary issued when a patient is discharged from a hospital stay",
    DocumentType.PROGRESS_NOTE: "Daily clinical note documenting patient status and plan (often SOAP format)",
    DocumentType.HISTORY_PHYSICAL: "Comprehensive history and physical examination (H&P)",
    DocumentType.OPERATIVE_NOTE: "Surgeon's documentation of a performed procedure",
    DocumentType.CONSULTATION_NOTE: "Specialist evaluation requested by another provider",
    DocumentType.RADIOLOGY_REPORT: "Interpretation of radiological imaging studies",
    DocumentType.PATHOLOGY_REPORT: "Histopathological or cytopathological analysis of specimens",
    DocumentType.LABORATORY_REPORT: "Clinical laboratory test results and reference ranges",
    DocumentType.NURSING_NOTE: "Nursing assessment and documentation of patient care",
    DocumentType.EMERGENCY_NOTE: "Emergency department encounter documentation",
    DocumentType.DENTAL_NOTE: "Dental examination and procedure documentation",
    DocumentType.PRESCRIPTION: "Medication prescription with dosage instructions",
    DocumentType.REFERRAL: "Referral letter or request to a specialist or service",
    DocumentType.UNKNOWN: "Document type could not be determined",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _result_to_response(
    result: ClassificationResult,
    top_k: int,
    min_confidence: float,
) -> dict[str, Any]:
    """Convert a ClassificationResult to a response dict.

    Parameters
    ----------
    result:
        The classification result.
    top_k:
        Maximum number of scores to include.
    min_confidence:
        Minimum confidence threshold.

    Returns
    -------
    dict
        Serialised response data.
    """
    filtered_scores = [
        s for s in result.scores if s.confidence >= min_confidence
    ][:top_k]

    return {
        "predicted_type": result.predicted_type.value,
        "scores": [s.to_dict() for s in filtered_scores],
        "processing_time_ms": round(result.processing_time_ms, 2),
        "classifier_version": result.classifier_version,
    }


@router.post("/classify", response_model=ClassifyResponse)
async def classify_document(request: ClassifyRequest) -> dict[str, Any]:
    """Classify a clinical document by type.

    Analyses the input text and returns ranked predictions of the clinical
    document type with confidence scores and supporting evidence.

    Parameters
    ----------
    request:
        ClassifyRequest with the document text and optional parameters.

    Returns
    -------
    ClassifyResponse
        Predicted document type and ranked scores.
    """
    try:
        result = _classifier.classify(request.text)
        return _result_to_response(result, request.top_k, request.min_confidence)
    except Exception as e:
        logger.error("Classification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/classify/batch", response_model=ClassifyBatchResponse)
async def classify_documents_batch(
    request: ClassifyBatchRequest,
) -> dict[str, Any]:
    """Classify multiple clinical documents by type.

    Processes up to 50 documents in a single request and returns ranked
    predictions for each.

    Parameters
    ----------
    request:
        ClassifyBatchRequest with document texts and optional parameters.

    Returns
    -------
    ClassifyBatchResponse
        List of classification results.
    """
    try:
        results = _classifier.classify_batch(request.documents)
        responses = [
            _result_to_response(r, request.top_k, request.min_confidence)
            for r in results
        ]
        total_time = sum(r.processing_time_ms for r in results)
        return {
            "results": responses,
            "total_documents": len(results),
            "total_processing_time_ms": round(total_time, 2),
        }
    except Exception as e:
        logger.error("Batch classification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/classify/types", response_model=DocumentTypesResponse)
async def list_document_types() -> dict[str, Any]:
    """List all supported clinical document types.

    Returns a catalogue of document types that the classifier can identify,
    with human-readable descriptions.

    Returns
    -------
    DocumentTypesResponse
        List of supported document types and their descriptions.
    """
    types = [
        {"type": dt.value, "description": _TYPE_DESCRIPTIONS.get(dt, "")}
        for dt in DocumentType
        if dt != DocumentType.UNKNOWN
    ]
    return {"types": types, "count": len(types)}
