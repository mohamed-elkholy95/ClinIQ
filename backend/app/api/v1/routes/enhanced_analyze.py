"""Enhanced full-pipeline analysis endpoint.

Exposes the :class:`EnhancedClinicalPipeline` via a single ``POST /analyze/enhanced``
endpoint that runs all 14+ clinical NLP modules in one request.  Callers can
toggle individual modules via the request body.

Design decisions
----------------
* **Superset of /analyze** — This endpoint produces everything ``/analyze``
  does (via the base pipeline) plus all enhanced module outputs (sections,
  quality, medications, allergies, vitals, etc.).
* **Configurable** — Each module can be enabled/disabled per-request,
  defaulting to everything ON except de-identification.
* **Fault-tolerant** — Module failures are captured in ``component_errors``
  without aborting the overall request.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.ml.enhanced_pipeline import (
    EnhancedClinicalPipeline,
    EnhancedPipelineConfig,
)

router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)

# Singleton pipeline instance (modules initialized lazily)
_pipeline: EnhancedClinicalPipeline | None = None


def _get_pipeline() -> EnhancedClinicalPipeline:
    """Return the singleton enhanced pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = EnhancedClinicalPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class EnhancedAnalysisRequest(BaseModel):
    """Request body for ``POST /analyze/enhanced``.

    Attributes
    ----------
    text:
        Clinical document text to analyse.
    document_id:
        Optional caller-supplied document identifier.
    enable_classification:
        Run document type classification.
    enable_sections:
        Run section parsing.
    enable_quality:
        Run clinical note quality analysis.
    enable_deidentification:
        Run PHI de-identification (off by default).
    enable_abbreviations:
        Run abbreviation expansion.
    enable_medications:
        Run medication extraction.
    enable_allergies:
        Run allergy extraction.
    enable_vitals:
        Run vital signs extraction.
    enable_temporal:
        Run temporal expression extraction.
    enable_assertions:
        Run assertion status detection.
    enable_normalization:
        Run concept normalization.
    enable_sdoh:
        Run SDoH extraction.
    enable_relations:
        Run relation extraction.
    enable_comorbidity:
        Run Charlson Comorbidity Index calculation.
    """

    text: str = Field(..., min_length=1, max_length=500_000)
    document_id: str | None = None
    enable_classification: bool = True
    enable_sections: bool = True
    enable_quality: bool = True
    enable_deidentification: bool = False
    enable_abbreviations: bool = True
    enable_medications: bool = True
    enable_allergies: bool = True
    enable_vitals: bool = True
    enable_temporal: bool = True
    enable_assertions: bool = True
    enable_normalization: bool = True
    enable_sdoh: bool = True
    enable_relations: bool = True
    enable_comorbidity: bool = True


class EnhancedBatchRequest(BaseModel):
    """Request body for ``POST /analyze/enhanced/batch``."""

    documents: list[EnhancedAnalysisRequest] = Field(
        ..., min_length=1, max_length=20,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/analyze/enhanced",
    summary="Enhanced full-pipeline analysis",
    response_description="Comprehensive clinical NLP analysis with all modules",
    status_code=status.HTTP_200_OK,
)
async def enhanced_analyze(request: EnhancedAnalysisRequest) -> dict:
    """Run the enhanced clinical NLP pipeline on a single document.

    Executes all enabled analysis modules and returns a comprehensive
    result including document classification, section parsing, quality
    analysis, medication/allergy/vital extraction, SDoH, comorbidity
    scoring, and more.

    Parameters
    ----------
    request:
        Analysis request with text and module toggles.

    Returns
    -------
    dict
        Complete enhanced pipeline result.
    """
    pipeline = _get_pipeline()

    config = EnhancedPipelineConfig(
        # Disable base pipeline models (not loaded in rule-based mode)
        enable_ner=False,
        enable_icd=False,
        enable_summarization=False,
        enable_risk=False,
        enable_dental=False,
        # Enhanced modules from request
        enable_classification=request.enable_classification,
        enable_sections=request.enable_sections,
        enable_quality=request.enable_quality,
        enable_deidentification=request.enable_deidentification,
        enable_abbreviations=request.enable_abbreviations,
        enable_medications=request.enable_medications,
        enable_allergies=request.enable_allergies,
        enable_vitals=request.enable_vitals,
        enable_temporal=request.enable_temporal,
        enable_assertions=request.enable_assertions,
        enable_normalization=request.enable_normalization,
        enable_sdoh=request.enable_sdoh,
        enable_relations=request.enable_relations,
        enable_comorbidity=request.enable_comorbidity,
    )

    try:
        result = pipeline.process(
            text=request.text,
            config=config,
            document_id=request.document_id,
        )
        return result.to_dict()
    except Exception as exc:
        logger.exception("Enhanced analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc


@router.post(
    "/analyze/enhanced/batch",
    summary="Batch enhanced analysis",
    response_description="Multiple document analysis results",
    status_code=status.HTTP_200_OK,
)
async def enhanced_analyze_batch(request: EnhancedBatchRequest) -> dict:
    """Run enhanced analysis on multiple documents.

    Parameters
    ----------
    request:
        Batch request with up to 20 documents.

    Returns
    -------
    dict
        List of results with aggregate statistics.
    """
    pipeline = _get_pipeline()
    results = []

    for doc_req in request.documents:
        config = EnhancedPipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=False,
            enable_classification=doc_req.enable_classification,
            enable_sections=doc_req.enable_sections,
            enable_quality=doc_req.enable_quality,
            enable_deidentification=doc_req.enable_deidentification,
            enable_abbreviations=doc_req.enable_abbreviations,
            enable_medications=doc_req.enable_medications,
            enable_allergies=doc_req.enable_allergies,
            enable_vitals=doc_req.enable_vitals,
            enable_temporal=doc_req.enable_temporal,
            enable_assertions=doc_req.enable_assertions,
            enable_normalization=doc_req.enable_normalization,
            enable_sdoh=doc_req.enable_sdoh,
            enable_relations=doc_req.enable_relations,
            enable_comorbidity=doc_req.enable_comorbidity,
        )
        result = pipeline.process(
            text=doc_req.text,
            config=config,
            document_id=doc_req.document_id,
        )
        results.append(result.to_dict())

    return {
        "results": results,
        "document_count": len(results),
    }


@router.get(
    "/analyze/enhanced/modules",
    summary="List available enhanced modules",
    response_description="Catalogue of all enhanced analysis modules",
    status_code=status.HTTP_200_OK,
)
async def list_enhanced_modules() -> dict:
    """Return a catalogue of all available enhanced analysis modules.

    Returns
    -------
    dict
        Module names, descriptions, and default enabled states.
    """
    return {
        "modules": [
            {
                "name": "classification",
                "description": "Document type classification (13 clinical document types)",
                "default_enabled": True,
            },
            {
                "name": "sections",
                "description": "Section parsing (35 clinical categories with offset tracking)",
                "default_enabled": True,
            },
            {
                "name": "quality",
                "description": "Clinical note quality analysis (5 dimensions, letter grading)",
                "default_enabled": True,
            },
            {
                "name": "deidentification",
                "description": "PHI de-identification (18 HIPAA Safe Harbor categories)",
                "default_enabled": False,
            },
            {
                "name": "abbreviations",
                "description": "Abbreviation expansion (220+ unambiguous, 10 ambiguous entries)",
                "default_enabled": True,
            },
            {
                "name": "medications",
                "description": "Structured medication extraction (220+ drug dictionary)",
                "default_enabled": True,
            },
            {
                "name": "allergies",
                "description": "Allergy extraction (150+ allergens, 30+ reaction patterns)",
                "default_enabled": True,
            },
            {
                "name": "vitals",
                "description": "Vital signs extraction (9 types with clinical interpretation)",
                "default_enabled": True,
            },
            {
                "name": "temporal",
                "description": "Temporal extraction (dates, durations, frequencies, ages)",
                "default_enabled": True,
            },
            {
                "name": "assertions",
                "description": "Assertion detection (present/absent/possible/conditional/hypothetical/family)",
                "default_enabled": True,
            },
            {
                "name": "normalization",
                "description": "Concept normalization (UMLS CUI, SNOMED-CT, RxNorm, ICD-10, LOINC)",
                "default_enabled": True,
            },
            {
                "name": "sdoh",
                "description": "Social Determinants of Health (8 domains, Z-code mapping)",
                "default_enabled": True,
            },
            {
                "name": "relations",
                "description": "Clinical relation extraction (12 semantic types)",
                "default_enabled": True,
            },
            {
                "name": "comorbidity",
                "description": "Charlson Comorbidity Index (17 disease categories)",
                "default_enabled": True,
            },
        ],
        "total_modules": 14,
    }
