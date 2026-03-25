"""Full-pipeline analysis endpoint.

Orchestrates NER, ICD-10 prediction, clinical summarisation, and risk scoring
into a single HTTP call using real ML models from the model registry.
Individual stage results are aggregated into one AnalysisResponse and the
invocation is written to the audit trail.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.analysis import AnalysisRequest, AnalysisResponse, RiskSummary, StageTiming
from app.api.schemas.icd import ICDCodeResponse
from app.api.schemas.ner import EntityResponse
from app.api.schemas.risk import RiskFactorResponse
from app.api.schemas.summary import SummarizationResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.db.session import get_db_session
from app.services.model_registry import (
    get_icd_model,
    get_ner_model,
    get_risk_scorer,
    get_summarizer,
)

router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real pipeline stage helpers
# ---------------------------------------------------------------------------


def _run_ner(text: str, min_confidence: float) -> list[EntityResponse]:
    """Extract clinical entities using the NER model.

    Parameters
    ----------
    text:
        Raw clinical document text.
    min_confidence:
        Minimum confidence threshold for returned entities.

    Returns
    -------
    list[EntityResponse]
    """
    model = get_ner_model()
    raw = model.extract_entities(text)
    return [
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
        for e in raw
        if e.confidence >= min_confidence
    ]


def _run_icd(text: str, top_k: int, min_confidence: float) -> list[ICDCodeResponse]:
    """Predict ICD-10 codes using the ICD classifier.

    Parameters
    ----------
    text:
        Raw clinical document text.
    top_k:
        Maximum number of predictions to return.
    min_confidence:
        Minimum confidence threshold.

    Returns
    -------
    list[ICDCodeResponse]
    """
    model = get_icd_model()
    result = model.predict(text, top_k=top_k)
    return [
        ICDCodeResponse(
            code=p.code,
            description=p.description,
            confidence=p.confidence,
            chapter=p.chapter,
            category=p.category,
            contributing_text=p.contributing_text,
        )
        for p in result.predictions
        if p.confidence >= min_confidence
    ]


def _run_summary(text: str, detail_level: str) -> SummarizationResponse:
    """Generate a clinical summary using the extractive summarizer.

    Parameters
    ----------
    text:
        Raw clinical document text.
    detail_level:
        One of 'brief', 'standard', 'detailed'.

    Returns
    -------
    SummarizationResponse
    """
    model = get_summarizer()
    result = model.summarize(text, detail_level=detail_level)

    original_wc = len(text.split())
    summary_wc = len(result.summary.split())
    compression = round(original_wc / summary_wc, 2) if summary_wc else 1.0

    return SummarizationResponse(
        summary=result.summary,
        key_points=result.key_findings,
        original_word_count=original_wc,
        summary_word_count=summary_wc,
        compression_ratio=compression,
        summary_type="extractive",
        model_name=model.model_name,
        model_version=model.version,
        processing_time_ms=0.0,
    )


def _run_risk(text: str) -> RiskSummary:
    """Score clinical risk using the rule-based risk scorer.

    Parameters
    ----------
    text:
        Raw clinical document text.

    Returns
    -------
    RiskSummary
    """
    model = get_risk_scorer()
    assessment = model.assess_risk(text)

    top_factors = [
        RiskFactorResponse(
            name=f.name,
            description=f.description,
            weight=f.weight,
            value=f.score,
            source="derived",
            evidence=None,
        )
        for f in assessment.factors[:5]
        if f.score > 0
    ]

    return RiskSummary(
        score=round(assessment.overall_score / 100.0, 4),
        category="critical" if assessment.overall_score >= 80
                 else "high" if assessment.overall_score >= 60
                 else "moderate" if assessment.overall_score >= 40
                 else "low",
        top_factors=top_factors,
        recommendations=assessment.recommendations,
    )


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


async def _write_audit_log(
    db: AsyncSession,
    *,
    action: str,
    document_hash: str,
    ip_address: str | None,
    user_agent: str | None,
    status_code: int,
    response_time_ms: int,
) -> None:
    """Insert a row into audit_log.

    Silently swallowed on failure to avoid masking the primary response —
    audit failures should be handled by an async alerting pipeline.
    """
    try:
        from app.db.models import AuditLog

        entry = AuditLog(
            action=action,
            resource_type="analysis",
            document_hash=document_hash,
            ip_address=ip_address,
            user_agent=user_agent,
            status_code=status_code,
            response_time_ms=response_time_ms,
        )
        db.add(entry)
        # get_db_session commits after the request completes.
    except Exception:
        pass  # Non-critical path — never surface audit errors to the caller.


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Full clinical NLP pipeline",
    description=(
        "Run the complete ClinIQ analysis pipeline over the supplied clinical text. "
        "Each stage — NER, ICD-10 prediction, summarisation, and risk scoring — can be "
        "toggled individually via the `config` object. "
        "Results from all active stages are returned in a single response with per-stage "
        "timing. Every invocation is written to the audit trail."
    ),
    responses={
        200: {"description": "Analysis completed successfully"},
        422: {"description": "Input validation error"},
        500: {"description": "Internal pipeline error"},
    },
)
async def run_analysis(
    payload: AnalysisRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisResponse:
    """Orchestrate all active pipeline stages and return the aggregated result."""
    wall_start = time.monotonic()
    document_hash = hashlib.sha256(payload.text.encode()).hexdigest()

    try:
        cfg = payload.config
        ner_ms: float | None = None
        icd_ms: float | None = None
        summary_ms: float | None = None
        risk_ms: float | None = None

        entities: list[EntityResponse] | None = None
        icd_codes: list[ICDCodeResponse] | None = None
        summary: SummarizationResponse | None = None
        risk_result: RiskSummary | None = None

        # --- NER stage ---
        if cfg.ner.enabled:
            t0 = time.monotonic()
            entities = _run_ner(payload.text, cfg.ner.min_confidence)
            ner_ms = (time.monotonic() - t0) * 1000

        # --- ICD-10 prediction stage ---
        if cfg.icd.enabled:
            t0 = time.monotonic()
            icd_codes = _run_icd(payload.text, cfg.icd.top_k, cfg.icd.min_confidence)
            icd_ms = (time.monotonic() - t0) * 1000

        # --- Summarisation stage ---
        if cfg.summary.enabled:
            t0 = time.monotonic()
            summary = _run_summary(payload.text, cfg.summary.detail_level)
            summary_ms = (time.monotonic() - t0) * 1000
            summary.processing_time_ms = summary_ms

        # --- Risk scoring stage ---
        if cfg.risk.enabled:
            t0 = time.monotonic()
            risk_result = _run_risk(payload.text)
            risk_ms = (time.monotonic() - t0) * 1000

        total_ms = (time.monotonic() - wall_start) * 1000

        await _write_audit_log(
            db,
            action="analyze",
            document_hash=document_hash,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status_code=200,
            response_time_ms=int(total_ms),
        )

        return AnalysisResponse(
            document_id=payload.document_id,
            result_id=None,
            text_length=len(payload.text),
            entities=entities,
            icd_codes=icd_codes,
            summary=summary,
            risk_score=risk_result,
            timing=StageTiming(
                ner_ms=ner_ms,
                icd_ms=icd_ms,
                summary_ms=summary_ms,
                risk_ms=risk_ms,
                total_ms=total_ms,
            ),
        )

    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis. Please try again.",
        ) from exc
