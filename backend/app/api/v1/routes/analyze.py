"""Full-pipeline analysis endpoint.

Orchestrates NER, ICD-10 prediction, clinical summarisation, and risk scoring
into a single HTTP call. Individual stage results are aggregated into one
AnalysisResponse and the invocation is written to the audit trail.
"""

from __future__ import annotations

import hashlib
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    RiskSummary,
    StageTiming,
)
from app.api.schemas.icd import ICDCodeResponse
from app.api.schemas.ner import EntityResponse
from app.api.schemas.risk import RiskFactorResponse
from app.api.schemas.summary import SummarizationResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentProcessingError
from app.db.session import get_db_session

router = APIRouter(tags=["analysis"])


# ---------------------------------------------------------------------------
# Placeholder pipeline stage helpers
# Replace with real ML service calls once the model layer is ready.
# ---------------------------------------------------------------------------


def _mock_ner(text: str, min_confidence: float) -> list[EntityResponse]:
    """Return placeholder NER entities."""
    return [
        EntityResponse(
            text="chest pain",
            entity_type="SYMPTOM",
            start_char=0,
            end_char=10,
            confidence=0.92,
            normalized_text="chest pain",
            umls_cui="C0008031",
            is_negated=False,
            is_uncertain=False,
        ),
        EntityResponse(
            text="hypertension",
            entity_type="DISEASE",
            start_char=15,
            end_char=27,
            confidence=0.88,
            normalized_text="Hypertension",
            umls_cui="C0020538",
            is_negated=False,
            is_uncertain=False,
        ),
    ]


def _mock_icd(text: str, top_k: int, min_confidence: float) -> list[ICDCodeResponse]:
    """Return placeholder ICD-10 predictions."""
    all_codes = [
        ICDCodeResponse(
            code="R07.9",
            description="Chest pain, unspecified",
            confidence=0.87,
            chapter="Symptoms, signs and abnormal clinical and laboratory findings",
            category="R07",
            contributing_text=["chest pain"],
        ),
        ICDCodeResponse(
            code="I25.10",
            description="Atherosclerotic heart disease of native coronary artery without angina pectoris",
            confidence=0.61,
            chapter="Diseases of the circulatory system",
            category="I25",
            contributing_text=None,
        ),
    ]
    return [c for c in all_codes if c.confidence >= min_confidence][:top_k]


def _mock_summary(text: str, detail_level: str) -> SummarizationResponse:
    """Return a placeholder clinical summary."""
    word_count = len(text.split())
    truncated = text[:100].rstrip()
    return SummarizationResponse(
        summary=f"[Placeholder summary] {truncated}...",
        key_points=["Placeholder key point 1", "Placeholder key point 2"],
        original_word_count=word_count,
        summary_word_count=15,
        compression_ratio=round(word_count / 15, 2) if word_count else 1.0,
        summary_type="extractive",
        model_name="textrank",
        model_version="1.0.0",
        processing_time_ms=0.0,
    )


def _mock_risk(text: str) -> RiskSummary:
    """Return a placeholder risk assessment."""
    return RiskSummary(
        score=0.55,
        category="moderate",
        top_factors=[
            RiskFactorResponse(
                name="placeholder_factor",
                description="Placeholder risk factor — wire up real risk model.",
                weight=0.5,
                value=0.5,
                source="text",
                evidence=None,
            )
        ],
        recommendations=["Placeholder recommendation — wire up real risk model."],
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
            entities = _mock_ner(payload.text, cfg.ner.min_confidence)
            ner_ms = (time.monotonic() - t0) * 1000

        # --- ICD-10 prediction stage ---
        if cfg.icd.enabled:
            t0 = time.monotonic()
            icd_codes = _mock_icd(payload.text, cfg.icd.top_k, cfg.icd.min_confidence)
            icd_ms = (time.monotonic() - t0) * 1000

        # --- Summarisation stage ---
        if cfg.summary.enabled:
            t0 = time.monotonic()
            summary = _mock_summary(payload.text, cfg.summary.detail_level)
            summary_ms = (time.monotonic() - t0) * 1000
            summary.processing_time_ms = summary_ms

        # --- Risk scoring stage ---
        if cfg.risk.enabled:
            t0 = time.monotonic()
            risk_result = _mock_risk(payload.text)
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
