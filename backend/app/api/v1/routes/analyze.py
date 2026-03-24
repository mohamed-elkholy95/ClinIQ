"""Main analysis endpoint."""

import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps import PipelineDep
from app.api.v1.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    EntityResponse,
    ICDCodePrediction,
    RiskScoreResponse,
    RiskFactor,
    SummaryResponse,
)
from app.ml.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post(
    "",
    response_model=AnalysisResponse,
    summary="Analyze clinical text",
    description="Perform comprehensive analysis of clinical text including NER, ICD-10 coding, summarization, and risk scoring.",
)
async def analyze_text(
    request: AnalyzeRequest,
    pipeline: PipelineDep,
) -> AnalysisResponse:
    """Analyze clinical text through the full ML pipeline."""
    try:
        # Build pipeline config from request
        config = PipelineConfig(
            enable_ner=request.enable_ner,
            enable_icd=request.enable_icd,
            enable_summarization=request.enable_summarization,
            enable_risk=request.enable_risk,
            max_icd_codes=request.max_icd_codes,
            summary_max_length=request.summary_max_length,
        )

        # Run analysis
        result = pipeline.analyze(
            text=request.text,
            document_id=request.document_id,
            config_override=config,
        )

        # Build response
        response = AnalysisResponse(
            document_id=result.document_id,
            text_hash=result.text_hash,
            entities=[
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
                for e in result.entities
            ],
            icd_predictions=[
                ICDCodePrediction(**pred) for pred in result.icd_predictions
            ],
            summary=_build_summary_response(result.summary) if result.summary else None,
            risk_score=_build_risk_response(result.risk_score) if result.risk_score else None,
            processing_time_ms=result.processing_time_ms,
            component_times_ms=result.component_times_ms,
            model_versions=result.model_versions,
        )

        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


def _build_summary_response(summary_result) -> SummaryResponse:
    """Build summary response from result."""
    return SummaryResponse(
        summary=summary_result.summary,
        original_length=summary_result.original_length,
        summary_length=summary_result.summary_length,
        compression_ratio=summary_result.compression_ratio,
        summary_type=summary_result.summary_type,
        key_points=summary_result.key_points,
    )


def _build_risk_response(risk_result) -> RiskScoreResponse:
    """Build risk response from result."""
    return RiskScoreResponse(
        overall_score=risk_result.overall_score,
        risk_level=risk_result.risk_level,
        category_scores=risk_result.category_scores,
        risk_factors=[
            RiskFactor(
                name=rf.name,
                description=rf.description,
                weight=rf.weight,
                value=rf.value,
                source=rf.source,
                evidence=rf.evidence,
            )
            for rf in risk_result.risk_factors
        ],
        protective_factors=[
            RiskFactor(
                name=pf.name,
                description=pf.description,
                weight=pf.weight,
                value=pf.value,
                source=pf.source,
                evidence=pf.evidence,
            )
            for pf in risk_result.protective_factors
        ],
        recommendations=risk_result.recommendations,
    )
