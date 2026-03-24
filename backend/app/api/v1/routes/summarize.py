"""Summarization endpoint routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.deps import PipelineDep
from app.api.v1.schemas import SummarizeRequest, SummarizeResponse, SummaryResponse
from app.ml.pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarize", tags=["summarization"])


@router.post(
    "",
    response_model=SummarizeResponse,
    summary="Summarize clinical text",
    description="Generate a concise summary of clinical text.",
)
async def summarize_text(
    request: SummarizeRequest,
    pipeline: PipelineDep,
) -> SummarizeResponse:
    """Summarize clinical text."""
    import time

    start_time = time.time()

    try:
        result = pipeline.summarize(request.text, max_length=request.max_length)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Summarization service unavailable",
            )

        summary = SummaryResponse(
            summary=result.summary,
            original_length=result.original_length,
            summary_length=result.summary_length,
            compression_ratio=result.compression_ratio,
            summary_type=result.summary_type,
            key_points=result.key_points,
        )

        processing_time = (time.time() - start_time) * 1000

        return SummarizeResponse(
            summary=summary,
            processing_time_ms=processing_time,
            model_version=result.model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )


@router.post(
    "/sections",
    summary="Section-based summary",
    description="Generate section-based summary for structured clinical notes.",
)
async def summarize_by_sections(
    request: SummarizeRequest,
    pipeline: PipelineDep,
) -> dict:
    """Generate section-based summary."""
    from app.ml.summarization.model import SectionBasedSummarizer

    try:
        summarizer = SectionBasedSummarizer()
        summarizer.load()

        result = summarizer.summarize(request.text, max_length=request.max_length)

        return {
            "summary": result.summary,
            "summary_type": result.summary_type,
            "sections_extracted": result.key_points or [],
            "compression_ratio": result.compression_ratio,
        }

    except Exception as e:
        logger.error(f"Section-based summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )


@router.post(
    "/compare",
    summary="Compare summarization methods",
    description="Compare extractive vs abstractive summarization.",
)
async def compare_summaries(
    request: SummarizeRequest,
    pipeline: PipelineDep,
) -> dict:
    """Compare different summarization approaches."""
    import time

    results = {}

    try:
        # Extractive summary
        from app.ml.summarization.model import ExtractiveSummarizer

        start = time.time()
        extractive = ExtractiveSummarizer()
        extractive.load()
        ext_result = extractive.summarize(request.text, max_length=request.max_length)
        results["extractive"] = {
            "summary": ext_result.summary,
            "compression_ratio": ext_result.compression_ratio,
            "processing_time_ms": (time.time() - start) * 1000,
        }

        # Section-based summary
        from app.ml.summarization.model import SectionBasedSummarizer

        start = time.time()
        section_based = SectionBasedSummarizer()
        section_based.load()
        sec_result = section_based.summarize(request.text, max_length=request.max_length)
        results["section_based"] = {
            "summary": sec_result.summary,
            "compression_ratio": sec_result.compression_ratio,
            "processing_time_ms": (time.time() - start) * 1000,
        }

        # Hybrid summary (if available)
        if pipeline._summarizer:
            start = time.time()
            hybrid_result = pipeline.summarize(request.text, max_length=request.max_length)
            if hybrid_result:
                results["hybrid"] = {
                    "summary": hybrid_result.summary,
                    "compression_ratio": hybrid_result.compression_ratio,
                    "processing_time_ms": (time.time() - start) * 1000,
                }

        return {
            "original_length": len(request.text.split()),
            "summaries": results,
        }

    except Exception as e:
        logger.error(f"Summary comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary comparison failed: {str(e)}",
        )
