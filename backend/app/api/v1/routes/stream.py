"""Streaming analysis endpoint using Server-Sent Events (SSE).

Provides real-time, stage-by-stage progress updates during clinical NLP
analysis.  Each pipeline stage (NER, ICD-10, summarisation, risk) emits
its own SSE event as soon as it completes, allowing frontends to render
partial results incrementally instead of waiting for the full pipeline.

Design decisions
----------------
* **SSE over WebSocket** — The pipeline is a one-shot request → stream
  pattern; SSE is simpler, auto-reconnects, and works through HTTP/1.1
  proxies without upgrade negotiation.
* **JSON event payloads** — Each event's ``data`` field is a self-
  contained JSON object so the client can parse it with ``JSON.parse``
  and merge into local state without buffering the whole response.
* **Stage isolation** — A failure in one stage (e.g. summarisation
  timeout) emits an error event for that stage but does *not* abort
  remaining stages.  This mirrors the batch endpoint's partial-failure
  semantics and avoids all-or-nothing fragility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from app.api.schemas.analysis import AnalysisRequest
from app.api.schemas.icd import ICDCodeResponse
from app.api.schemas.ner import EntityResponse
from app.api.schemas.risk import RiskFactorResponse
from app.api.schemas.summary import SummarizationResponse
from app.services.model_registry import (
    get_icd_model,
    get_ner_model,
    get_risk_scorer,
    get_summarizer,
)

router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame.

    Parameters
    ----------
    event:
        Event name (e.g. ``"ner"``, ``"complete"``).
    data:
        JSON-serialisable payload.

    Returns
    -------
    str
        A properly formatted SSE text block ending with a blank line.
    """
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Stage runners (self-contained, no cross-module imports)
# ---------------------------------------------------------------------------


def _run_ner_stage(text: str, min_confidence: float) -> list[dict]:
    """Extract entities via the NER model and return serialisable dicts.

    Parameters
    ----------
    text:
        Clinical document text.
    min_confidence:
        Minimum entity confidence threshold.

    Returns
    -------
    list[dict]
        Entity dicts ready for JSON serialisation.
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
        ).model_dump()
        for e in raw
        if e.confidence >= min_confidence
    ]


def _run_icd_stage(text: str, top_k: int, min_confidence: float) -> list[dict]:
    """Predict ICD-10 codes and return serialisable dicts.

    Parameters
    ----------
    text:
        Clinical document text.
    top_k:
        Maximum predictions.
    min_confidence:
        Minimum prediction confidence.

    Returns
    -------
    list[dict]
        ICD prediction dicts.
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
        ).model_dump()
        for p in result.predictions
        if p.confidence >= min_confidence
    ]


def _run_summary_stage(text: str, detail_level: str) -> dict:
    """Generate a clinical summary and return a serialisable dict.

    Parameters
    ----------
    text:
        Clinical document text.
    detail_level:
        One of 'brief', 'standard', 'detailed'.

    Returns
    -------
    dict
        Summary response dict.
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
    ).model_dump()


def _run_risk_stage(text: str) -> dict:
    """Score clinical risk and return a serialisable dict.

    Parameters
    ----------
    text:
        Clinical document text.

    Returns
    -------
    dict
        Risk summary dict.
    """
    model = get_risk_scorer()
    assessment = model.assess_risk(text)

    top_factors = [
        RiskFactorResponse(
            name=f.name,
            description=f.description,
            weight=min(f.weight, 1.0),
            value=min(f.score / 100.0, 1.0) if f.score > 1 else f.score,
            source="derived",
            evidence=None,
        ).model_dump()
        for f in assessment.factors[:5]
        if f.score > 0
    ]

    return {
        "score": round(assessment.overall_score / 100.0, 4),
        "category": (
            "critical" if assessment.overall_score >= 80
            else "high" if assessment.overall_score >= 60
            else "moderate" if assessment.overall_score >= 40
            else "low"
        ),
        "top_factors": top_factors,
        "recommendations": assessment.recommendations,
    }


# ---------------------------------------------------------------------------
# Generator that runs each pipeline stage and yields SSE frames
# ---------------------------------------------------------------------------


async def _stream_analysis(request: AnalysisRequest) -> AsyncGenerator[str, None]:
    """Yield SSE events as each pipeline stage completes.

    The generator emits the following event sequence (skipping disabled
    stages):

    1. ``started``  — metadata: text length, document hash, timestamp
    2. ``ner``      — entities list + timing
    3. ``icd``      — ICD-10 predictions + timing
    4. ``summary``  — extractive summary + timing
    5. ``risk``     — risk assessment + timing
    6. ``complete`` — aggregate timing + stage count

    If a stage raises an exception the generator emits a ``stage_error``
    event with the stage name and error message, then continues with the
    next stage.

    Parameters
    ----------
    request:
        Validated analysis request with pipeline config.

    Yields
    ------
    str
        SSE-formatted text frames.
    """
    wall_start = time.monotonic()
    doc_hash = hashlib.sha256(request.text.encode()).hexdigest()

    # --- started event ---
    yield _sse_event("started", {
        "text_length": len(request.text),
        "document_hash": doc_hash,
        "document_id": request.document_id,
        "stages_enabled": {
            "ner": request.config.ner.enabled,
            "icd": request.config.icd.enabled,
            "summary": request.config.summary.enabled,
            "risk": request.config.risk.enabled,
        },
    })

    cfg = request.config
    stages_completed = 0
    timings: dict[str, float] = {}

    # --- NER ---
    if cfg.ner.enabled:
        try:
            t0 = time.monotonic()
            entities = _run_ner_stage(request.text, cfg.ner.min_confidence)
            elapsed = (time.monotonic() - t0) * 1000
            timings["ner_ms"] = round(elapsed, 2)
            stages_completed += 1
            yield _sse_event("ner", {
                "entities": entities,
                "count": len(entities),
                "processing_time_ms": round(elapsed, 2),
            })
        except Exception as exc:
            logger.exception("Streaming NER stage failed")
            yield _sse_event("stage_error", {
                "stage": "ner",
                "error": str(exc),
            })

    # --- ICD-10 ---
    if cfg.icd.enabled:
        try:
            t0 = time.monotonic()
            predictions = _run_icd_stage(
                request.text, cfg.icd.top_k, cfg.icd.min_confidence,
            )
            elapsed = (time.monotonic() - t0) * 1000
            timings["icd_ms"] = round(elapsed, 2)
            stages_completed += 1
            yield _sse_event("icd", {
                "predictions": predictions,
                "count": len(predictions),
                "processing_time_ms": round(elapsed, 2),
            })
        except Exception as exc:
            logger.exception("Streaming ICD stage failed")
            yield _sse_event("stage_error", {
                "stage": "icd",
                "error": str(exc),
            })

    # --- Summarisation ---
    if cfg.summary.enabled:
        try:
            t0 = time.monotonic()
            summary = _run_summary_stage(request.text, cfg.summary.detail_level)
            elapsed = (time.monotonic() - t0) * 1000
            summary["processing_time_ms"] = elapsed
            timings["summary_ms"] = round(elapsed, 2)
            stages_completed += 1
            yield _sse_event("summary", {
                "summary": summary,
                "processing_time_ms": round(elapsed, 2),
            })
        except Exception as exc:
            logger.exception("Streaming summary stage failed")
            yield _sse_event("stage_error", {
                "stage": "summary",
                "error": str(exc),
            })

    # --- Risk ---
    if cfg.risk.enabled:
        try:
            t0 = time.monotonic()
            risk = _run_risk_stage(request.text)
            elapsed = (time.monotonic() - t0) * 1000
            timings["risk_ms"] = round(elapsed, 2)
            stages_completed += 1
            yield _sse_event("risk", {
                "risk_score": risk,
                "processing_time_ms": round(elapsed, 2),
            })
        except Exception as exc:
            logger.exception("Streaming risk stage failed")
            yield _sse_event("stage_error", {
                "stage": "risk",
                "error": str(exc),
            })

    # --- complete ---
    total_ms = (time.monotonic() - wall_start) * 1000
    yield _sse_event("complete", {
        "stages_completed": stages_completed,
        "total_processing_time_ms": round(total_ms, 2),
        "stage_timings": timings,
    })


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/analyze/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream clinical NLP analysis via Server-Sent Events",
    description=(
        "Runs the same full clinical pipeline as ``POST /analyze`` but "
        "returns results incrementally as Server-Sent Events.  Each "
        "pipeline stage emits its own event so the frontend can render "
        "partial results immediately instead of waiting for the entire "
        "pipeline to finish.\n\n"
        "**Event types:** ``started``, ``ner``, ``icd``, ``summary``, "
        "``risk``, ``stage_error``, ``complete``."
    ),
    responses={
        200: {
            "description": "SSE stream of analysis results",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_analysis(
    payload: AnalysisRequest,
    request: Request,
) -> StreamingResponse:
    """Stream pipeline results as Server-Sent Events.

    Parameters
    ----------
    payload:
        The analysis request with clinical text and pipeline configuration.
    request:
        FastAPI request for client metadata.

    Returns
    -------
    StreamingResponse
        ``text/event-stream`` response that yields one SSE frame per
        pipeline stage.
    """
    logger.info(
        "Starting streaming analysis — text_length=%d",
        len(payload.text),
    )
    return StreamingResponse(
        _stream_analysis(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )
