"""Assertion detection API endpoints.

Provides REST endpoints for classifying clinical entity assertions
as present, absent, possible, conditional, hypothetical, or family.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.ml.assertions import AssertionResult, AssertionStatus, ConTextAssertionDetector

logger = logging.getLogger(__name__)

# Module-level detector instance (lazy initialization)
_detector: ConTextAssertionDetector | None = None


def _get_detector() -> ConTextAssertionDetector:
    """Get or create the assertion detector singleton.

    Returns
    -------
    ConTextAssertionDetector
        Shared detector instance.
    """
    global _detector
    if _detector is None:
        _detector = ConTextAssertionDetector()
    return _detector


# -- Request/Response schemas ---


class EntitySpan(BaseModel):
    """An entity span to classify.

    Parameters
    ----------
    start : int
        Start character offset (inclusive).
    end : int
        End character offset (exclusive).
    label : str | None
        Optional entity type label for context.
    """

    start: int = Field(..., ge=0, description="Start character offset")
    end: int = Field(..., ge=0, description="End character offset")
    label: str | None = Field(None, description="Optional entity type label")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        """Validate that end is after start."""
        if "start" in info.data and v <= info.data["start"]:
            msg = "end must be greater than start"
            raise ValueError(msg)
        return v


class AssertionRequest(BaseModel):
    """Request for assertion detection on a single entity.

    Parameters
    ----------
    text : str
        Clinical text containing the entity.
    entity_start : int
        Entity start character offset.
    entity_end : int
        Entity end character offset.
    """

    text: str = Field(..., min_length=1, max_length=100_000, description="Clinical text")
    entity_start: int = Field(..., ge=0, description="Entity start offset")
    entity_end: int = Field(..., ge=0, description="Entity end offset")

    @field_validator("entity_end")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        """Validate that entity_end > entity_start."""
        if "entity_start" in info.data and v <= info.data["entity_start"]:
            msg = "entity_end must be greater than entity_start"
            raise ValueError(msg)
        return v


class AssertionBatchRequest(BaseModel):
    """Request for batch assertion detection.

    Parameters
    ----------
    text : str
        Clinical text containing all entities.
    entities : list[EntitySpan]
        Entity spans to classify.
    """

    text: str = Field(..., min_length=1, max_length=100_000, description="Clinical text")
    entities: list[EntitySpan] = Field(
        ..., min_length=1, max_length=200, description="Entity spans"
    )


class AssertionResponse(BaseModel):
    """Response for a single assertion detection.

    Parameters
    ----------
    status : str
        Assertion status (present, absent, possible, etc.).
    confidence : float
        Detection confidence.
    trigger_text : str | None
        Text that triggered the assertion change.
    trigger_type : str | None
        Type of trigger (pre, post, pseudo).
    entity_text : str
        The classified entity text.
    sentence : str
        Sentence containing the entity.
    metadata : dict[str, Any]
        Additional detection metadata.
    """

    status: str
    confidence: float
    trigger_text: str | None
    trigger_type: str | None
    entity_text: str
    sentence: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssertionBatchResponse(BaseModel):
    """Response for batch assertion detection.

    Parameters
    ----------
    results : list[AssertionResponse]
        Assertion results for each entity.
    total : int
        Total entities processed.
    processing_time_ms : float
        Processing time in milliseconds.
    summary : dict[str, int]
        Count of each assertion status.
    """

    results: list[AssertionResponse]
    total: int
    processing_time_ms: float
    summary: dict[str, int]


class AssertionStatusInfo(BaseModel):
    """Information about an assertion status type.

    Parameters
    ----------
    status : str
        Status value.
    description : str
        Human-readable description.
    """

    status: str
    description: str


def _result_to_response(result: AssertionResult) -> AssertionResponse:
    """Convert internal result to API response.

    Parameters
    ----------
    result : AssertionResult
        Internal assertion result.

    Returns
    -------
    AssertionResponse
        API response model.
    """
    return AssertionResponse(
        status=result.status.value,
        confidence=round(result.confidence, 4),
        trigger_text=result.trigger_text,
        trigger_type=result.trigger_type.value if result.trigger_type else None,
        entity_text=result.entity_text,
        sentence=result.sentence,
        metadata=result.metadata,
    )


# -- Route handlers ---

try:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/assertions", tags=["assertions"])

    @router.post("", response_model=AssertionResponse)
    async def detect_assertion(request: AssertionRequest) -> AssertionResponse:
        """Detect assertion status for a single clinical entity.

        Classifies whether the entity is present, absent (negated),
        possible (uncertain), conditional, hypothetical, or family-related.

        Parameters
        ----------
        request : AssertionRequest
            Request with text and entity offsets.

        Returns
        -------
        AssertionResponse
            Assertion classification result.

        Raises
        ------
        HTTPException
            422 if entity offsets are out of bounds.
        """
        if request.entity_end > len(request.text):
            raise HTTPException(
                status_code=422,
                detail="entity_end exceeds text length",
            )

        detector = _get_detector()
        result = detector.detect(
            request.text,
            request.entity_start,
            request.entity_end,
        )
        return _result_to_response(result)

    @router.post("/batch", response_model=AssertionBatchResponse)
    async def detect_assertions_batch(
        request: AssertionBatchRequest,
    ) -> AssertionBatchResponse:
        """Detect assertions for multiple entities in the same text.

        Efficiently processes multiple entity spans against the same
        clinical text in a single request.

        Parameters
        ----------
        request : AssertionBatchRequest
            Request with text and entity spans.

        Returns
        -------
        AssertionBatchResponse
            Batch assertion results with summary.

        Raises
        ------
        HTTPException
            422 if any entity offset is out of bounds.
        """
        text_len = len(request.text)
        for i, entity in enumerate(request.entities):
            if entity.end > text_len:
                raise HTTPException(
                    status_code=422,
                    detail=f"Entity {i}: end offset {entity.end} exceeds text length {text_len}",
                )

        detector = _get_detector()
        start_time = time.perf_counter()

        results: list[AssertionResponse] = []
        summary: dict[str, int] = {}

        for entity in request.entities:
            result = detector.detect(request.text, entity.start, entity.end)
            response = _result_to_response(result)
            results.append(response)

            status_val = result.status.value
            summary[status_val] = summary.get(status_val, 0) + 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return AssertionBatchResponse(
            results=results,
            total=len(results),
            processing_time_ms=round(elapsed_ms, 2),
            summary=summary,
        )

    @router.get("/statuses", response_model=list[AssertionStatusInfo])
    async def list_assertion_statuses() -> list[AssertionStatusInfo]:
        """List all supported assertion status types.

        Returns
        -------
        list[AssertionStatusInfo]
            Catalogue of assertion statuses with descriptions.
        """
        descriptions = {
            AssertionStatus.PRESENT: "Condition is affirmed/present in the patient",
            AssertionStatus.ABSENT: "Condition is negated/denied/absent",
            AssertionStatus.POSSIBLE: "Condition is uncertain/suspected/differential",
            AssertionStatus.CONDITIONAL: "Condition depends on a future event or criterion",
            AssertionStatus.HYPOTHETICAL: "Condition is planned/future/pending",
            AssertionStatus.FAMILY: "Condition is attributed to a family member",
        }
        return [
            AssertionStatusInfo(status=s.value, description=d)
            for s, d in descriptions.items()
        ]

    @router.get("/stats")
    async def get_assertion_stats() -> dict[str, Any]:
        """Get detection statistics for the current detector instance.

        Returns
        -------
        dict[str, Any]
            Detection counts and trigger configuration info.
        """
        detector = _get_detector()
        return {
            "detection_stats": detector.stats,
            "trigger_count": detector.trigger_count,
            "version": "1.0.0",
        }

except ImportError:
    router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — assertion routes not registered")
