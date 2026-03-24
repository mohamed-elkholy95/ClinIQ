"""Batch processing request and response schemas."""

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

MAX_TEXT_LENGTH = 100_000
MAX_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class BatchDocument(BaseModel):
    """A single document within a batch request."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text for this document (up to 100 000 characters)",
        ),
    ]
    document_id: str | None = Field(
        default=None,
        max_length=36,
        description="Optional client-supplied identifier for this document (returned in results)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary key-value metadata attached to this document (returned verbatim in results)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Patient presents with acute chest pain and shortness of breath.",
                "document_id": "enc-2026-001",
                "metadata": {"ward": "cardiology", "admitted": "2026-03-24"},
            }
        }
    }


class BatchPipelineConfig(BaseModel):
    """Pipeline stages to run for every document in the batch."""

    run_ner: bool = Field(default=True, description="Run NER on each document")
    run_icd: bool = Field(default=True, description="Run ICD-10 prediction on each document")
    run_summary: bool = Field(default=False, description="Run summarization on each document (increases latency)")
    run_risk: bool = Field(default=True, description="Run risk scoring on each document")
    ner_model: str = Field(default="rule-based", description="NER backend to use for the whole batch")
    icd_model: str = Field(default="sklearn-baseline", description="ICD classifier backend for the whole batch")
    icd_top_k: int = Field(default=5, ge=1, le=50, description="Maximum ICD codes per document")
    summary_model: str = Field(
        default="extractive",
        description="Summarizer backend: 'extractive', 'section-based', 'abstractive', or 'hybrid'",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "run_ner": True,
                "run_icd": True,
                "run_summary": False,
                "run_risk": True,
                "ner_model": "rule-based",
                "icd_model": "sklearn-baseline",
                "icd_top_k": 5,
                "summary_model": "extractive",
            }
        }
    }


class BatchRequest(BaseModel):
    """Request body for submitting an asynchronous batch processing job."""

    documents: Annotated[
        list[BatchDocument],
        Field(
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description=f"List of 1 to {MAX_BATCH_SIZE} clinical documents to process",
        ),
    ]
    pipeline: BatchPipelineConfig = Field(
        default_factory=BatchPipelineConfig,
        description="Pipeline stages and model settings applied uniformly to all documents",
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Job scheduling priority. 'high' jobs are queued ahead of 'normal' and 'low' jobs.",
    )
    notify_webhook: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "HTTPS URL to POST a completion notification to when the job finishes. "
            "The payload will be a BatchStatusResponse JSON object."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_stage_enabled(self) -> "BatchRequest":
        """Ensure the batch request enables at least one ML pipeline stage."""
        p = self.pipeline
        if not any([p.run_ner, p.run_icd, p.run_summary, p.run_risk]):
            raise ValueError("At least one pipeline stage must be enabled in the batch config")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "documents": [
                    {
                        "text": "72-year-old male with STEMI and diabetes.",
                        "document_id": "enc-001",
                        "metadata": None,
                    },
                    {
                        "text": "58-year-old female with COPD exacerbation.",
                        "document_id": "enc-002",
                        "metadata": None,
                    },
                ],
                "pipeline": {
                    "run_ner": True,
                    "run_icd": True,
                    "run_summary": False,
                    "run_risk": True,
                    "icd_top_k": 5,
                },
                "priority": "normal",
                "notify_webhook": None,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response atoms
# ---------------------------------------------------------------------------


class BatchDocumentResult(BaseModel):
    """Analysis result for a single document within a completed batch job."""

    document_id: str | None = Field(
        default=None,
        description="Client-supplied document identifier (echoed from the request)",
    )
    index: int = Field(ge=0, description="Zero-based position of this document in the original request list")
    status: Literal["success", "failed"] = Field(description="Per-document processing outcome")
    error: str | None = Field(
        default=None,
        description="Error message when status is 'failed'",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Analysis output as a JSON object (structure mirrors AnalysisResponse)",
    )
    processing_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Per-document inference latency in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "enc-001",
                "index": 0,
                "status": "success",
                "error": None,
                "result": {"entities": [], "icd_codes": [], "risk_score": {"score": 0.72, "category": "high"}},
                "processing_time_ms": 95.4,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


class BatchSubmitResponse(BaseModel):
    """Returned immediately after a batch job is accepted."""

    job_id: UUID = Field(description="Server-assigned unique job identifier (UUID v4)")
    status: Literal["pending"] = Field(default="pending", description="Always 'pending' at submission time")
    document_count: int = Field(ge=1, description="Number of documents in this job")
    estimated_duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Rough ETA for job completion in seconds, if available",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the job was accepted",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "pending",
                "document_count": 2,
                "estimated_duration_seconds": 12,
                "created_at": "2026-03-24T10:00:00Z",
            }
        }
    }


class BatchStatusResponse(BaseModel):
    """Polling response for batch job status; also sent as webhook payload on completion."""

    job_id: UUID = Field(description="Server-assigned unique job identifier (UUID v4)")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        description="Current lifecycle state of the job"
    )
    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of documents processed so far in [0, 1]",
    )
    total_documents: int = Field(ge=0, description="Total number of documents submitted in this job")
    processed_documents: int = Field(ge=0, description="Number of documents that have finished processing")
    failed_documents: int = Field(ge=0, description="Number of documents that failed to process")
    created_at: datetime = Field(description="UTC timestamp when the job was accepted")
    started_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the worker started processing (null if still pending)",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when the job reached a terminal state (null if still in progress)",
    )
    error_message: str | None = Field(
        default=None,
        description="Top-level error message when status is 'failed'",
    )
    result_file: str | None = Field(
        default=None,
        description="Pre-signed object-storage URL for the full results file (available when status='completed')",
    )
    results: list[BatchDocumentResult] | None = Field(
        default=None,
        description=(
            "Inline per-document results (only populated on the final GET /batch/{job_id} "
            "call for small jobs; large jobs use result_file instead)"
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "processing",
                "progress": 0.5,
                "total_documents": 2,
                "processed_documents": 1,
                "failed_documents": 0,
                "created_at": "2026-03-24T10:00:00Z",
                "started_at": "2026-03-24T10:00:01Z",
                "completed_at": None,
                "error_message": None,
                "result_file": None,
                "results": None,
            }
        }
    }
