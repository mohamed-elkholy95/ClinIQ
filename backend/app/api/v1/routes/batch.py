"""Batch processing endpoints.

Allows callers to submit up to 100 clinical documents as a single asynchronous
batch job.  Jobs are persisted in the database and dispatched to Celery workers
for background inference.  The polling endpoint returns live progress.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.batch import BatchRequest, BatchStatusResponse, BatchSubmitResponse
from app.core.config import Settings, get_settings
from app.db.models import BatchJob
from app.db.session import get_db_session

router = APIRouter(tags=["batch"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch processing job",
    description=(
        "Submit 1–100 clinical documents for asynchronous analysis. "
        "The pipeline stages to run (NER, ICD, summarisation, risk scoring) are "
        "configured globally for the whole batch via the `pipeline` object. "
        "Returns a `job_id` that can be used to poll `GET /batch/{job_id}` for progress. "
        "An optional `notify_webhook` URL receives a POST when the job completes."
    ),
    responses={
        202: {"description": "Job accepted and queued"},
        422: {"description": "Input validation error"},
    },
)
async def submit_batch_job(
    payload: BatchRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BatchSubmitResponse:
    """Persist the batch job record and enqueue background processing."""
    job_id = uuid.uuid4()
    doc_count = len(payload.documents)

    pipeline_config = payload.pipeline.model_dump()

    batch_job = BatchJob(
        id=job_id,
        # user_id will be set once auth middleware is wired; using a sentinel UUID for now.
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        status="pending",
        total_documents=doc_count,
        processed_documents=0,
        failed_documents=0,
        pipeline_config=pipeline_config,
    )
    db.add(batch_job)
    # Session is committed by the get_db_session dependency after the response is sent.

    # Dispatch the batch to Celery for background processing.
    # Each document is serialised to a dict for the JSON task payload.
    from app.worker import process_batch_task

    serialised_docs = [
        {
            "text": doc.text,
            "document_id": doc.document_id,
            "metadata": doc.metadata,
        }
        for doc in payload.documents
    ]
    process_batch_task.delay(str(job_id), serialised_docs, pipeline_config)

    # Rough ETA: assume ~0.5 s per document per active stage.
    active_stages = sum(
        [
            payload.pipeline.run_ner,
            payload.pipeline.run_icd,
            payload.pipeline.run_summary,
            payload.pipeline.run_risk,
        ]
    )
    estimated_seconds = max(1, int(doc_count * active_stages * 0.5))

    return BatchSubmitResponse(
        job_id=job_id,
        status="pending",
        document_count=doc_count,
        estimated_duration_seconds=estimated_seconds,
    )


@router.get(
    "/batch/{job_id}",
    response_model=BatchStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll batch job status",
    description=(
        "Retrieve the current status and progress of a batch job. "
        "When `status` is 'completed', a `result_file` pre-signed URL is included "
        "for downloading the full results. "
        "For small batches (≤10 documents), inline `results` are also included "
        "on the completion response."
    ),
    responses={
        200: {"description": "Job status returned"},
        404: {"description": "Job not found"},
    },
)
async def get_batch_status(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> BatchStatusResponse:
    """Return the current status and progress of a batch job."""
    result = await db.execute(select(BatchJob).where(BatchJob.id == job_id))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )

    progress = (
        round(job.processed_documents / job.total_documents, 4) if job.total_documents else 0.0
    )

    return BatchStatusResponse(
        job_id=job.id,
        status=job.status,  # type: ignore[arg-type]
        progress=progress,
        total_documents=job.total_documents,
        processed_documents=job.processed_documents,
        failed_documents=job.failed_documents,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        result_file=job.result_file,
        results=None,  # Inline results populated once result persistence is implemented.
    )
