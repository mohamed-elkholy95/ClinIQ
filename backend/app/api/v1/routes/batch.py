"""Batch processing endpoint routes."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from celery.result import AsyncResult
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.v1.deps import CurrentUser, DBSession, PipelineDep
from app.api.v1.schemas import (
    BatchJobResponse,
    BatchRequest,
    BatchStatusResponse,
)
from app.db.models import BatchJob
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch-processing"])


@router.post(
    "",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create batch processing job",
    description="Submit multiple documents for batch processing.",
)
async def create_batch_job(
    request: BatchRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> BatchJobResponse:
    """Create a new batch processing job."""
    if len(request.documents) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 documents per batch",
        )

    # Create job record
    job_id = str(uuid.uuid4())
    batch_job = BatchJob(
        id=job_id,
        user_id=current_user.id,
        status="pending",
        total_documents=len(request.documents),
        pipeline_config={
            "enable_ner": request.enable_ner,
            "enable_icd": request.enable_icd,
            "enable_summarization": request.enable_summarization,
            "enable_risk": request.enable_risk,
        },
    )

    db.add(batch_job)
    await db.commit()

    # Queue background task
    # In production, this would use Celery
    # _process_batch.delay(job_id, request.documents, request.dict())

    return BatchJobResponse(
        job_id=job_id,
        status="pending",
        total_documents=len(request.documents),
        message="Batch job created and queued for processing",
    )


@router.get(
    "/{job_id}",
    response_model=BatchStatusResponse,
    summary="Get batch job status",
    description="Check the status of a batch processing job.",
)
async def get_batch_status(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> BatchStatusResponse:
    """Get status of a batch job."""
    result = await db.execute(
        select(BatchJob).where(
            BatchJob.id == job_id,
            BatchJob.user_id == current_user.id,
        )
    )
    batch_job = result.scalar_one_or_none()

    if not batch_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )

    return BatchStatusResponse(
        job_id=str(batch_job.id),
        status=batch_job.status,
        total_documents=batch_job.total_documents,
        processed_documents=batch_job.processed_documents,
        failed_documents=batch_job.failed_documents,
        created_at=batch_job.created_at,
        started_at=batch_job.started_at,
        completed_at=batch_job.completed_at,
        result_url=f"/api/v1/batch/{job_id}/results" if batch_job.status == "completed" else None,
        error_message=batch_job.error_message,
    )


@router.get(
    "/{job_id}/results",
    summary="Get batch job results",
    description="Download results of a completed batch job.",
)
async def get_batch_results(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Get results of completed batch job."""
    result = await db.execute(
        select(BatchJob).where(
            BatchJob.id == job_id,
            BatchJob.user_id == current_user.id,
        )
    )
    batch_job = result.scalar_one_or_none()

    if not batch_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )

    if batch_job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch job is {batch_job.status}, not completed",
        )

    # In production, this would return a signed URL to MinIO/S3
    return {
        "job_id": job_id,
        "result_file": batch_job.result_file,
        "download_url": f"/download/{batch_job.result_file}",
        "expires_at": None,
    }


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel batch job",
    description="Cancel a pending or running batch job.",
)
async def cancel_batch_job(
    job_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """Cancel a batch job."""
    result = await db.execute(
        select(BatchJob).where(
            BatchJob.id == job_id,
            BatchJob.user_id == current_user.id,
        )
    )
    batch_job = result.scalar_one_or_none()

    if not batch_job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found",
        )

    if batch_job.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {batch_job.status}",
        )

    batch_job.status = "cancelled"
    await db.commit()


@router.get(
    "",
    summary="List batch jobs",
    description="List batch jobs for the current user.",
)
async def list_batch_jobs(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = 20,
    offset: int = 0,
    status_filter: str | None = None,
) -> dict:
    """List batch jobs for current user."""
    query = select(BatchJob).where(BatchJob.user_id == current_user.id)

    if status_filter:
        query = query.where(BatchJob.status == status_filter)

    query = query.order_by(BatchJob.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    jobs = result.scalars().all()

    # Get total count
    from sqlalchemy import func

    count_query = select(func.count(BatchJob.id)).where(
        BatchJob.user_id == current_user.id
    )
    if status_filter:
        count_query = count_query.where(BatchJob.status == status_filter)

    count_result = await db.execute(count_query)
    total = count_result.scalar()

    return {
        "jobs": [
            {
                "job_id": str(job.id),
                "status": job.status,
                "total_documents": job.total_documents,
                "processed_documents": job.processed_documents,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            for job in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
