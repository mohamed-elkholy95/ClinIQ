"""Unit tests for the batch processing route module.

Tests cover:
- ``POST /batch`` — job creation, Celery dispatch, ETA calculation
- ``GET /batch/{job_id}`` — status polling, progress computation, 404 handling
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.routes.batch import get_batch_status, submit_batch_job

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_payload(*, doc_count: int = 2, run_summary: bool = False) -> MagicMock:
    """Create a mock BatchRequest payload."""
    docs = []
    for i in range(doc_count):
        doc = MagicMock()
        doc.text = f"Patient {i} presents with acute chest pain."
        doc.document_id = f"doc-{i}"
        doc.metadata = None
        docs.append(doc)

    pipeline = MagicMock()
    pipeline.run_ner = True
    pipeline.run_icd = True
    pipeline.run_summary = run_summary
    pipeline.run_risk = True
    pipeline.model_dump.return_value = {
        "run_ner": True,
        "run_icd": True,
        "run_summary": run_summary,
        "run_risk": True,
        "ner_model": "rule-based",
        "icd_model": "sklearn-baseline",
        "icd_top_k": 5,
        "summary_model": "extractive",
    }

    payload = MagicMock()
    payload.documents = docs
    payload.pipeline = pipeline
    return payload


def _make_batch_job(
    *,
    job_id: uuid.UUID | None = None,
    status: str = "processing",
    total: int = 10,
    processed: int = 5,
    failed: int = 1,
) -> MagicMock:
    """Create a mock BatchJob ORM object."""
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.status = status
    job.total_documents = total
    job.processed_documents = processed
    job.failed_documents = failed
    job.pipeline_config = {"run_ner": True}
    job.created_at = datetime(2026, 3, 24, 10, 0, 0, tzinfo=UTC)
    job.started_at = datetime(2026, 3, 24, 10, 0, 1, tzinfo=UTC)
    job.completed_at = None
    job.error_message = None
    job.result_file = None
    return job


def _mock_scalar_result(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


# ---------------------------------------------------------------------------
# POST /batch (submit_batch_job)
# ---------------------------------------------------------------------------


class TestSubmitBatchJob:
    """Tests for the batch submission endpoint."""

    @pytest.mark.asyncio
    async def test_creates_job_and_dispatches_celery(self) -> None:
        """Valid payload creates a DB record and calls process_batch_task.delay()."""
        payload = _make_batch_payload(doc_count=3)
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_settings = MagicMock()

        mock_delay = MagicMock()

        with patch(
            "app.worker.process_batch_task",
            MagicMock(delay=mock_delay),
        ):
            result = await submit_batch_job(payload, mock_db, mock_settings)

        # DB record was added
        mock_db.add.assert_called_once()

        # Celery task was dispatched
        mock_delay.assert_called_once()
        call_args = mock_delay.call_args
        # First arg: job_id string, second: serialised docs, third: pipeline config
        assert isinstance(call_args[0][0], str)  # job_id
        assert len(call_args[0][1]) == 3  # 3 documents
        assert isinstance(call_args[0][2], dict)  # pipeline config

        # Response
        assert result.status == "pending"
        assert result.document_count == 3

    @pytest.mark.asyncio
    async def test_eta_scales_with_docs_and_stages(self) -> None:
        """ETA should increase with more documents and more active stages."""
        # 2 docs × 3 stages (ner, icd, risk) × 0.5s = 3s
        payload_3stages = _make_batch_payload(doc_count=2, run_summary=False)
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_settings = MagicMock()

        with patch("app.worker.process_batch_task", MagicMock(delay=MagicMock())):
            result_3 = await submit_batch_job(payload_3stages, mock_db, mock_settings)

        # 2 docs × 4 stages (+ summary) × 0.5s = 4s
        payload_4stages = _make_batch_payload(doc_count=2, run_summary=True)
        with patch("app.worker.process_batch_task", MagicMock(delay=MagicMock())):
            result_4 = await submit_batch_job(payload_4stages, mock_db, mock_settings)

        assert result_4.estimated_duration_seconds > result_3.estimated_duration_seconds

    @pytest.mark.asyncio
    async def test_serialised_docs_contain_required_fields(self) -> None:
        """Each serialised document should have text, document_id, and metadata."""
        payload = _make_batch_payload(doc_count=1)
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_settings = MagicMock()

        captured_docs = None
        original_delay = MagicMock()

        def capture_delay(job_id, docs, config):
            nonlocal captured_docs
            captured_docs = docs

        original_delay.side_effect = capture_delay

        with patch(
            "app.worker.process_batch_task",
            MagicMock(delay=original_delay),
        ):
            await submit_batch_job(payload, mock_db, mock_settings)

        assert captured_docs is not None
        assert len(captured_docs) == 1
        doc = captured_docs[0]
        assert "text" in doc
        assert "document_id" in doc
        assert "metadata" in doc


# ---------------------------------------------------------------------------
# GET /batch/{job_id} (get_batch_status)
# ---------------------------------------------------------------------------


class TestGetBatchStatus:
    """Tests for the batch status polling endpoint."""

    @pytest.mark.asyncio
    async def test_returns_status_for_existing_job(self) -> None:
        """Valid job_id returns BatchStatusResponse with correct progress."""
        job = _make_batch_job(total=10, processed=5)
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(job))

        result = await get_batch_status(job.id, mock_db)

        assert result.job_id == job.id
        assert result.status == "processing"
        assert result.progress == 0.5
        assert result.total_documents == 10
        assert result.processed_documents == 5
        assert result.failed_documents == 1

    @pytest.mark.asyncio
    async def test_progress_zero_when_no_documents_processed(self) -> None:
        """Job with zero processed documents reports 0.0 progress."""
        job = _make_batch_job(total=5, processed=0, failed=0)
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(job))

        result = await get_batch_status(job.id, mock_db)
        assert result.progress == 0.0

    @pytest.mark.asyncio
    async def test_progress_one_when_all_processed(self) -> None:
        """Completed job has progress=1.0."""
        job = _make_batch_job(
            status="completed", total=10, processed=10, failed=0
        )
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(job))

        result = await get_batch_status(job.id, mock_db)
        assert result.progress == 1.0

    @pytest.mark.asyncio
    async def test_handles_zero_total_documents(self) -> None:
        """Edge case: total_documents=0 should not cause division by zero."""
        job = _make_batch_job(total=0, processed=0, failed=0)
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(job))

        result = await get_batch_status(job.id, mock_db)
        assert result.progress == 0.0

    @pytest.mark.asyncio
    async def test_404_for_unknown_job(self) -> None:
        """Unknown job_id raises HTTP 404."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(None))

        with pytest.raises(HTTPException) as exc_info:
            await get_batch_status(uuid.uuid4(), mock_db)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
