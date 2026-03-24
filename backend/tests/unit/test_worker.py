"""Unit tests for the Celery worker task definitions.

Tests the process_batch_task and health_check tasks with mocked
pipeline and Celery context to verify task behaviour without
requiring a running broker.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pipeline_result(text: str) -> MagicMock:
    """Build a mock PipelineResult with to_dict-compatible sub-objects."""
    entity = MagicMock()
    entity.to_dict.return_value = {
        "text": "aspirin",
        "entity_type": "MEDICATION",
        "confidence": 0.9,
    }

    icd = MagicMock()
    icd.to_dict.return_value = {
        "code": "I21.0",
        "description": "STEMI",
        "confidence": 0.85,
    }

    summary = MagicMock()
    summary.to_dict.return_value = {"summary": text[:50], "sentence_count": 2}

    risk = MagicMock()
    risk.to_dict.return_value = {"overall_score": 0.6, "risk_level": "high"}

    result = MagicMock()
    result.entities = [entity]
    result.icd_predictions = [icd]
    result.summary = summary
    result.risk_assessment = risk
    return result


# ---------------------------------------------------------------------------
# health_check task
# ---------------------------------------------------------------------------


class TestHealthCheckTask:
    """Tests for the cliniq.health_check Celery task.

    Note: The worker module calls get_settings() at import time to configure
    the Celery broker.  We patch it before importing so the test never needs
    a live Redis instance.
    """

    def test_returns_healthy_status(self):
        """health_check should return a dict with status and worker name."""
        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                celery_broker_url="memory://",
                celery_result_backend="cache+memory://",
            )
            # Force re-import with patched settings
            import importlib
            import app.worker as worker_mod
            importlib.reload(worker_mod)

            result = worker_mod.health_check()
            assert result["status"] == "healthy"
            assert result["worker"] == "cliniq"


# ---------------------------------------------------------------------------
# process_batch_task
# ---------------------------------------------------------------------------


class TestProcessBatchTask:
    """Tests for the cliniq.process_batch Celery task.

    The task defers ClinicalPipeline import to call-time, so we patch it
    inside ``app.ml.pipeline`` and call the underlying function directly
    (bypassing the Celery decorator).
    """

    @patch("app.ml.pipeline.ClinicalPipeline")
    def test_processes_documents(self, MockPipeline: MagicMock):
        """Each document should be processed and results collected."""
        mock_instance = MockPipeline.return_value
        mock_instance.process.side_effect = lambda text, cfg: _make_mock_pipeline_result(text)

        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                celery_broker_url="memory://",
                celery_result_backend="cache+memory://",
            )
            import importlib
            import app.worker as worker_mod
            importlib.reload(worker_mod)

            mock_self = MagicMock()
            documents = [
                {"document_id": "doc1", "text": "Patient has chest pain."},
                {"document_id": "doc2", "text": "Routine follow-up visit."},
            ]
            config = {"enable_ner": True, "enable_icd": True}

            result = worker_mod.process_batch_task(mock_self, "job-123", documents, config)

            assert result["job_id"] == "job-123"
            assert result["total"] == 2
            assert len(result["results"]) == 2
            assert result["results"][0]["document_id"] == "doc1"
            assert result["results"][0]["status"] == "completed"

    @patch("app.ml.pipeline.ClinicalPipeline")
    def test_handles_individual_document_failure(self, MockPipeline: MagicMock):
        """A failing document should produce an error result, not crash the batch."""
        mock_instance = MockPipeline.return_value
        mock_instance.process.side_effect = [
            _make_mock_pipeline_result("ok"),
            RuntimeError("Model inference failed"),
        ]

        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                celery_broker_url="memory://",
                celery_result_backend="cache+memory://",
            )
            import importlib
            import app.worker as worker_mod
            importlib.reload(worker_mod)

            mock_self = MagicMock()
            documents = [
                {"document_id": "doc1", "text": "Good note."},
                {"document_id": "doc2", "text": "Bad note."},
            ]

            result = worker_mod.process_batch_task(mock_self, "job-456", documents, {})

            assert result["results"][0]["status"] == "completed"
            assert result["results"][1]["status"] == "failed"
            assert "Model inference failed" in result["results"][1]["error"]

    @patch("app.ml.pipeline.ClinicalPipeline")
    def test_updates_progress(self, MockPipeline: MagicMock):
        """Task should call update_state with progress after each document."""
        mock_instance = MockPipeline.return_value
        mock_instance.process.return_value = _make_mock_pipeline_result("text")

        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                celery_broker_url="memory://",
                celery_result_backend="cache+memory://",
            )
            import importlib
            import app.worker as worker_mod
            importlib.reload(worker_mod)

            mock_self = MagicMock()
            documents = [
                {"document_id": f"doc{i}", "text": f"Note {i}."} for i in range(3)
            ]

            worker_mod.process_batch_task(mock_self, "job-789", documents, {})

            assert mock_self.update_state.call_count == 3
            last_call = mock_self.update_state.call_args
            assert last_call.kwargs["meta"]["current"] == 3
            assert last_call.kwargs["meta"]["total"] == 3
