"""Tests for BatchRequest model_validator — at_least_one_stage_enabled (lines 111-114)."""

import pytest
from pydantic import ValidationError

from app.api.schemas.batch import BatchRequest, BatchPipelineConfig


class TestBatchRequestValidator:
    """Cover the at_least_one_stage_enabled validator."""

    def test_all_stages_disabled_raises(self) -> None:
        """Disabling every pipeline stage should raise a ValidationError."""
        with pytest.raises(ValidationError, match="At least one pipeline stage"):
            BatchRequest(
                documents=[{"text": "Some clinical note.", "document_id": "d1"}],
                pipeline=BatchPipelineConfig(
                    run_ner=False,
                    run_icd=False,
                    run_summary=False,
                    run_risk=False,
                ),
            )

    def test_one_stage_enabled(self) -> None:
        """Enabling just one stage should pass validation."""
        req = BatchRequest(
            documents=[{"text": "Some clinical note.", "document_id": "d1"}],
            pipeline=BatchPipelineConfig(
                run_ner=True,
                run_icd=False,
                run_summary=False,
                run_risk=False,
            ),
        )
        assert req.pipeline.run_ner is True

    def test_default_stages(self) -> None:
        """Default BatchPipelineConfig has NER, ICD and risk enabled, summary disabled."""
        req = BatchRequest(
            documents=[{"text": "Some clinical note.", "document_id": "d1"}],
        )
        assert req.pipeline.run_ner is True
        assert req.pipeline.run_icd is True
        assert req.pipeline.run_risk is True
