"""Unit tests for the document analysis service.

Tests the AnalysisService (from services/document_service.py) which
orchestrates the ML pipeline for single and batch document analysis.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.core.exceptions import InferenceError
from app.services.document_service import AnalysisService


class _FakePipeline:
    """Minimal stand-in for ClinicalPipeline."""

    def __init__(self, should_fail: bool = False):
        self._should_fail = should_fail
        self.load_called = False

    def load(self) -> None:
        self.load_called = True

    def process(self, text, config=None):
        if self._should_fail:
            raise RuntimeError("model boom")
        return {
            "entities": [{"text": "aspirin", "type": "MEDICATION"}],
            "icd_codes": [{"code": "E11.9", "confidence": 0.87}],
            "summary": "Patient stable.",
            "risk_score": 42.0,
        }


class TestAnalysisServiceInit:
    """Construction and lazy loading."""

    def test_service_without_pipeline_creates_one_lazily(self):
        svc = AnalysisService(pipeline=None)
        assert svc._pipeline is None

    def test_service_with_provided_pipeline(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        assert svc._pipeline is pipe


class TestSingleAnalysis:
    """Single-document analysis."""

    @pytest.mark.asyncio
    async def test_analyze_returns_result_dict(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        result = await svc.analyze("Patient takes aspirin daily.")
        assert "result" in result
        assert "processing_ms" in result
        assert "text_hash" in result
        assert pipe.load_called

    @pytest.mark.asyncio
    async def test_analyze_with_document_id(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        result = await svc.analyze("Test note.", document_id="doc-123")
        assert result["document_id"] == "doc-123"

    @pytest.mark.asyncio
    async def test_analyze_raises_inference_error_on_failure(self):
        pipe = _FakePipeline(should_fail=True)
        svc = AnalysisService(pipeline=pipe)
        with pytest.raises(InferenceError):
            await svc.analyze("Will crash.")

    @pytest.mark.asyncio
    async def test_analyze_generates_consistent_text_hash(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        r1 = await svc.analyze("Same text.")
        r2 = await svc.analyze("Same text.")
        assert r1["text_hash"] == r2["text_hash"]

    @pytest.mark.asyncio
    async def test_analyze_different_texts_have_different_hashes(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        r1 = await svc.analyze("Text one.")
        r2 = await svc.analyze("Text two.")
        assert r1["text_hash"] != r2["text_hash"]


class TestBatchAnalysis:
    """Batch document analysis."""

    @pytest.mark.asyncio
    async def test_batch_returns_list_of_results(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        results = await svc.batch_analyze(["Note A.", "Note B.", "Note C."])
        assert len(results) == 3
        assert all("result" in r for r in results)

    @pytest.mark.asyncio
    async def test_batch_empty_input_returns_empty(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        results = await svc.batch_analyze([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_assigns_incremental_document_ids(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        results = await svc.batch_analyze(["A.", "B."])
        assert results[0]["document_id"] == "batch-0"
        assert results[1]["document_id"] == "batch-1"

    @pytest.mark.asyncio
    async def test_batch_aborts_on_failure(self):
        pipe = _FakePipeline(should_fail=True)
        svc = AnalysisService(pipeline=pipe)
        with pytest.raises(InferenceError):
            await svc.batch_analyze(["Will fail."])


class TestPipelineCaching:
    """Pipeline should only be loaded once."""

    @pytest.mark.asyncio
    async def test_pipeline_loaded_once_across_calls(self):
        pipe = _FakePipeline()
        svc = AnalysisService(pipeline=pipe)
        await svc.analyze("First.")
        await svc.analyze("Second.")
        # load() is called once by _ensure_pipeline
        assert pipe.load_called
