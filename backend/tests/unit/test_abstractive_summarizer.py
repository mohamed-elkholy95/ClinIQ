"""Unit tests for AbstractiveSummarizer.

Tests the HuggingFace BART/T5 wrapper with mocked transformers pipeline,
including loading, single-chunk and multi-chunk summarization, hierarchical
two-pass summarization, chunking logic, and error propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.summarization.model import (
    AbstractiveSummarizer,
    SummarizationResult,
)


class TestAbstractiveSummarizerLoad:
    """Tests for model loading."""

    def test_load_success(self) -> None:
        mock_tokenizer = MagicMock()
        mock_pipeline = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.pipeline.return_value = mock_pipeline

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            model = AbstractiveSummarizer(model_name="facebook/bart-large-cnn")
            model.load()
            assert model.is_loaded

    def test_load_from_path(self) -> None:
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            model = AbstractiveSummarizer(model_path="/custom/model")
            model.load()
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_with("/custom/model")

    def test_load_failure_raises(self) -> None:
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("bad model")
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            model = AbstractiveSummarizer()
            with pytest.raises(ModelLoadError):
                model.load()


class TestAbstractiveSummarizerInference:
    """Tests for summarization inference."""

    def _make_model(self) -> AbstractiveSummarizer:
        model = AbstractiveSummarizer()
        model._is_loaded = True
        model._tokenizer = MagicMock()
        model._pipeline = MagicMock()
        return model

    def test_single_chunk_summarization(self) -> None:
        model = self._make_model()
        # Simulate short text (fits in one chunk)
        model._tokenizer.encode.return_value = list(range(50))  # 50 tokens < 900
        model._pipeline.return_value = [{"summary_text": "Patient stable."}]

        result = model.summarize("Patient presents with stable vitals.", detail_level="brief")

        assert isinstance(result, SummarizationResult)
        assert result.summary == "Patient stable."
        assert result.detail_level == "brief"
        assert result.processing_time_ms > 0
        assert result.metadata.get("chunk_count") == 1

    def test_multi_chunk_summarization(self) -> None:
        model = self._make_model()
        # Simulate long text that needs 2 chunks
        model._tokenizer.encode.side_effect = [
            list(range(1500)),  # First call in _chunk_text: too long
            list(range(50)),    # combined summary token count check
        ]
        model._tokenizer.decode.side_effect = ["chunk1 text", "chunk2 text"]
        model._pipeline.return_value = [{"summary_text": "Summarized chunk."}]

        result = model.summarize("Very long clinical note " * 200, detail_level="standard")

        assert isinstance(result, SummarizationResult)
        assert result.metadata.get("chunk_count") == 2

    def test_inference_error_propagation(self) -> None:
        model = self._make_model()
        model._tokenizer.encode.side_effect = RuntimeError("tokenizer broke")

        with pytest.raises(InferenceError):
            model.summarize("Some text")

    def test_detail_level_detailed(self) -> None:
        model = self._make_model()
        model._tokenizer.encode.return_value = list(range(50))
        model._pipeline.return_value = [{"summary_text": "Detailed summary."}]

        result = model.summarize("Patient note.", detail_level="detailed")
        assert result.detail_level == "detailed"


class TestSummarizationResultDataclass:
    """Tests for the SummarizationResult dataclass."""

    def test_to_dict(self) -> None:
        result = SummarizationResult(
            summary="Patient stable",
            key_findings=["stable vitals", "discharged"],
            detail_level="standard",
            processing_time_ms=42.5,
            model_name="extractive-textrank",
            model_version="1.0.0",
            sentence_count_original=20,
            sentence_count_summary=5,
            metadata={"method": "textrank"},
        )
        d = result.to_dict()
        assert d["summary"] == "Patient stable"
        assert d["sentence_count_original"] == 20
        assert d["metadata"]["method"] == "textrank"
        assert len(d["key_findings"]) == 2

    def test_defaults(self) -> None:
        result = SummarizationResult(
            summary="text", key_findings=[], detail_level="brief",
            processing_time_ms=1.0, model_name="t", model_version="1"
        )
        assert result.sentence_count_original == 0
        assert result.metadata == {}
