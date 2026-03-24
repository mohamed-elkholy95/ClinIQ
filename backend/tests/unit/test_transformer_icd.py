"""Unit tests for TransformerICDClassifier and HierarchicalICDClassifier.

Covers model loading, single-text prediction, long-document sliding-window,
batch prediction, and the hierarchical two-stage dispatch strategy.
All transformer/torch dependencies are mocked to avoid GPU/weight downloads.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import sys

import numpy as np
import pytest

# Provide a minimal torch mock so model.py can import torch inside methods
_torch_mock = MagicMock()
# Simulate torch.tensor returning real numpy-backed objects for sigmoid
def _fake_tensor(data):
    """Create a mock tensor that supports .cpu().numpy() and basic math."""
    arr = np.array(data, dtype=np.float32)
    t = MagicMock()
    t.cpu.return_value = t
    t.numpy.return_value = arr
    t.__getitem__ = lambda self, idx: _fake_tensor(arr[idx])
    t.tolist.return_value = arr.tolist()
    t.to.return_value = t
    return t

_torch_mock.tensor = _fake_tensor
_no_grad_ctx = MagicMock()
_no_grad_ctx.__enter__ = MagicMock(return_value=None)
_no_grad_ctx.__exit__ = MagicMock(return_value=False)
_torch_mock.no_grad.return_value = _no_grad_ctx
_torch_mock.sigmoid = lambda x: _fake_tensor(1.0 / (1.0 + np.exp(-x.cpu().numpy())))

from app.ml.icd.model import (
    HierarchicalICDClassifier,
    ICDCodePrediction,
    ICDPredictionResult,
    TransformerICDClassifier,
    get_chapter_for_code,
)
from app.core.exceptions import InferenceError, ModelLoadError


# ---------------------------------------------------------------------------
# TransformerICDClassifier — load
# ---------------------------------------------------------------------------


class TestTransformerICDLoad:
    """Test model loading behaviour."""

    def test_load_success(self) -> None:
        """Model loads tokenizer + model and marks itself loaded."""
        clf = TransformerICDClassifier(model_name="test-model", model_path="/tmp/model")

        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "E11.9", 1: "I10"}

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": MagicMock()}):
            clf.load()

        assert clf._is_loaded is True
        assert clf.label_map == {0: "E11.9", 1: "I10"}

    def test_load_failure_raises_model_load_error(self) -> None:
        """When transformers raises, we wrap in ModelLoadError."""
        clf = TransformerICDClassifier(model_name="bad-model")

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("Not found")

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": MagicMock()}):
            with pytest.raises(ModelLoadError):
                clf.load()

    def test_load_no_id2label(self) -> None:
        """Model without id2label still loads — label_map stays empty."""
        clf = TransformerICDClassifier(model_name="test")

        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # no id2label attr

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": MagicMock()}):
            clf.load()

        assert clf._is_loaded is True
        assert clf.label_map == {}


# ---------------------------------------------------------------------------
# TransformerICDClassifier — predict
# ---------------------------------------------------------------------------


class TestTransformerICDPredict:
    """Test single-text prediction."""

    @pytest.fixture(autouse=True)
    def _patch_torch(self):
        with patch.dict("sys.modules", {"torch": _torch_mock}):
            yield

    @pytest.fixture()
    def loaded_clf(self) -> TransformerICDClassifier:
        """Return a TransformerICDClassifier with mocked internals."""
        clf = TransformerICDClassifier(model_name="test", max_length=512)
        clf._is_loaded = True
        clf.label_map = {0: "E11.9", 1: "I10", 2: "J44.1"}
        clf.code_descriptions = {"E11.9": "Type 2 DM", "I10": "Hypertension"}

        # Mock tokenizer
        clf.tokenizer = MagicMock()
        clf.tokenizer.return_value = {
            "input_ids": MagicMock(**{"to.return_value": MagicMock()}),
            "attention_mask": MagicMock(**{"to.return_value": MagicMock()}),
        }

        # Mock model producing logits → sigmoid → probabilities
        logits = _fake_tensor([[2.0, 0.5, -1.0]])  # sigmoid ≈ [0.88, 0.62, 0.27]
        mock_output = MagicMock()
        mock_output.logits = logits
        clf.model = MagicMock(return_value=mock_output)

        return clf

    def test_predict_returns_result(self, loaded_clf: TransformerICDClassifier) -> None:
        result = loaded_clf.predict("Patient has diabetes", top_k=3)
        assert isinstance(result, ICDPredictionResult)
        assert result.model_name == "test"
        assert result.processing_time_ms > 0

    def test_predict_top_k_limits(self, loaded_clf: TransformerICDClassifier) -> None:
        result = loaded_clf.predict("Patient has diabetes", top_k=1)
        assert len(result.predictions) <= 1

    def test_predict_filters_low_confidence(self, loaded_clf: TransformerICDClassifier) -> None:
        """Predictions below 0.1 threshold are excluded."""
        import torch

        # All logits very negative → sigmoid < 0.1
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[-5.0, -5.0, -5.0]])
        loaded_clf.model = MagicMock(return_value=mock_output)

        result = loaded_clf.predict("nothing relevant", top_k=10)
        assert len(result.predictions) == 0

    def test_predict_chapter_populated(self, loaded_clf: TransformerICDClassifier) -> None:
        result = loaded_clf.predict("diabetes", top_k=3)
        for pred in result.predictions:
            # get_chapter_for_code should produce a chapter string
            assert pred.chapter is not None or pred.code.startswith("UNKNOWN")

    def test_predict_error_raises_inference_error(self, loaded_clf: TransformerICDClassifier) -> None:
        loaded_clf.model.side_effect = RuntimeError("CUDA OOM")
        with pytest.raises(InferenceError):
            loaded_clf.predict("test text")

    def test_predict_unknown_label_index(self, loaded_clf: TransformerICDClassifier) -> None:
        """When label_map is missing an index, code is UNKNOWN_<idx>."""
        loaded_clf.label_map = {}  # empty label map
        result = loaded_clf.predict("diabetes", top_k=3)
        for pred in result.predictions:
            assert pred.code.startswith("UNKNOWN_")


# ---------------------------------------------------------------------------
# TransformerICDClassifier — long document sliding window
# ---------------------------------------------------------------------------


class TestTransformerICDLongDoc:
    """Test sliding-window handling for documents exceeding max_length."""

    @pytest.fixture(autouse=True)
    def _patch_torch(self):
        with patch.dict("sys.modules", {"torch": _torch_mock}):
            yield

    @pytest.fixture()
    def long_clf(self) -> TransformerICDClassifier:
        clf = TransformerICDClassifier(model_name="test", max_length=120)
        clf._is_loaded = True
        clf.label_map = {0: "E11.9", 1: "I10"}
        clf.code_descriptions = {}

        clf.tokenizer = MagicMock()
        clf.tokenizer.return_value = {
            "input_ids": MagicMock(**{"to.return_value": MagicMock()}),
            "attention_mask": MagicMock(**{"to.return_value": MagicMock()}),
        }

        mock_output = MagicMock()
        mock_output.logits = _fake_tensor([[1.0, 0.5]])
        clf.model = MagicMock(return_value=mock_output)

        return clf

    def test_long_doc_triggers_sliding_window(self, long_clf: TransformerICDClassifier) -> None:
        """Text longer than max_length words triggers _predict_long_document."""
        long_text = " ".join(["word"] * 200)  # 200 words > max_length=120
        result = long_clf.predict(long_text, top_k=2)
        assert isinstance(result, ICDPredictionResult)
        # Model should be called multiple times (once per window)
        assert long_clf.model.call_count > 1

    def test_sliding_window_aggregates_max(self, long_clf: TransformerICDClassifier) -> None:
        """Max-pooling aggregation across windows."""
        call_count = [0]
        def varying_output(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return MagicMock(logits=_fake_tensor([[0.1, 3.0]]))
            return MagicMock(logits=_fake_tensor([[3.0, 0.1]]))

        long_clf.model = MagicMock(side_effect=varying_output)
        long_text = " ".join(["word"] * 200)
        result = long_clf.predict(long_text, top_k=2)
        # Both codes should appear with high confidence (max across windows)
        assert len(result.predictions) == 2


# ---------------------------------------------------------------------------
# TransformerICDClassifier — batch predict
# ---------------------------------------------------------------------------


class TestTransformerICDBatch:
    """Test batch prediction."""

    @pytest.fixture(autouse=True)
    def _patch_torch(self):
        with patch.dict("sys.modules", {"torch": _torch_mock}):
            yield

    @pytest.fixture()
    def batch_clf(self) -> TransformerICDClassifier:
        clf = TransformerICDClassifier(model_name="batch-test", max_length=512)
        clf._is_loaded = True
        clf.label_map = {0: "E11.9", 1: "I10"}
        clf.code_descriptions = {"E11.9": "DM", "I10": "HTN"}

        clf.tokenizer = MagicMock()
        clf.tokenizer.return_value = {
            "input_ids": MagicMock(**{"to.return_value": MagicMock()}),
            "attention_mask": MagicMock(**{"to.return_value": MagicMock()}),
        }

        mock_output = MagicMock()
        # batch of 3 documents
        mock_output.logits = _fake_tensor([[2.0, 0.5], [0.5, 2.0], [1.0, 1.0]])
        clf.model = MagicMock(return_value=mock_output)

        return clf

    def test_batch_returns_correct_count(self, batch_clf: TransformerICDClassifier) -> None:
        texts = ["diabetes", "hypertension", "both conditions"]
        results = batch_clf.predict_batch(texts, top_k=2)
        assert len(results) == 3

    def test_batch_timing_averaged(self, batch_clf: TransformerICDClassifier) -> None:
        texts = ["doc1", "doc2", "doc3"]
        results = batch_clf.predict_batch(texts, top_k=2)
        # All results should have same average time
        times = [r.processing_time_ms for r in results]
        assert all(t == times[0] for t in times)

    def test_batch_error_raises_inference_error(self, batch_clf: TransformerICDClassifier) -> None:
        batch_clf.model.side_effect = RuntimeError("OOM")
        with pytest.raises(InferenceError):
            batch_clf.predict_batch(["text1", "text2"])


# ---------------------------------------------------------------------------
# HierarchicalICDClassifier
# ---------------------------------------------------------------------------


class TestHierarchicalICDClassifier:
    """Test the two-stage chapter→code classifier."""

    @pytest.fixture()
    def hier_clf(self) -> HierarchicalICDClassifier:
        chapter_clf = MagicMock()
        chapter_clf.predict.return_value = ICDPredictionResult(
            predictions=[
                ICDCodePrediction(code="E", description="Endocrine", confidence=0.9, chapter="E"),
                ICDCodePrediction(code="I", description="Circulatory", confidence=0.7, chapter="I"),
            ],
            processing_time_ms=10.0,
            model_name="chapter",
            model_version="1.0",
        )

        e_clf = MagicMock()
        e_clf.predict.return_value = ICDPredictionResult(
            predictions=[
                ICDCodePrediction(code="E11.9", description="DM", confidence=0.85, chapter="E"),
            ],
            processing_time_ms=5.0,
            model_name="e-codes",
            model_version="1.0",
        )

        i_clf = MagicMock()
        i_clf.predict.return_value = ICDPredictionResult(
            predictions=[
                ICDCodePrediction(code="I10", description="HTN", confidence=0.8, chapter="I"),
            ],
            processing_time_ms=5.0,
            model_name="i-codes",
            model_version="1.0",
        )

        return HierarchicalICDClassifier(
            chapter_classifier=chapter_clf,
            code_classifiers={"E": e_clf, "I": i_clf},
        )

    def test_load_delegates_to_all(self, hier_clf: HierarchicalICDClassifier) -> None:
        hier_clf.load()
        hier_clf.chapter_classifier.load.assert_called_once()
        for clf in hier_clf.code_classifiers.values():
            clf.load.assert_called_once()
        assert hier_clf._is_loaded is True

    def test_predict_combines_chapters(self, hier_clf: HierarchicalICDClassifier) -> None:
        hier_clf._is_loaded = True
        result = hier_clf.predict("diabetes and hypertension", top_k=10)
        assert isinstance(result, ICDPredictionResult)
        codes = [p.code for p in result.predictions]
        assert "E11.9" in codes
        assert "I10" in codes

    def test_predict_skips_missing_chapter_classifier(self) -> None:
        """If a chapter has no code classifier, skip it gracefully."""
        chapter_clf = MagicMock()
        chapter_clf.predict.return_value = ICDPredictionResult(
            predictions=[
                ICDCodePrediction(code="Z", description="Health services", confidence=0.9, chapter="Z"),
            ],
            processing_time_ms=5.0,
            model_name="chapter",
            model_version="1.0",
        )

        clf = HierarchicalICDClassifier(
            chapter_classifier=chapter_clf,
            code_classifiers={},  # No code classifiers
        )
        clf._is_loaded = True
        result = clf.predict("routine checkup", top_k=5)
        assert isinstance(result, ICDPredictionResult)
        assert len(result.predictions) == 0

    def test_predict_sorts_by_confidence(self, hier_clf: HierarchicalICDClassifier) -> None:
        hier_clf._is_loaded = True
        result = hier_clf.predict("diabetes and htn", top_k=10)
        confidences = [p.confidence for p in result.predictions]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_batch_delegates(self, hier_clf: HierarchicalICDClassifier) -> None:
        hier_clf._is_loaded = True
        results = hier_clf.predict_batch(["text1", "text2"], top_k=5)
        assert len(results) == 2
