"""Unit tests for SklearnICDClassifier and TransformerICDClassifier.

Tests the sklearn-based ICD-10 classifier baseline including model loading
(both from file and empty initialisation), prediction with predict_proba
and decision_function classifiers, batch prediction, top-k extraction,
and the hierarchical classifier.

Also covers ICDCodePrediction/ICDPredictionResult dataclasses and the
get_chapter_for_code helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.icd.model import (
    BaseICDClassifier,
    HierarchicalICDClassifier,
    ICDCodePrediction,
    ICDPredictionResult,
    SklearnICDClassifier,
    TransformerICDClassifier,
    get_chapter_for_code,
)

# ---------------------------------------------------------------------------
# get_chapter_for_code
# ---------------------------------------------------------------------------

class TestGetChapterForCode:
    """Tests for ICD-10 chapter resolution."""

    def test_circulatory_code(self) -> None:
        assert get_chapter_for_code("I10") == "Diseases of the circulatory system"

    def test_respiratory_code(self) -> None:
        assert get_chapter_for_code("J44.1") == "Diseases of the respiratory system"

    def test_endocrine_code(self) -> None:
        assert get_chapter_for_code("E11.9") == "Endocrine, nutritional and metabolic diseases"

    def test_injury_code_s(self) -> None:
        assert get_chapter_for_code("S72.001A") == "Injury, poisoning and certain other consequences of external causes"

    def test_injury_code_t(self) -> None:
        assert get_chapter_for_code("T39.1") == "Injury, poisoning and certain other consequences of external causes"

    def test_external_causes_v(self) -> None:
        assert get_chapter_for_code("V03.10") == "External causes of morbidity"

    def test_external_causes_w(self) -> None:
        assert get_chapter_for_code("W19") == "External causes of morbidity"

    def test_external_causes_y(self) -> None:
        assert get_chapter_for_code("Y93.11") == "External causes of morbidity"

    def test_z_code(self) -> None:
        assert get_chapter_for_code("Z00.00") == "Factors influencing health status and contact with health services"

    def test_lowercase_input(self) -> None:
        result = get_chapter_for_code("i10")
        assert result == "Diseases of the circulatory system"

    def test_unknown_prefix(self) -> None:
        assert get_chapter_for_code("0XX") is None

    def test_empty_string(self) -> None:
        assert get_chapter_for_code("") is None

    def test_mental_code(self) -> None:
        assert get_chapter_for_code("F32.9") == "Mental and behavioral disorders"


# ---------------------------------------------------------------------------
# ICDCodePrediction and ICDPredictionResult dataclasses
# ---------------------------------------------------------------------------

class TestICDCodePrediction:
    """Tests for the ICDCodePrediction dataclass."""

    def test_to_dict(self) -> None:
        pred = ICDCodePrediction(
            code="I10", description="Essential hypertension", confidence=0.9,
            chapter="Circulatory", category="Hypertensive", contributing_text=["htn"]
        )
        d = pred.to_dict()
        assert d["code"] == "I10"
        assert d["confidence"] == 0.9
        assert d["contributing_text"] == ["htn"]

    def test_defaults(self) -> None:
        pred = ICDCodePrediction(code="I10", description=None, confidence=0.5)
        assert pred.chapter is None
        assert pred.category is None
        assert pred.contributing_text is None


class TestICDPredictionResult:
    """Tests for the ICDPredictionResult dataclass."""

    def test_top_k(self) -> None:
        preds = [
            ICDCodePrediction(code="A", description=None, confidence=0.3),
            ICDCodePrediction(code="B", description=None, confidence=0.9),
            ICDCodePrediction(code="C", description=None, confidence=0.6),
        ]
        result = ICDPredictionResult(
            predictions=preds, processing_time_ms=10.0,
            model_name="test", model_version="1.0"
        )
        top2 = result.top_k(2)
        assert len(top2) == 2
        assert top2[0].code == "B"
        assert top2[1].code == "C"

    def test_to_dict(self) -> None:
        result = ICDPredictionResult(
            predictions=[ICDCodePrediction(code="I10", description="HTN", confidence=0.9)],
            processing_time_ms=5.0, model_name="test", model_version="1.0",
            document_summary="summary"
        )
        d = result.to_dict()
        assert d["model_name"] == "test"
        assert len(d["predictions"]) == 1
        assert d["document_summary"] == "summary"


# ---------------------------------------------------------------------------
# SklearnICDClassifier
# ---------------------------------------------------------------------------

class TestSklearnICDClassifier:
    """Tests for the sklearn baseline ICD classifier."""

    def test_load_empty_model(self) -> None:
        clf = SklearnICDClassifier()
        clf.load()
        assert clf.is_loaded
        assert clf.classifier is None
        assert clf.feature_extractor is not None

    def test_load_from_file(self) -> None:
        """Test loading a pickled model from a file path."""
        mock_extractor = MagicMock()
        mock_classifier = MagicMock()
        mock_binarizer = MagicMock()
        pickled_data = {
            "model": mock_classifier,
            "feature_extractor": mock_extractor,
            "label_binarizer": mock_binarizer,
            "code_descriptions": {"I10": "HTN"},
        }
        with patch("builtins.open", mock_open()), \
             patch("pickle.load", return_value=pickled_data):
            clf = SklearnICDClassifier(model_path="/fake/model.pkl")
            clf.load()
        assert clf.is_loaded
        assert clf.classifier is mock_classifier
        assert clf.code_descriptions == {"I10": "HTN"}

    def test_load_failure_raises(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError("not found")):
            clf = SklearnICDClassifier(model_path="/bad/path.pkl")
            with pytest.raises(ModelLoadError):
                clf.load()

    def test_predict_untrained_raises(self) -> None:
        clf = SklearnICDClassifier()
        clf.load()  # empty model
        with pytest.raises(InferenceError, match="not trained"):
            clf.predict("some text")

    def test_predict_with_proba(self) -> None:
        """Test prediction with a classifier that has predict_proba."""
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([[0.05, 0.8, 0.6]])

        mock_binarizer = MagicMock()
        mock_binarizer.classes_ = np.array(["A01", "I10", "E11.9"])

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1, 2, 3]])

        clf = SklearnICDClassifier()
        clf._is_loaded = True
        clf.classifier = mock_clf
        clf.label_binarizer = mock_binarizer
        clf.feature_extractor = mock_extractor
        clf.code_descriptions = {"I10": "HTN", "E11.9": "Diabetes"}

        result = clf.predict("patient has hypertension", top_k=5)
        assert isinstance(result, ICDPredictionResult)
        assert result.model_name == "sklearn-baseline"
        assert len(result.predictions) >= 1
        # Should NOT include A01 (confidence 0.05 < 0.1 threshold)
        codes = [p.code for p in result.predictions]
        assert "A01" not in codes
        assert "I10" in codes

    def test_predict_with_decision_function(self) -> None:
        """Test prediction using decision_function (no predict_proba)."""
        mock_clf = MagicMock(spec=[])  # no predict_proba attribute
        mock_clf.decision_function = MagicMock(return_value=np.array([[2.0, -2.0]]))

        mock_binarizer = MagicMock()
        mock_binarizer.classes_ = np.array(["I10", "J44"])

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1, 2]])

        clf = SklearnICDClassifier()
        clf._is_loaded = True
        clf.classifier = mock_clf
        clf.label_binarizer = mock_binarizer
        clf.feature_extractor = mock_extractor
        clf.code_descriptions = {}

        result = clf.predict("text")
        # sigmoid(2.0) ≈ 0.88, sigmoid(-2.0) ≈ 0.12
        assert len(result.predictions) >= 1
        assert result.predictions[0].code == "I10"

    def test_predict_batch_untrained_raises(self) -> None:
        clf = SklearnICDClassifier()
        clf.load()
        with pytest.raises(InferenceError, match="not trained"):
            clf.predict_batch(["a", "b"])

    def test_predict_batch(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = np.array([
            [0.9, 0.2],
            [0.3, 0.8],
        ])

        mock_binarizer = MagicMock()
        mock_binarizer.classes_ = np.array(["I10", "E11.9"])

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1, 2], [3, 4]])

        clf = SklearnICDClassifier()
        clf._is_loaded = True
        clf.classifier = mock_clf
        clf.label_binarizer = mock_binarizer
        clf.feature_extractor = mock_extractor
        clf.code_descriptions = {}

        results = clf.predict_batch(["text1", "text2"])
        assert len(results) == 2
        assert all(r.processing_time_ms > 0 for r in results)

    def test_ensure_loaded_triggers_load(self) -> None:
        clf = SklearnICDClassifier()
        assert not clf.is_loaded
        clf.ensure_loaded()
        assert clf.is_loaded


# ---------------------------------------------------------------------------
# TransformerICDClassifier (mocked — no real model weights)
# ---------------------------------------------------------------------------

class TestTransformerICDClassifier:
    """Tests for the transformer-based ICD classifier with mocked torch/transformers."""

    def test_load_success(self) -> None:
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "I10", 1: "E11.9"}
        MagicMock()

        with patch("app.ml.icd.model.TransformerICDClassifier.load") as mock_load:
            mock_load.side_effect = lambda: setattr(
                clf, '_is_loaded', True
            )
            clf = TransformerICDClassifier(model_name="test-bert")
            clf.load()
            assert clf._is_loaded

    def test_load_failure_raises(self) -> None:
        clf = TransformerICDClassifier(model_name="nonexistent-model")
        with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": MagicMock()}):
            with patch("app.ml.icd.model.TransformerICDClassifier.load",
                       side_effect=ModelLoadError("test", "import error")):
                with pytest.raises(ModelLoadError):
                    clf.load()


# ---------------------------------------------------------------------------
# HierarchicalICDClassifier
# ---------------------------------------------------------------------------

class TestHierarchicalICDClassifier:
    """Tests for the hierarchical ICD classifier."""

    def _make_mock_classifier(self, predictions: list[ICDCodePrediction]) -> BaseICDClassifier:
        mock = MagicMock(spec=BaseICDClassifier)
        mock.predict.return_value = ICDPredictionResult(
            predictions=predictions, processing_time_ms=1.0,
            model_name="mock", model_version="1.0"
        )
        return mock

    def test_predict_dispatches_to_code_classifiers(self) -> None:
        chapter_clf = self._make_mock_classifier([
            ICDCodePrediction(code="circulatory", description=None, confidence=0.9),
        ])
        code_clf = self._make_mock_classifier([
            ICDCodePrediction(code="I10", description="HTN", confidence=0.85),
        ])

        hier = HierarchicalICDClassifier(
            chapter_classifier=chapter_clf,
            code_classifiers={"circulatory": code_clf},
        )
        hier._is_loaded = True

        result = hier.predict("patient with hypertension", top_k=5)
        assert len(result.predictions) == 1
        assert result.predictions[0].code == "I10"

    def test_predict_no_matching_chapter(self) -> None:
        chapter_clf = self._make_mock_classifier([
            ICDCodePrediction(code="neuro", description=None, confidence=0.7),
        ])
        hier = HierarchicalICDClassifier(
            chapter_classifier=chapter_clf, code_classifiers={},
        )
        hier._is_loaded = True

        result = hier.predict("text")
        assert result.predictions == []

    def test_predict_batch(self) -> None:
        chapter_clf = self._make_mock_classifier([])
        hier = HierarchicalICDClassifier(
            chapter_classifier=chapter_clf, code_classifiers={},
        )
        hier._is_loaded = True

        results = hier.predict_batch(["a", "b"])
        assert len(results) == 2

    def test_load_loads_all_sub_classifiers(self) -> None:
        chapter_clf = MagicMock(spec=BaseICDClassifier)
        code_clf = MagicMock(spec=BaseICDClassifier)

        hier = HierarchicalICDClassifier(
            chapter_classifier=chapter_clf,
            code_classifiers={"ch1": code_clf},
        )
        hier.load()
        chapter_clf.load.assert_called_once()
        code_clf.load.assert_called_once()
        assert hier.is_loaded
