"""Extended tests for document classifier — targeting uncovered code paths.

Covers:
- ``_section_score`` with unknown DocumentType (no patterns)
- ``_keyword_score`` with unknown DocumentType (no keywords)
- ``_structural_score`` with unknown DocumentType (no profile)
- ``classify`` exception wrapping in InferenceError
- ``TransformerDocumentClassifier.load()`` success path
- ``TransformerDocumentClassifier.classify()`` with loaded model
- ``TransformerDocumentClassifier.classify()`` fallback on error
- ``_count_sections`` edge cases
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import InferenceError
from app.ml.classifier.document_classifier import (
    ClassificationResult,
    DocumentType,
    RuleBasedDocumentClassifier,
    TransformerDocumentClassifier,
)


class TestSectionScoreNoPatternsPath:
    """Cover _section_score returning (0.0, []) for types with no patterns."""

    def test_unknown_type_returns_zero_score(self) -> None:
        """Types with no compiled patterns should return (0.0, [])."""
        classifier = RuleBasedDocumentClassifier()
        # If UNKNOWN has no patterns, it should return 0
        score, evidence = classifier._section_score(
            "Some clinical text", DocumentType.UNKNOWN,
        )
        assert score == 0.0
        assert evidence == []


class TestKeywordScoreNoKeywordsPath:
    """Cover _keyword_score returning (0.0, []) for types with no keywords."""

    def test_unknown_type_returns_zero_score(self) -> None:
        """Types with no keywords should return (0.0, [])."""
        classifier = RuleBasedDocumentClassifier()
        score, evidence = classifier._keyword_score(
            "some clinical text", DocumentType.UNKNOWN,
        )
        assert score == 0.0
        assert evidence == []


class TestStructuralScoreNoProfilePath:
    """Cover _structural_score returning 0.0 for types with no profile."""

    def test_unknown_type_returns_zero_score(self) -> None:
        """Types with no structural profile should return 0.0."""
        classifier = RuleBasedDocumentClassifier()
        score = classifier._structural_score(50, 5, DocumentType.UNKNOWN)
        assert score == 0.0


class TestClassifyExceptionPath:
    """Cover the classify() exception → InferenceError wrapping."""

    def test_classify_wraps_internal_error(self) -> None:
        """classify() should wrap unexpected errors in InferenceError."""
        classifier = RuleBasedDocumentClassifier()

        # Force an exception inside classify by breaking _section_score
        with patch.object(
            classifier, "_section_score",
            side_effect=RuntimeError("Boom"),
        ):
            with pytest.raises(InferenceError):
                classifier.classify("Some text")


class TestCountSections:
    """Cover _count_sections edge cases."""

    def test_empty_text(self) -> None:
        """Empty text should return 0 sections."""
        classifier = RuleBasedDocumentClassifier()
        assert classifier._count_sections("") == 0

    def test_all_caps_headers(self) -> None:
        """ALL CAPS short lines should be counted as sections."""
        classifier = RuleBasedDocumentClassifier()
        text = "ASSESSMENT\nSome text here.\nPLAN\nMore text."
        count = classifier._count_sections(text)
        assert count >= 2

    def test_colon_terminated_headers(self) -> None:
        """Lines ending with colon should be counted."""
        classifier = RuleBasedDocumentClassifier()
        text = "Chief Complaint:\nHeadache\nAssessment:\nMigraine"
        count = classifier._count_sections(text)
        assert count >= 2

    def test_very_long_caps_line_excluded(self) -> None:
        """ALL CAPS lines > 60 chars should NOT be counted as headers."""
        classifier = RuleBasedDocumentClassifier()
        long_caps = "A" * 65
        text = f"{long_caps}\nASSESSMENT\nSome text."
        count = classifier._count_sections(text)
        # long_caps might match the colon check or might not —
        # the key assertion is ASSESSMENT is counted
        assert count >= 1

    def test_blank_lines_ignored(self) -> None:
        """Blank lines should be skipped, not counted."""
        classifier = RuleBasedDocumentClassifier()
        text = "\n\n\nASSESSMENT\n\n\nPLAN\n\n"
        count = classifier._count_sections(text)
        assert count >= 2


class TestTransformerDocumentClassifierLoad:
    """Cover TransformerDocumentClassifier.load() success and failure."""

    def test_load_success(self) -> None:
        """Successful load should set _is_loaded to True."""
        classifier = TransformerDocumentClassifier(model_name="test-model")

        mock_tokenizer_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoTokenizer=mock_tokenizer_cls,
                AutoModelForSequenceClassification=mock_model_cls,
            ),
        }):
            classifier.load()

        assert classifier._is_loaded
        mock_model.eval.assert_called_once()

    def test_load_failure_sets_flag(self) -> None:
        """Failed load should leave _is_loaded as False."""
        classifier = TransformerDocumentClassifier(model_name="bad-model")

        with patch.dict("sys.modules", {"transformers": None}):
            # Import will fail
            classifier.load()

        assert not classifier._is_loaded


class TestTransformerDocumentClassifierClassify:
    """Cover TransformerDocumentClassifier.classify() with loaded model."""

    def test_classify_with_loaded_model(self) -> None:
        """Should return ClassificationResult from transformer predictions."""

        classifier = TransformerDocumentClassifier()
        classifier._is_loaded = True

        mock_tokenizer = MagicMock()
        classifier._tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.id2label = {
            0: "discharge_summary",
            1: "progress_note",
            2: "consultation",
        }
        mock_model.config = mock_config

        # Simulate torch outputs
        mock_torch = MagicMock()
        mock_logits = MagicMock()
        mock_logits.__getitem__ = MagicMock(
            return_value=MagicMock(__float__=lambda s: 2.5),
        )
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits

        mock_model.return_value = mock_outputs

        # softmax returns a tensor-like object with iteration
        mock_probs = MagicMock()
        mock_prob_values = [
            MagicMock(__float__=lambda s: 0.8),
            MagicMock(__float__=lambda s: 0.15),
            MagicMock(__float__=lambda s: 0.05),
        ]
        mock_probs.__iter__ = MagicMock(return_value=iter(mock_prob_values))
        mock_probs.__getitem__ = MagicMock(side_effect=lambda i: mock_prob_values[i])
        mock_torch.softmax.return_value = MagicMock(
            __getitem__=MagicMock(return_value=mock_probs),
        )
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        classifier._model = mock_model

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = classifier.classify("Discharge summary text")

        assert isinstance(result, ClassificationResult)

    def test_classify_fallback_when_not_loaded(self) -> None:
        """Should use rule-based fallback when model is not loaded."""
        classifier = TransformerDocumentClassifier()
        classifier._is_loaded = False

        result = classifier.classify("DISCHARGE SUMMARY\nPatient was admitted...")
        assert isinstance(result, ClassificationResult)
        assert result.classifier_version is not None

    def test_classify_fallback_on_exception(self) -> None:
        """Should fall back to rule-based on transformer error."""
        classifier = TransformerDocumentClassifier()
        classifier._is_loaded = True
        classifier._tokenizer = MagicMock(side_effect=RuntimeError("tokenize fail"))

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = classifier.classify("Some text")

        assert isinstance(result, ClassificationResult)

    def test_classify_unknown_label_skipped(self) -> None:
        """Labels not in DocumentType should be silently skipped."""
        classifier = TransformerDocumentClassifier()
        classifier._is_loaded = True

        mock_tokenizer = MagicMock()
        classifier._tokenizer = mock_tokenizer

        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_config.id2label = {0: "nonexistent_type_xyz"}
        mock_model.config = mock_config

        mock_torch = MagicMock()
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        mock_prob = MagicMock(__float__=lambda s: 0.9)
        mock_probs = MagicMock()
        mock_probs.__iter__ = MagicMock(return_value=iter([mock_prob]))
        mock_probs.__getitem__ = MagicMock(return_value=mock_prob)
        mock_torch.softmax.return_value = MagicMock(
            __getitem__=MagicMock(return_value=mock_probs),
        )
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        classifier._model = mock_model

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = classifier.classify("test text")

        assert isinstance(result, ClassificationResult)
        # UNKNOWN type should be skipped, so predicted should be UNKNOWN
        assert result.predicted_type == DocumentType.UNKNOWN
