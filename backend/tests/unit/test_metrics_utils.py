"""Tests for app.ml.utils.metrics — covering precision_at_k, compute_rouge, _compute_lcs_ratio,
and compute_confusion_metrics edge cases."""

import numpy as np
import pytest

from app.ml.utils.metrics import (
    compute_confusion_matrix_metrics,
    compute_rouge_scores,
    compute_precision_at_k,
)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    """Cover precision_at_k including the k<=0 guard (line 123)."""

    def test_k_zero_returns_zero(self) -> None:
        """k=0 should short-circuit to 0.0."""
        y_true = np.array([[1, 0, 1]])
        y_scores = np.array([[0.9, 0.1, 0.8]])
        assert compute_precision_at_k(y_true, y_scores, k=0) == 0.0

    def test_k_negative_returns_zero(self) -> None:
        """Negative k should also return 0.0."""
        y_true = np.array([[1, 0, 1]])
        y_scores = np.array([[0.9, 0.1, 0.8]])
        assert compute_precision_at_k(y_true, y_scores, k=-1) == 0.0

    def test_perfect_precision(self) -> None:
        """All top-k predictions are relevant."""
        y_true = np.array([[1, 1, 0, 0]])
        y_scores = np.array([[0.9, 0.8, 0.1, 0.2]])
        assert compute_precision_at_k(y_true, y_scores, k=2) == 1.0

    def test_partial_precision(self) -> None:
        """Only some top-k predictions are relevant."""
        y_true = np.array([[1, 0, 0, 1]])
        y_scores = np.array([[0.9, 0.8, 0.7, 0.6]])
        # top-2 indices by score: 0 (true=1), 1 (true=0) → 1/2
        assert compute_precision_at_k(y_true, y_scores, k=2) == pytest.approx(0.5)

    def test_multi_sample(self) -> None:
        """Average across multiple samples."""
        y_true = np.array([[1, 0], [0, 1]])
        y_scores = np.array([[0.9, 0.1], [0.1, 0.9]])
        assert compute_precision_at_k(y_true, y_scores, k=1) == 1.0


# ---------------------------------------------------------------------------
# compute_rouge
# ---------------------------------------------------------------------------


class TestComputeRouge:
    """Cover ROUGE edge cases: empty strings, single words, bigram edges (lines 220, 234, 252)."""

    def test_empty_reference(self) -> None:
        """Empty reference → all zeros (line 220)."""
        result = compute_rouge_scores("", "some hypothesis text")
        assert result["rouge1"] == 0.0
        assert result["rouge2"] == 0.0
        assert result["rougeL"] == 0.0

    def test_empty_hypothesis(self) -> None:
        """Empty hypothesis → all zeros."""
        result = compute_rouge_scores("reference text", "")
        assert result["rouge1"] == 0.0
        assert result["rouge2"] == 0.0
        assert result["rougeL"] == 0.0

    def test_both_empty(self) -> None:
        """Both empty → all zeros."""
        result = compute_rouge_scores("", "")
        assert result["rouge1"] == 0.0
        assert result["rouge2"] == 0.0
        assert result["rougeL"] == 0.0

    def test_single_word_no_bigrams(self) -> None:
        """Single-word strings have no bigrams (line 234)."""
        result = compute_rouge_scores("hello", "hello")
        assert result["rouge1"] == 1.0
        assert result["rouge2"] == 0.0  # No bigrams
        assert result["rougeL"] == 1.0

    def test_identical_strings(self) -> None:
        """Perfect overlap."""
        text = "the patient has diabetes"
        result = compute_rouge_scores(text, text)
        assert result["rouge1"] == 1.0
        assert result["rouge2"] == 1.0
        assert result["rougeL"] == 1.0

    def test_partial_overlap(self) -> None:
        """Partial word overlap yields intermediate scores."""
        ref = "the patient has diabetes and hypertension"
        hyp = "the patient has fever and chills"
        result = compute_rouge_scores(ref, hyp)
        assert 0.0 < result["rouge1"] < 1.0


# ---------------------------------------------------------------------------
# compute_confusion_metrics
# ---------------------------------------------------------------------------


class TestComputeConfusionMetrics:
    """Cover all-zero edge (lines 275-281)."""

    def test_all_zero(self) -> None:
        """tp=fp=fn=tn=0 should not divide by zero."""
        result = compute_confusion_matrix_metrics(tp=0, fp=0, fn=0, tn=0)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["specificity"] == 0.0
        assert result["accuracy"] == 0.0

    def test_perfect_classifier(self) -> None:
        """tp=10, tn=10, fp=fn=0 → perfect metrics."""
        result = compute_confusion_matrix_metrics(tp=10, fp=0, fn=0, tn=10)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["specificity"] == 1.0
        assert result["accuracy"] == 1.0

    def test_no_positive_predictions(self) -> None:
        """tp=0, fp=0 → precision is 0.0 (no divide-by-zero)."""
        result = compute_confusion_matrix_metrics(tp=0, fp=0, fn=5, tn=5)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_no_true_positives_but_false_positives(self) -> None:
        """tp=0, fp=5 → precision=0, recall=0."""
        result = compute_confusion_matrix_metrics(tp=0, fp=5, fn=0, tn=5)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
