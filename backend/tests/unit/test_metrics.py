"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from app.ml.utils.metrics import (
    ClassificationMetrics,
    NERMetrics,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_ner_metrics,
    compute_overall_ner_metrics,
    compute_precision_at_k,
    compute_rouge_scores,
)


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_compute_classification_metrics(self):
        """Test basic classification metrics computation."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert isinstance(metrics, ClassificationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_macro <= 1
        assert 0 <= metrics.precision_macro <= 1
        assert 0 <= metrics.recall_macro <= 1

    def test_compute_multilabel_metrics(self):
        """Test multi-label classification metrics."""
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ])
        y_pred = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
        ])

        metrics = compute_multilabel_metrics(y_true, y_pred)

        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.hamming_loss is not None
        assert metrics.subset_accuracy is not None

    def test_compute_precision_at_k(self):
        """Test precision@k computation."""
        y_true = np.array([
            [1, 0, 1, 0, 1],  # 3 positives
            [0, 1, 0, 1, 0],  # 2 positives
        ])
        y_scores = np.array([
            [0.9, 0.8, 0.7, 0.3, 0.2],  # Top 3 should include positions 0, 1, 3
            [0.9, 0.8, 0.7, 0.3, 0.2],  # Top 2 should include positions 0, 1
        ])

        p_at_3 = compute_precision_at_k(y_true, y_scores, k=3)

        assert 0 <= p_at_3 <= 1

    def test_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.f1_macro == 1.0


class TestNERMetrics:
    """Tests for NER evaluation metrics."""

    def test_compute_ner_metrics(self):
        """Test NER metrics computation."""
        true_entities = [
            ("DISEASE", 0, 10),
            ("MEDICATION", 20, 30),
            ("DISEASE", 40, 50),
        ]
        pred_entities = [
            ("DISEASE", 0, 10),
            ("MEDICATION", 20, 30),
            ("PROCEDURE", 60, 70),
        ]

        metrics = compute_ner_metrics(true_entities, pred_entities)

        assert len(metrics) > 0
        assert all(isinstance(m, NERMetrics) for m in metrics)

        # Find disease metrics
        disease_metrics = next((m for m in metrics if m.entity_type == "DISEASE"), None)
        assert disease_metrics is not None
        assert disease_metrics.recall > 0

    def test_compute_overall_ner_metrics(self):
        """Test overall NER metrics aggregation."""
        metrics_list = [
            NERMetrics(entity_type="DISEASE", precision=0.8, recall=0.9, f1=0.85, support=10),
            NERMetrics(entity_type="MEDICATION", precision=0.9, recall=0.8, f1=0.85, support=15),
        ]

        overall = compute_overall_ner_metrics(metrics_list)

        assert overall.entity_type == "OVERALL"
        assert 0 <= overall.f1 <= 1
        assert overall.support == 25


class TestSummarizationMetrics:
    """Tests for summarization metrics."""

    def test_compute_rouge_scores(self):
        """Test ROUGE score computation."""
        reference = "The patient has diabetes and takes metformin daily."
        hypothesis = "Patient has diabetes on metformin."

        scores = compute_rouge_scores(reference, hypothesis)

        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
        assert 0 <= scores["rouge1"] <= 1
        assert 0 <= scores["rouge2"] <= 1
        assert 0 <= scores["rougeL"] <= 1

    def test_rouge_identical_texts(self):
        """Test ROUGE for identical texts."""
        text = "This is the same text."
        scores = compute_rouge_scores(text, text)

        assert scores["rouge1"] == 1.0
        assert scores["rougeL"] == 1.0

    def test_rouge_no_overlap(self):
        """Test ROUGE for completely different texts."""
        reference = "apple banana orange"
        hypothesis = "dog cat fish"

        scores = compute_rouge_scores(reference, hypothesis)

        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0


class TestMetricEdgeCases:
    """Test edge cases for metrics."""

    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        true_entities = []
        pred_entities = []

        metrics = compute_ner_metrics(true_entities, pred_entities)
        assert len(metrics) == 0

    def test_single_class(self):
        """Test metrics with single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])

        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 0.75

    def test_all_wrong_predictions(self):
        """Test metrics when all predictions are wrong."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics.accuracy == 0.0
        assert metrics.recall_macro == 0.0
