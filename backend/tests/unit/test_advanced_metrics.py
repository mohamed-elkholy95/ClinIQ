"""Tests for advanced evaluation metrics module.

Covers: Cohen's Kappa, MCC, Calibration, Partial NER matching,
ROUGE F-measure, Hierarchical ICD-10 evaluation, and AUPRC.
"""

from __future__ import annotations

import math

import pytest

from app.ml.utils.advanced_metrics import (
    AUPRCResult,
    CalibrationResult,
    HierarchicalICDMetrics,
    KappaResult,
    MCCResult,
    PartialNERMetrics,
    ROUGEResult,
    ROUGEScore,
    compute_auprc,
    compute_calibration,
    compute_cohens_kappa,
    compute_hierarchical_icd_metrics,
    compute_mcc,
    compute_partial_ner_metrics,
    compute_rouge,
)


# =========================================================================
# Cohen's Kappa
# =========================================================================


class TestCohensKappa:
    """Tests for compute_cohens_kappa."""

    def test_perfect_agreement(self) -> None:
        """Identical labels should yield kappa = 1.0."""
        labels = ["A", "B", "A", "C", "B"]
        result = compute_cohens_kappa(labels, labels)
        assert result.kappa == pytest.approx(1.0)
        assert result.observed_agreement == 1.0

    def test_no_agreement_above_chance(self) -> None:
        """Systematically opposing labels should yield kappa < 0."""
        rater_a = ["A", "B", "A", "B"]
        rater_b = ["B", "A", "B", "A"]
        result = compute_cohens_kappa(rater_a, rater_b)
        assert result.kappa < 0

    def test_moderate_agreement(self) -> None:
        """Partial agreement should yield 0 < kappa < 1."""
        rater_a = ["A", "B", "A", "B", "A"]
        rater_b = ["A", "B", "B", "B", "A"]
        result = compute_cohens_kappa(rater_a, rater_b)
        assert 0 < result.kappa < 1

    def test_unequal_lengths_raises(self) -> None:
        """Different-length inputs should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            compute_cohens_kappa(["A", "B"], ["A"])

    def test_empty_inputs_raises(self) -> None:
        """Empty inputs should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_cohens_kappa([], [])

    def test_integer_labels(self) -> None:
        """Should work with integer label types."""
        result = compute_cohens_kappa([1, 2, 3, 1], [1, 2, 3, 1])
        assert result.kappa == pytest.approx(1.0)

    def test_result_dataclass_fields(self) -> None:
        """KappaResult should have all expected fields."""
        result = compute_cohens_kappa(["A", "B"], ["A", "A"])
        assert isinstance(result, KappaResult)
        assert result.n_items == 2
        assert 0 <= result.observed_agreement <= 1
        assert 0 <= result.expected_agreement <= 1

    def test_all_same_label(self) -> None:
        """When both raters use the same label for everything, kappa = 1."""
        result = compute_cohens_kappa(["X", "X", "X"], ["X", "X", "X"])
        assert result.kappa == pytest.approx(1.0)


# =========================================================================
# Matthews Correlation Coefficient
# =========================================================================


class TestMCC:
    """Tests for compute_mcc."""

    def test_perfect_prediction(self) -> None:
        """Perfect binary predictions should yield MCC = 1.0."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        result = compute_mcc(y_true, y_true)
        assert result.mcc == pytest.approx(1.0)

    def test_inverse_prediction(self) -> None:
        """Inverted predictions should yield MCC = -1.0."""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 0, 1]
        result = compute_mcc(y_true, y_pred)
        assert result.mcc == pytest.approx(-1.0)

    def test_random_prediction(self) -> None:
        """Random-ish predictions should yield MCC near 0."""
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 1, 0, 0, 1, 0]
        result = compute_mcc(y_true, y_pred)
        assert -0.5 < result.mcc < 0.5

    def test_confusion_matrix_correct(self) -> None:
        """Confusion matrix components should be accurately computed."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        result = compute_mcc(y_true, y_pred)
        assert result.tp == 2
        assert result.fp == 1
        assert result.fn == 1
        assert result.tn == 1

    def test_unequal_lengths_raises(self) -> None:
        """Different-length inputs should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            compute_mcc([1, 0], [1])

    def test_all_positive(self) -> None:
        """All-positive predictions with all-positive labels should give MCC=0 (denom=0)."""
        result = compute_mcc([1, 1, 1], [1, 1, 1])
        # denom = sqrt((3)(3)(0)(0)) = 0, so MCC = 0 by convention
        assert result.mcc == 0.0

    def test_result_dataclass(self) -> None:
        """MCCResult should be a proper dataclass."""
        result = compute_mcc([1, 0], [1, 0])
        assert isinstance(result, MCCResult)


# =========================================================================
# Calibration Metrics
# =========================================================================


class TestCalibration:
    """Tests for compute_calibration."""

    def test_perfectly_calibrated(self) -> None:
        """Perfect calibration should yield ECE near 0."""
        # Model says 90% → actually correct 90% of the time.
        y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        y_prob = [0.9] * 10
        result = compute_calibration(y_true, y_prob, n_bins=10)
        assert result.expected_calibration_error < 0.15

    def test_poorly_calibrated(self) -> None:
        """High confidence + low accuracy → high ECE."""
        y_true = [0, 0, 0, 0, 0]
        y_prob = [0.95, 0.95, 0.95, 0.95, 0.95]
        result = compute_calibration(y_true, y_prob, n_bins=10)
        assert result.expected_calibration_error > 0.5

    def test_brier_score_perfect(self) -> None:
        """Perfect predictions should yield Brier score near 0."""
        y_true = [1, 0, 1, 0]
        y_prob = [1.0, 0.0, 1.0, 0.0]
        result = compute_calibration(y_true, y_prob)
        assert result.brier_score == pytest.approx(0.0)

    def test_brier_score_worst(self) -> None:
        """Worst-case predictions should yield Brier score = 1.0."""
        y_true = [1, 0]
        y_prob = [0.0, 1.0]
        result = compute_calibration(y_true, y_prob)
        assert result.brier_score == pytest.approx(1.0)

    def test_empty_inputs(self) -> None:
        """Empty inputs should return zero ECE and Brier."""
        result = compute_calibration([], [])
        assert result.expected_calibration_error == 0.0
        assert result.brier_score == 0.0

    def test_unequal_lengths_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            compute_calibration([1, 0], [0.5])

    def test_invalid_bins_raises(self) -> None:
        """n_bins < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bins"):
            compute_calibration([1], [0.5], n_bins=0)

    def test_bin_counts_sum(self) -> None:
        """Total of bin counts should equal sample count."""
        y_true = [1, 0, 1, 0, 1]
        y_prob = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = compute_calibration(y_true, y_prob, n_bins=5)
        assert sum(result.bin_counts) == 5

    def test_result_dataclass(self) -> None:
        """CalibrationResult should have correct fields."""
        result = compute_calibration([1, 0], [0.8, 0.2], n_bins=5)
        assert isinstance(result, CalibrationResult)
        assert result.n_bins == 5
        assert len(result.bin_accuracies) == 5


# =========================================================================
# Partial NER Span Matching
# =========================================================================


class TestPartialNER:
    """Tests for compute_partial_ner_metrics."""

    def test_exact_matches(self) -> None:
        """Exact span + type matches should yield F1 = 1.0."""
        entities = [("DISEASE", 0, 10), ("DRUG", 20, 30)]
        result = compute_partial_ner_metrics(entities, entities)
        assert result.exact_f1 == pytest.approx(1.0)
        assert result.partial_f1 == pytest.approx(1.0)
        assert result.n_exact_matches == 2

    def test_partial_overlap(self) -> None:
        """Overlapping but non-exact spans should get partial credit."""
        gold = [("DISEASE", 0, 20)]
        pred = [("DISEASE", 5, 25)]
        # Overlap: 15 chars, union: 25 chars → Jaccard = 0.6
        result = compute_partial_ner_metrics(gold, pred)
        assert result.partial_f1 > 0
        assert result.n_partial_matches == 1
        assert result.n_exact_matches == 0

    def test_no_overlap(self) -> None:
        """Non-overlapping spans should yield 0 F1."""
        gold = [("DISEASE", 0, 10)]
        pred = [("DISEASE", 50, 60)]
        result = compute_partial_ner_metrics(gold, pred)
        assert result.exact_f1 == 0.0
        assert result.partial_f1 == 0.0

    def test_type_mismatch_halves_credit(self) -> None:
        """Type mismatch should halve the partial credit."""
        gold = [("DISEASE", 0, 20)]
        pred = [("DRUG", 0, 20)]
        result = compute_partial_ner_metrics(gold, pred)
        # Exact overlap but wrong type → type_weighted < partial
        assert result.type_weighted_f1 < result.partial_f1

    def test_both_empty(self) -> None:
        """Both empty should return perfect scores."""
        result = compute_partial_ner_metrics([], [])
        assert result.exact_f1 == 1.0
        assert result.partial_f1 == 1.0

    def test_gold_empty(self) -> None:
        """No gold entities + some predictions → 0 F1."""
        result = compute_partial_ner_metrics([], [("DRUG", 0, 10)])
        assert result.exact_f1 == 0.0
        assert result.n_unmatched_pred == 1

    def test_pred_empty(self) -> None:
        """No predictions + some gold → 0 F1."""
        result = compute_partial_ner_metrics([("DRUG", 0, 10)], [])
        assert result.exact_f1 == 0.0
        assert result.n_unmatched_gold == 1

    def test_overlap_threshold(self) -> None:
        """Overlaps below threshold should not count as matches."""
        gold = [("DISEASE", 0, 100)]
        pred = [("DISEASE", 90, 110)]
        # Overlap: 10 chars, union: 110 chars → Jaccard ≈ 0.09
        result = compute_partial_ner_metrics(gold, pred, overlap_threshold=0.5)
        assert result.n_partial_matches == 0
        assert result.n_exact_matches == 0

    def test_result_dataclass(self) -> None:
        """PartialNERMetrics should have all fields."""
        result = compute_partial_ner_metrics(
            [("A", 0, 5)], [("A", 0, 5)]
        )
        assert isinstance(result, PartialNERMetrics)
        assert result.n_gold == 1
        assert result.n_pred == 1

    def test_multiple_entities_greedy(self) -> None:
        """Greedy matching should assign each gold to at most one pred."""
        gold = [("DRUG", 0, 10), ("DRUG", 20, 30)]
        pred = [("DRUG", 0, 10), ("DRUG", 20, 30), ("DRUG", 40, 50)]
        result = compute_partial_ner_metrics(gold, pred)
        assert result.n_exact_matches == 2
        assert result.n_unmatched_pred == 1


# =========================================================================
# ROUGE F-measure
# =========================================================================


class TestROUGE:
    """Tests for compute_rouge."""

    def test_identical_texts(self) -> None:
        """Identical reference and hypothesis should yield F1 = 1.0."""
        text = "The patient presents with chest pain and shortness of breath"
        result = compute_rouge(text, text)
        assert result.rouge1.f1 == pytest.approx(1.0)
        assert result.rouge2.f1 == pytest.approx(1.0)
        assert result.rougeL.f1 == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Completely disjoint texts should yield F1 = 0."""
        result = compute_rouge("alpha beta gamma", "delta epsilon zeta")
        assert result.rouge1.f1 == 0.0
        assert result.rouge2.f1 == 0.0

    def test_partial_overlap(self) -> None:
        """Partial word overlap should yield 0 < F1 < 1."""
        ref = "patient has diabetes and hypertension"
        hyp = "patient presents with diabetes"
        result = compute_rouge(ref, hyp)
        assert 0 < result.rouge1.f1 < 1

    def test_precision_recall_distinct(self) -> None:
        """Precision and recall should differ when lengths differ."""
        ref = "a b c d e"
        hyp = "a b"
        result = compute_rouge(ref, hyp)
        assert result.rouge1.precision > result.rouge1.recall

    def test_length_ratio(self) -> None:
        """Length ratio should reflect compression."""
        ref = "the patient has many conditions and medications"
        hyp = "many conditions"
        result = compute_rouge(ref, hyp)
        assert result.length_ratio < 1.0
        assert result.reference_length == 7
        assert result.hypothesis_length == 2

    def test_empty_reference(self) -> None:
        """Empty reference should yield all zeros."""
        result = compute_rouge("", "some text")
        assert result.rouge1.f1 == 0.0

    def test_empty_hypothesis(self) -> None:
        """Empty hypothesis should yield all zeros."""
        result = compute_rouge("some text", "")
        assert result.rouge1.f1 == 0.0

    def test_result_structure(self) -> None:
        """ROUGEResult should contain ROUGEScore substructures."""
        result = compute_rouge("a b c", "a b")
        assert isinstance(result, ROUGEResult)
        assert isinstance(result.rouge1, ROUGEScore)
        assert isinstance(result.rouge2, ROUGEScore)
        assert isinstance(result.rougeL, ROUGEScore)

    def test_rouge2_bigram_order(self) -> None:
        """ROUGE-2 should respect bigram order."""
        ref = "a b c d"
        hyp = "d c b a"  # Same words, reversed bigrams
        result = compute_rouge(ref, hyp)
        # ROUGE-1 should be perfect (same unigrams) but ROUGE-2 should be low
        assert result.rouge1.f1 == pytest.approx(1.0)
        assert result.rouge2.f1 < 1.0


# =========================================================================
# Hierarchical ICD-10 Evaluation
# =========================================================================


class TestHierarchicalICD:
    """Tests for compute_hierarchical_icd_metrics."""

    def test_perfect_predictions(self) -> None:
        """Exact code matches at all levels."""
        codes = ["E11.65", "I10", "J44.1"]
        result = compute_hierarchical_icd_metrics(codes, codes)
        assert result.full_code_accuracy == 1.0
        assert result.block_accuracy == 1.0
        assert result.chapter_accuracy == 1.0

    def test_block_match_only(self) -> None:
        """Same 3-char block but different specificity."""
        gold = ["E11.65"]
        pred = ["E11.9"]
        result = compute_hierarchical_icd_metrics(gold, pred)
        assert result.full_code_accuracy == 0.0
        assert result.block_accuracy == 1.0
        assert result.chapter_accuracy == 1.0

    def test_chapter_match_only(self) -> None:
        """Same chapter but different block."""
        gold = ["E11.65"]
        pred = ["E78.5"]
        result = compute_hierarchical_icd_metrics(gold, pred)
        assert result.full_code_accuracy == 0.0
        assert result.block_accuracy == 0.0
        assert result.chapter_accuracy == 1.0

    def test_no_match(self) -> None:
        """Completely wrong chapter."""
        gold = ["E11.65"]
        pred = ["I10"]
        result = compute_hierarchical_icd_metrics(gold, pred)
        assert result.full_code_accuracy == 0.0
        assert result.block_accuracy == 0.0
        assert result.chapter_accuracy == 0.0

    def test_mixed_results(self) -> None:
        """Mix of exact, block, chapter, and no matches."""
        gold = ["E11.65", "I10", "J44.1", "M79.3"]
        pred = ["E11.65", "I25.9", "J44.0", "A01.0"]
        result = compute_hierarchical_icd_metrics(gold, pred)
        assert result.full_code_matches == 1  # E11.65
        assert result.block_matches == 2      # E11 + J44
        assert result.chapter_matches == 3    # E + I + J

    def test_unequal_lengths_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            compute_hierarchical_icd_metrics(["E11"], ["E11", "I10"])

    def test_empty_codes(self) -> None:
        """Empty inputs should return zeros."""
        result = compute_hierarchical_icd_metrics([], [])
        assert result.n_samples == 0
        assert result.full_code_accuracy == 0.0

    def test_case_insensitive(self) -> None:
        """Comparison should be case-insensitive."""
        result = compute_hierarchical_icd_metrics(["e11.65"], ["E11.65"])
        assert result.full_code_accuracy == 1.0

    def test_dot_normalisation(self) -> None:
        """Codes with and without dots should match."""
        result = compute_hierarchical_icd_metrics(["E1165"], ["E11.65"])
        assert result.full_code_accuracy == 1.0

    def test_result_dataclass(self) -> None:
        """HierarchicalICDMetrics fields check."""
        result = compute_hierarchical_icd_metrics(["I10"], ["I10"])
        assert isinstance(result, HierarchicalICDMetrics)
        assert result.n_samples == 1


# =========================================================================
# AUPRC
# =========================================================================


class TestAUPRC:
    """Tests for compute_auprc."""

    def test_perfect_ranking(self) -> None:
        """Perfect separation should yield AUPRC = 1.0."""
        y_true = [1, 1, 1, 0, 0, 0]
        y_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        result = compute_auprc(y_true, y_scores)
        assert result.auprc == pytest.approx(1.0)

    def test_worst_ranking(self) -> None:
        """Inverted ranking should yield low AUPRC."""
        y_true = [0, 0, 0, 1, 1, 1]
        y_scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        result = compute_auprc(y_true, y_scores)
        assert result.auprc < 0.5

    def test_all_positive(self) -> None:
        """All positives should yield AUPRC = 1.0 regardless of scores."""
        result = compute_auprc([1, 1, 1], [0.5, 0.6, 0.7])
        assert result.auprc == pytest.approx(1.0)

    def test_no_positives(self) -> None:
        """No positives should yield AUPRC = 0."""
        result = compute_auprc([0, 0, 0], [0.5, 0.6, 0.7])
        assert result.auprc == 0.0

    def test_empty_inputs(self) -> None:
        """Empty inputs should yield 0."""
        result = compute_auprc([], [])
        assert result.auprc == 0.0

    def test_label_propagated(self) -> None:
        """Custom label should appear in result."""
        result = compute_auprc([1, 0], [0.9, 0.1], label="diabetes")
        assert result.label == "diabetes"

    def test_n_counts(self) -> None:
        """n_positive and n_total should be correct."""
        result = compute_auprc([1, 0, 1, 0, 0], [0.9, 0.1, 0.8, 0.2, 0.3])
        assert result.n_positive == 2
        assert result.n_total == 5

    def test_result_dataclass(self) -> None:
        """AUPRCResult should be a proper dataclass."""
        result = compute_auprc([1, 0], [0.9, 0.1])
        assert isinstance(result, AUPRCResult)
