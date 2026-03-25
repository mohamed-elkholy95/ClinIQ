"""Evaluation metrics for clinical NLP tasks."""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    hamming_loss: float | None = None  # For multi-label
    subset_accuracy: float | None = None  # For multi-label


@dataclass
class NERMetrics:
    """Container for NER evaluation metrics."""

    entity_type: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of true entities


@dataclass
class SummarizationMetrics:
    """Container for summarization metrics."""

    rouge1: float
    rouge2: float
    rougeL: float
    avg_length: float
    length_ratio: float


def compute_classification_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    average: str = "macro",
) -> ClassificationMetrics:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision_micro=precision_score(y_true, y_pred, average="micro", zero_division=0),
        recall_micro=recall_score(y_true, y_pred, average="micro", zero_division=0),
        f1_micro=f1_score(y_true, y_pred, average="micro", zero_division=0),
        precision_weighted=precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        recall_weighted=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


def compute_multilabel_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
) -> ClassificationMetrics:
    """Compute metrics for multi-label classification (e.g., ICD-10 codes)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import hamming_loss as sklearn_hamming_loss

    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision_micro=precision_score(y_true, y_pred, average="micro", zero_division=0),
        recall_micro=recall_score(y_true, y_pred, average="micro", zero_division=0),
        f1_micro=f1_score(y_true, y_pred, average="micro", zero_division=0),
        precision_weighted=precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        recall_weighted=recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_weighted=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        hamming_loss=sklearn_hamming_loss(y_true, y_pred),
        subset_accuracy=accuracy_score(y_true, y_pred),
    )
    return metrics


def compute_precision_at_k(
    y_true: NDArray[np.int_],
    y_scores: NDArray[np.float64],
    k: int = 5,
) -> float:
    """Compute precision@k for multi-label classification."""
    if k <= 0:
        return 0.0

    # Get top k predictions for each sample
    top_k_indices = np.argsort(y_scores, axis=1)[:, ::-1][:, :k]

    precisions = []
    for i in range(len(y_true)):
        true_positives = sum(y_true[i, idx] for idx in top_k_indices[i])
        precisions.append(true_positives / k)

    return float(np.mean(precisions))


def compute_ner_metrics(
    true_entities: list[tuple[str, int, int]],  # (entity_type, start, end)
    pred_entities: list[tuple[str, int, int]],
) -> list[NERMetrics]:
    """Compute per-entity-type NER metrics."""
    # Count true, predicted, and matched entities per type
    true_by_type: Counter[str] = Counter(e[0] for e in true_entities)
    pred_by_type: Counter[str] = Counter(e[0] for e in pred_entities)

    # Match entities (exact match on type and span)
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    matched = true_set & pred_set

    matched_by_type: Counter[str] = Counter(e[0] for e in matched)

    # Compute metrics per entity type
    all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
    metrics = []

    for entity_type in sorted(all_types):
        tp = matched_by_type.get(entity_type, 0)
        fp = pred_by_type.get(entity_type, 0) - tp
        fn = true_by_type.get(entity_type, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(
            NERMetrics(
                entity_type=entity_type,
                precision=precision,
                recall=recall,
                f1=f1,
                support=true_by_type.get(entity_type, 0),
            )
        )

    return metrics


def compute_overall_ner_metrics(metrics_list: list[NERMetrics]) -> NERMetrics:
    """Compute overall NER metrics (micro-averaged)."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for m in metrics_list:
        # Back-compute tp, fp, fn from precision, recall, support
        tp = int(m.recall * m.support) if m.recall > 0 else 0
        fn = m.support - tp
        fp = int(tp * (1 / m.precision - 1)) if m.precision > 0 else 0

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return NERMetrics(
        entity_type="OVERALL",
        precision=precision,
        recall=recall,
        f1=f1,
        support=total_tp + total_fn,
    )


def compute_rouge_scores(
    reference: str,
    hypothesis: str,
) -> dict[str, float]:
    """Compute ROUGE scores for summarization evaluation."""
    # Simple implementation without external dependencies
    # For production, consider rouge-score library

    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    # ROUGE-1 (unigrams)
    if len(ref_words) == 0 or len(hyp_words) == 0:
        rouge1 = 0.0
    else:
        overlap = len(ref_words & hyp_words)
        rouge1 = overlap / len(ref_words)  # Recall-focused

    # ROUGE-2 (bigrams)
    ref_bigrams = {
        tuple(reference.lower().split()[i : i + 2]) for i in range(len(reference.split()) - 1)
    }
    hyp_bigrams = {
        tuple(hypothesis.lower().split()[i : i + 2]) for i in range(len(hypothesis.split()) - 1)
    }

    if len(ref_bigrams) == 0 or len(hyp_bigrams) == 0:
        rouge2 = 0.0
    else:
        overlap = len(ref_bigrams & hyp_bigrams)
        rouge2 = overlap / len(ref_bigrams)

    # ROUGE-L (longest common subsequence)
    rougeL = _compute_lcs_ratio(reference.lower().split(), hypothesis.lower().split())

    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }


def _compute_lcs_ratio(seq1: list[str], seq2: list[str]) -> float:
    """Compute longest common subsequence ratio."""
    if not seq1 or not seq2:
        return 0.0

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    return lcs_length / len(seq1)  # Recall-focused


def compute_confusion_matrix_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float]:
    """Compute metrics from confusion matrix components."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
    }
