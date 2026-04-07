"""Advanced evaluation metrics for clinical NLP benchmarking.

Extends the base metrics module with clinical-domain-specific evaluation
tools that go beyond standard F1:

* **Cohen's Kappa & Weighted Kappa** — Inter-annotator agreement metrics
  essential for clinical NLP where label reliability varies.  Kappa
  corrects for chance agreement, making it the standard for reporting
  annotation quality in i2b2/n2c2 shared tasks and clinical coding audits.

* **Matthews Correlation Coefficient (MCC)** — A balanced metric that
  accounts for all four confusion matrix quadrants.  Unlike F1 which
  ignores true negatives, MCC is informative even when class distributions
  are highly skewed — common in ICD-10 coding where most codes are absent
  for any given encounter.

* **Calibration metrics** — Expected Calibration Error (ECE) and Brier
  score for assessing whether predicted confidence scores are trustworthy.
  In clinical decision support, a model that reports 90% confidence should
  be correct ~90% of the time; poorly calibrated models erode clinician
  trust and can cause alert fatigue.

* **Partial span matching for NER** — Relaxed entity evaluation using
  token-level overlap (Jaccard) instead of exact boundaries.  Exact
  matching is the gold standard (see ``metrics.py``), but partial matching
  reveals how close a model is to correct extraction — a model that
  captures "type 2 diabetes" as "diabetes" is closer than one that
  misses it entirely.

* **ROUGE F-measure** — Full precision/recall/F1 ROUGE instead of
  recall-only, giving a more balanced summarisation assessment.

* **Hierarchical ICD-10 evaluation** — Chapter-level and block-level
  accuracy in addition to full-code matching.  A prediction of E11.9
  (Type 2 DM, unspecified) when the gold is E11.65 (Type 2 DM with
  hyperglycemia) is partially correct at the chapter (E00-E89) and
  3-character (E11) levels even though the full code is wrong.

Design decisions
----------------
* **No external dependencies beyond numpy** — All metrics are
  implemented from scratch or use only numpy/collections so the module
  stays importable without sklearn.  The base ``metrics.py`` handles
  sklearn-dependent metrics.
* **Dataclass results** — Every metric function returns a typed
  dataclass rather than loose dicts, enabling IDE autocompletion and
  type checking throughout the evaluation pipeline.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------


@dataclass
class KappaResult:
    """Cohen's Kappa agreement statistics.

    Attributes
    ----------
    kappa:
        Cohen's Kappa coefficient (−1 to 1).  1 = perfect agreement,
        0 = chance agreement, <0 = worse than chance.
    observed_agreement:
        Proportion of items where raters agree.
    expected_agreement:
        Agreement expected by chance given the marginal distributions.
    n_items:
        Total number of rated items.
    """

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_items: int


def compute_cohens_kappa(
    rater_a: Sequence[str | int],
    rater_b: Sequence[str | int],
) -> KappaResult:
    """Compute Cohen's Kappa for two raters on categorical labels.

    Parameters
    ----------
    rater_a:
        Labels assigned by rater A.
    rater_b:
        Labels assigned by rater B (same length as *rater_a*).

    Returns
    -------
    KappaResult
        Kappa coefficient with supporting statistics.

    Raises
    ------
    ValueError
        If the inputs have different lengths or are empty.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"Rater sequences must have equal length "
            f"({len(rater_a)} != {len(rater_b)})"
        )
    n = len(rater_a)
    if n == 0:
        raise ValueError("Cannot compute kappa on empty sequences")

    # Build confusion counts.
    labels = sorted(set(rater_a) | set(rater_b))
    label_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    matrix = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(rater_a, rater_b, strict=False):
        matrix[label_idx[a], label_idx[b]] += 1

    observed = float(np.trace(matrix)) / n

    # Expected agreement under independence.
    row_marginals = matrix.sum(axis=1).astype(np.float64) / n
    col_marginals = matrix.sum(axis=0).astype(np.float64) / n
    expected = float(np.dot(row_marginals, col_marginals))

    kappa = 1.0 if abs(1.0 - expected) < 1e-12 else (observed - expected) / (1.0 - expected)

    return KappaResult(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n_items=n,
    )


# ---------------------------------------------------------------------------
# Matthews Correlation Coefficient (MCC)
# ---------------------------------------------------------------------------


@dataclass
class MCCResult:
    """Matthews Correlation Coefficient result.

    Attributes
    ----------
    mcc:
        The coefficient (−1 to 1).  +1 = perfect, 0 = random, −1 = inverse.
    tp, fp, fn, tn:
        Confusion matrix components used in the calculation.
    """

    mcc: float
    tp: int
    fp: int
    fn: int
    tn: int


def compute_mcc(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> MCCResult:
    """Compute Matthews Correlation Coefficient for binary classification.

    Parameters
    ----------
    y_true:
        Ground truth binary labels (0 or 1).
    y_pred:
        Predicted binary labels (0 or 1).

    Returns
    -------
    MCCResult
        MCC value with confusion matrix breakdown.

    Raises
    ------
    ValueError
        If inputs have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input sequences must have equal length")

    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=False):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return MCCResult(mcc=mcc, tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Model confidence calibration metrics.

    Attributes
    ----------
    expected_calibration_error:
        ECE — weighted average of per-bin |accuracy − confidence|.
        Lower is better; 0 = perfectly calibrated.
    brier_score:
        Mean squared error between predicted probabilities and true
        binary outcomes.  Lower is better; 0 = perfect.
    bin_accuracies:
        Per-bin accuracy values (for reliability diagrams).
    bin_confidences:
        Per-bin mean confidence values.
    bin_counts:
        Number of samples in each bin.
    n_bins:
        Number of calibration bins used.
    """

    expected_calibration_error: float
    brier_score: float
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    n_bins: int


def compute_calibration(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute Expected Calibration Error (ECE) and Brier score.

    Parameters
    ----------
    y_true:
        Ground truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class (0.0 to 1.0).
    n_bins:
        Number of equal-width bins for the reliability diagram.

    Returns
    -------
    CalibrationResult
        Calibration statistics with per-bin breakdowns.

    Raises
    ------
    ValueError
        If inputs have different lengths or n_bins < 1.
    """
    if len(y_true) != len(y_prob):
        raise ValueError("Input sequences must have equal length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    n = len(y_true)
    if n == 0:
        return CalibrationResult(
            expected_calibration_error=0.0,
            brier_score=0.0,
            bin_accuracies=[],
            bin_confidences=[],
            bin_counts=[],
            n_bins=n_bins,
        )

    # Brier score — mean squared error of probabilities.
    brier = sum((p - t) ** 2 for t, p in zip(y_true, y_prob, strict=False)) / n

    # Bin predictions into equal-width intervals.
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_cnts: list[int] = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # Include right boundary for the last bin.
        indices = [
            j
            for j in range(n)
            if (lo <= y_prob[j] < hi) or (i == n_bins - 1 and y_prob[j] == hi)
        ]
        count = len(indices)
        bin_cnts.append(count)

        if count == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            continue

        acc = sum(y_true[j] for j in indices) / count
        conf = sum(y_prob[j] for j in indices) / count
        bin_accs.append(acc)
        bin_confs.append(conf)
        ece += (count / n) * abs(acc - conf)

    return CalibrationResult(
        expected_calibration_error=ece,
        brier_score=brier,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_cnts,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Partial NER Span Matching
# ---------------------------------------------------------------------------


@dataclass
class PartialNERMatch:
    """Result of a single partial span comparison.

    Attributes
    ----------
    pred_entity:
        The predicted entity span (type, start, end).
    gold_entity:
        The matched gold entity span, or None if unmatched.
    overlap_score:
        Jaccard overlap between the two spans (0.0 to 1.0).
    type_match:
        Whether entity types agree.
    """

    pred_entity: tuple[str, int, int]
    gold_entity: tuple[str, int, int] | None
    overlap_score: float
    type_match: bool


@dataclass
class PartialNERMetrics:
    """Aggregated partial span-matching NER metrics.

    Attributes
    ----------
    exact_f1:
        Standard exact-match F1 (for comparison).
    partial_f1:
        F1 using partial span credit.
    type_weighted_f1:
        F1 weighting partial matches by type correctness.
    mean_overlap:
        Average Jaccard overlap across matched pairs.
    n_gold:
        Total gold entities.
    n_pred:
        Total predicted entities.
    n_exact_matches:
        Entities matching both type and exact span.
    n_partial_matches:
        Entities with >0 span overlap but not exact.
    n_unmatched_pred:
        Predicted entities with no overlapping gold entity.
    n_unmatched_gold:
        Gold entities with no overlapping prediction.
    """

    exact_f1: float
    partial_f1: float
    type_weighted_f1: float
    mean_overlap: float
    n_gold: int
    n_pred: int
    n_exact_matches: int
    n_partial_matches: int
    n_unmatched_pred: int
    n_unmatched_gold: int


def _span_jaccard(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """Compute Jaccard overlap between two character spans.

    Parameters
    ----------
    start_a, end_a:
        Character offsets of span A.
    start_b, end_b:
        Character offsets of span B.

    Returns
    -------
    float
        Jaccard index (intersection / union) in [0.0, 1.0].
    """
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union if union > 0 else 0.0


def compute_partial_ner_metrics(
    gold_entities: list[tuple[str, int, int]],
    pred_entities: list[tuple[str, int, int]],
    *,
    overlap_threshold: float = 0.5,
) -> PartialNERMetrics:
    """Evaluate NER with partial span-matching credit.

    Each predicted entity is greedily matched to the gold entity with
    highest Jaccard overlap (above *overlap_threshold*).  Partial credit
    is the Jaccard score; type mismatches halve the credit.

    Parameters
    ----------
    gold_entities:
        List of (entity_type, start_char, end_char) gold annotations.
    pred_entities:
        List of (entity_type, start_char, end_char) predictions.
    overlap_threshold:
        Minimum Jaccard to count as a partial match.

    Returns
    -------
    PartialNERMetrics
        Aggregated partial matching statistics.
    """
    n_gold = len(gold_entities)
    n_pred = len(pred_entities)

    if n_gold == 0 and n_pred == 0:
        return PartialNERMetrics(
            exact_f1=1.0,
            partial_f1=1.0,
            type_weighted_f1=1.0,
            mean_overlap=1.0,
            n_gold=0,
            n_pred=0,
            n_exact_matches=0,
            n_partial_matches=0,
            n_unmatched_pred=0,
            n_unmatched_gold=0,
        )

    if n_gold == 0 or n_pred == 0:
        return PartialNERMetrics(
            exact_f1=0.0,
            partial_f1=0.0,
            type_weighted_f1=0.0,
            mean_overlap=0.0,
            n_gold=n_gold,
            n_pred=n_pred,
            n_exact_matches=0,
            n_partial_matches=0,
            n_unmatched_pred=n_pred,
            n_unmatched_gold=n_gold,
        )

    # Greedy matching: for each pred, find best overlapping gold.
    used_gold: set[int] = set()
    overlaps: list[float] = []
    type_weighted_scores: list[float] = []
    exact = 0
    partial = 0

    for p_type, p_start, p_end in pred_entities:
        best_j = -1
        best_score = 0.0
        for j, (_g_type, g_start, g_end) in enumerate(gold_entities):
            if j in used_gold:
                continue
            score = _span_jaccard(p_start, p_end, g_start, g_end)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j >= 0 and best_score >= overlap_threshold:
            used_gold.add(best_j)
            overlaps.append(best_score)
            g_type = gold_entities[best_j][0]
            type_ok = p_type == g_type
            type_weighted_scores.append(best_score if type_ok else best_score * 0.5)

            if best_score >= 1.0 - 1e-9 and p_type == g_type:
                exact += 1
            else:
                partial += 1
        else:
            overlaps.append(0.0)
            type_weighted_scores.append(0.0)

    n_unmatched_pred = n_pred - exact - partial
    n_unmatched_gold = n_gold - len(used_gold)

    # Partial precision/recall/F1: credit = overlap score.
    partial_precision = sum(overlaps) / n_pred if n_pred > 0 else 0.0
    partial_recall = sum(overlaps) / n_gold if n_gold > 0 else 0.0
    partial_f1 = (
        2 * partial_precision * partial_recall / (partial_precision + partial_recall)
        if (partial_precision + partial_recall) > 0
        else 0.0
    )

    # Type-weighted F1.
    tw_precision = sum(type_weighted_scores) / n_pred if n_pred > 0 else 0.0
    tw_recall = sum(type_weighted_scores) / n_gold if n_gold > 0 else 0.0
    tw_f1 = (
        2 * tw_precision * tw_recall / (tw_precision + tw_recall)
        if (tw_precision + tw_recall) > 0
        else 0.0
    )

    # Exact F1 for comparison.
    exact_prec = exact / n_pred if n_pred > 0 else 0.0
    exact_rec = exact / n_gold if n_gold > 0 else 0.0
    exact_f1 = (
        2 * exact_prec * exact_rec / (exact_prec + exact_rec)
        if (exact_prec + exact_rec) > 0
        else 0.0
    )

    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    return PartialNERMetrics(
        exact_f1=exact_f1,
        partial_f1=partial_f1,
        type_weighted_f1=tw_f1,
        mean_overlap=mean_overlap,
        n_gold=n_gold,
        n_pred=n_pred,
        n_exact_matches=exact,
        n_partial_matches=partial,
        n_unmatched_pred=n_unmatched_pred,
        n_unmatched_gold=n_unmatched_gold,
    )


# ---------------------------------------------------------------------------
# ROUGE F-measure (full precision/recall/F1)
# ---------------------------------------------------------------------------


@dataclass
class ROUGEScore:
    """Full ROUGE score with precision, recall, and F1.

    Attributes
    ----------
    precision:
        Fraction of hypothesis n-grams found in reference.
    recall:
        Fraction of reference n-grams found in hypothesis.
    f1:
        Harmonic mean of precision and recall.
    """

    precision: float
    recall: float
    f1: float


@dataclass
class ROUGEResult:
    """Complete ROUGE evaluation result.

    Attributes
    ----------
    rouge1:
        Unigram overlap scores.
    rouge2:
        Bigram overlap scores.
    rougeL:
        Longest Common Subsequence scores.
    reference_length:
        Word count of the reference text.
    hypothesis_length:
        Word count of the hypothesis text.
    length_ratio:
        hypothesis_length / reference_length (compression ratio).
    """

    rouge1: ROUGEScore
    rouge2: ROUGEScore
    rougeL: ROUGEScore
    reference_length: int
    hypothesis_length: int
    length_ratio: float


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Build n-gram frequency counter from token list."""
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(ref_tokens: list[str], hyp_tokens: list[str], n: int) -> ROUGEScore:
    """Compute ROUGE-N precision, recall, and F1."""
    ref_ngrams = _ngram_counts(ref_tokens, n)
    hyp_ngrams = _ngram_counts(hyp_tokens, n)

    if not ref_ngrams or not hyp_ngrams:
        return ROUGEScore(precision=0.0, recall=0.0, f1=0.0)

    # Clipped overlap: min of counts per n-gram.
    overlap = sum(
        min(hyp_ngrams[ng], ref_ngrams[ng])
        for ng in hyp_ngrams
        if ng in ref_ngrams
    )

    precision = overlap / sum(hyp_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ROUGEScore(precision=precision, recall=recall, f1=f1)


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of longest common subsequence via DP."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l(ref_tokens: list[str], hyp_tokens: list[str]) -> ROUGEScore:
    """Compute ROUGE-L precision, recall, and F1 via LCS."""
    if not ref_tokens or not hyp_tokens:
        return ROUGEScore(precision=0.0, recall=0.0, f1=0.0)

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ROUGEScore(precision=precision, recall=recall, f1=f1)


def compute_rouge(reference: str, hypothesis: str) -> ROUGEResult:
    """Compute full ROUGE-1/2/L with precision, recall, and F1.

    Unlike the simpler recall-only ``compute_rouge_scores`` in
    ``metrics.py``, this function returns the complete triplet for
    each ROUGE variant, enabling balanced summarisation assessment.

    Parameters
    ----------
    reference:
        Gold-standard reference summary.
    hypothesis:
        Model-generated summary.

    Returns
    -------
    ROUGEResult
        Full ROUGE evaluation with length statistics.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    ratio = hyp_len / ref_len if ref_len > 0 else 0.0

    return ROUGEResult(
        rouge1=_rouge_n(ref_tokens, hyp_tokens, 1),
        rouge2=_rouge_n(ref_tokens, hyp_tokens, 2),
        rougeL=_rouge_l(ref_tokens, hyp_tokens),
        reference_length=ref_len,
        hypothesis_length=hyp_len,
        length_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Hierarchical ICD-10 Evaluation
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalICDMetrics:
    """ICD-10 evaluation at multiple hierarchy levels.

    ICD-10 codes have structure: X##.### where:
    - Chapter: first letter group (e.g., E = Endocrine)
    - Block/3-char: first 3 characters (e.g., E11 = Type 2 DM)
    - Full code: complete code (e.g., E11.65)

    A prediction of E11.9 when gold is E11.65 is wrong at the full-code
    level but correct at the 3-character and chapter levels.  Reporting
    all three levels reveals whether errors are catastrophic (wrong
    organ system) or merely specificity mismatches.

    Attributes
    ----------
    full_code_accuracy:
        Exact full-code match rate.
    block_accuracy:
        Match rate at the 3-character block level.
    chapter_accuracy:
        Match rate at the chapter (first character) level.
    n_samples:
        Total number of code pairs evaluated.
    full_code_matches:
        Count of exact full-code matches.
    block_matches:
        Count of 3-character block matches.
    chapter_matches:
        Count of chapter-level matches.
    """

    full_code_accuracy: float
    block_accuracy: float
    chapter_accuracy: float
    n_samples: int
    full_code_matches: int
    block_matches: int
    chapter_matches: int


def compute_hierarchical_icd_metrics(
    gold_codes: Sequence[str],
    pred_codes: Sequence[str],
) -> HierarchicalICDMetrics:
    """Evaluate ICD-10 predictions at chapter, block, and full-code levels.

    Parameters
    ----------
    gold_codes:
        Ground truth ICD-10-CM codes.
    pred_codes:
        Predicted ICD-10-CM codes (same length as *gold_codes*).

    Returns
    -------
    HierarchicalICDMetrics
        Accuracy at each hierarchy level.

    Raises
    ------
    ValueError
        If inputs have different lengths.
    """
    if len(gold_codes) != len(pred_codes):
        raise ValueError("Code sequences must have equal length")

    n = len(gold_codes)
    if n == 0:
        return HierarchicalICDMetrics(
            full_code_accuracy=0.0,
            block_accuracy=0.0,
            chapter_accuracy=0.0,
            n_samples=0,
            full_code_matches=0,
            block_matches=0,
            chapter_matches=0,
        )

    full_matches = 0
    block_matches = 0
    chapter_matches = 0

    for gold, pred in zip(gold_codes, pred_codes, strict=False):
        g_norm = gold.strip().upper()
        p_norm = pred.strip().upper()

        # Full code comparison (normalise by removing dots for consistency).
        if g_norm.replace(".", "") == p_norm.replace(".", ""):
            full_matches += 1

        # Block: first 3 characters (e.g., E11).
        if len(g_norm) >= 3 and len(p_norm) >= 3 and g_norm[:3] == p_norm[:3]:
            block_matches += 1

        # Chapter: first character (e.g., E).
        if g_norm and p_norm and g_norm[0] == p_norm[0]:
            chapter_matches += 1

    return HierarchicalICDMetrics(
        full_code_accuracy=full_matches / n,
        block_accuracy=block_matches / n,
        chapter_accuracy=chapter_matches / n,
        n_samples=n,
        full_code_matches=full_matches,
        block_matches=block_matches,
        chapter_matches=chapter_matches,
    )


# ---------------------------------------------------------------------------
# Per-class AUPRC (Area Under Precision-Recall Curve)
# ---------------------------------------------------------------------------


@dataclass
class AUPRCResult:
    """Area Under the Precision-Recall Curve for a single class.

    Attributes
    ----------
    label:
        Class label or name.
    auprc:
        Area under the precision-recall curve (0.0 to 1.0).
        Computed via trapezoidal integration.
    n_positive:
        Number of positive examples for this class.
    n_total:
        Total number of examples.
    """

    label: str
    auprc: float
    n_positive: int
    n_total: int


def compute_auprc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    label: str = "positive",
) -> AUPRCResult:
    """Compute Area Under the Precision-Recall Curve.

    Uses threshold-based calculation: sort by descending score, sweep
    threshold, compute precision and recall at each unique score, then
    integrate via the trapezoidal rule.

    Parameters
    ----------
    y_true:
        Binary ground truth labels (0 or 1).
    y_scores:
        Predicted scores/probabilities for the positive class.
    label:
        Human-readable label name for the result.

    Returns
    -------
    AUPRCResult
        AUPRC value with class statistics.
    """
    n = len(y_true)
    n_pos = sum(y_true)

    if n == 0 or n_pos == 0:
        return AUPRCResult(label=label, auprc=0.0, n_positive=n_pos, n_total=n)

    # Sort by descending score.
    pairs = sorted(zip(y_scores, y_true, strict=False), key=lambda x: -x[0])

    precisions = [1.0]
    recalls = [0.0]
    tp = 0

    for i, (_score, truth) in enumerate(pairs, 1):
        if truth == 1:
            tp += 1
        prec = tp / i
        rec = tp / n_pos
        precisions.append(prec)
        recalls.append(rec)

    # Trapezoidal integration.
    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2

    return AUPRCResult(label=label, auprc=auprc, n_positive=n_pos, n_total=n)

