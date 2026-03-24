"""Data drift detection for the ClinIQ ML platform.

Implements Population Stability Index (PSI)-based drift detection for both
input text distributions and model prediction distributions.  All monitors
operate on an in-memory ring buffer so they add no external dependencies.

PSI thresholds (industry standard):
  - PSI < 0.1  : No significant drift
  - PSI < 0.2  : Moderate drift — worth investigating
  - PSI >= 0.2 : Significant drift — model may need retraining
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# PSI alert thresholds
PSI_WARNING_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.2

# Medical abbreviations used to estimate clinical density
_MEDICAL_ABBREVIATIONS = re.compile(
    r"\b(?:pt|hx|dx|rx|prn|bid|tid|qid|qhs|cc|hpi|pmh|ros|vs|bp|hr|rr|"
    r"o2|spo2|bmi|bun|cr|na|k|cl|co2|wbc|rbc|hgb|hct|plt|inr|ptt|"
    r"mcg|mg|ml|iv|im|po|sq|gt|gtt|tpn|npo|icu|ed|er|or|pacu|snf|ltac)\b",
    re.IGNORECASE,
)

# Common clinical section headings
_SECTION_PATTERNS = {
    "chief_complaint": re.compile(r"\b(?:chief complaint|cc:|reason for visit)\b", re.IGNORECASE),
    "history": re.compile(r"\b(?:history of present illness|hpi:|past medical history|pmh:)\b", re.IGNORECASE),
    "medications": re.compile(r"\b(?:medications?:|current medications?|meds:)\b", re.IGNORECASE),
    "assessment": re.compile(r"\b(?:assessment|impression|diagnosis|diagnoses)\b", re.IGNORECASE),
    "plan": re.compile(r"\b(?:plan:|treatment plan|recommendations?:)\b", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Summary of a drift detection computation.

    Attributes
    ----------
    is_drifted:
        ``True`` when the overall PSI score exceeds the warning threshold.
    drift_score:
        Aggregate PSI score across all monitored features.
    feature_drifts:
        Per-feature PSI scores and drift status.
    reference_stats:
        Statistical summary of the reference (historical) distribution.
    current_stats:
        Statistical summary of the recent (current) distribution.
    timestamp:
        UTC timestamp when this report was generated.
    window_size:
        Number of recent samples used for the current distribution.
    """

    is_drifted: bool
    drift_score: float
    feature_drifts: dict[str, dict[str, Any]]
    reference_stats: dict[str, Any]
    current_stats: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    window_size: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "is_drifted": self.is_drifted,
            "drift_score": round(self.drift_score, 6),
            "feature_drifts": self.feature_drifts,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "timestamp": self.timestamp.isoformat(),
            "window_size": self.window_size,
        }


# ---------------------------------------------------------------------------
# Population Stability Index helpers
# ---------------------------------------------------------------------------


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute the Population Stability Index between two 1-D distributions.

    Both arrays are binned using the reference quantiles so they share the
    same bucket boundaries.

    Parameters
    ----------
    reference:
        Historical distribution values (baseline).
    current:
        Recent distribution values (to compare against baseline).
    n_bins:
        Number of histogram bins.
    epsilon:
        Small constant added to proportions to avoid log(0).

    Returns
    -------
    float
        PSI score.  Values above :data:`PSI_WARNING_THRESHOLD` indicate drift.
    """
    if len(reference) < 2 or len(current) < 2:
        return 0.0

    # Determine bin edges from the reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    if max_val == min_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)

    # Convert to proportions
    ref_prop = ref_hist / (ref_hist.sum() + epsilon) + epsilon
    cur_prop = cur_hist / (cur_hist.sum() + epsilon) + epsilon

    # PSI = sum((actual% - expected%) * ln(actual% / expected%))
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return max(psi, 0.0)


def _compute_categorical_psi(
    reference: list[str],
    current: list[str],
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for a categorical distribution (e.g. ICD chapter labels).

    Parameters
    ----------
    reference:
        Historical categorical values.
    current:
        Recent categorical values.
    epsilon:
        Smoothing constant.

    Returns
    -------
    float
        PSI score.
    """
    all_categories = set(reference) | set(current)

    ref_total = len(reference) + epsilon
    cur_total = len(current) + epsilon

    psi = 0.0
    for cat in all_categories:
        ref_p = (reference.count(cat) / ref_total) + epsilon
        cur_p = (current.count(cat) / cur_total) + epsilon
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)

    return max(float(psi), 0.0)


# ---------------------------------------------------------------------------
# Text distribution monitor
# ---------------------------------------------------------------------------


class TextDistributionMonitor:
    """Track input text distribution metrics over time.

    Records per-document statistics and detects distributional shift between
    a reference window (the first *reference_size* documents) and a sliding
    current window.

    Parameters
    ----------
    reference_size:
        Number of initial documents to use as the reference distribution.
        Once this many documents have been tracked, the reference is frozen.
    max_history:
        Maximum number of recent records to keep in memory.

    Tracked features
    ----------------
    - ``doc_length``       : character count
    - ``word_count``       : whitespace-separated token count
    - ``vocab_diversity``  : type-token ratio (unique words / total words)
    - ``abbrev_density``   : medical abbreviation count / word count
    - ``section_count``    : number of recognised clinical section headings
    """

    def __init__(self, reference_size: int = 200, max_history: int = 5000) -> None:
        self.reference_size = reference_size
        self.max_history = max_history
        self._records: deque[dict[str, float]] = deque(maxlen=max_history)
        self._reference_records: list[dict[str, float]] = []
        self._reference_frozen: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def track(self, text: str) -> None:
        """Record statistics for a single document.

        Parameters
        ----------
        text:
            Raw clinical document text.
        """
        stats = self._extract_stats(text)
        self._records.append(stats)

        if not self._reference_frozen:
            self._reference_records.append(stats)
            if len(self._reference_records) >= self.reference_size:
                self._reference_frozen = True
                logger.info(
                    "TextDistributionMonitor: reference distribution frozen "
                    "at %d documents",
                    len(self._reference_records),
                )

    def compute_drift(self, window_size: int = 100) -> DriftReport:
        """Compare the most recent *window_size* documents to the reference.

        Parameters
        ----------
        window_size:
            Number of recent documents to use as the "current" distribution.

        Returns
        -------
        DriftReport
        """
        if not self._reference_records:
            logger.warning("No reference data available — cannot compute drift")
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                feature_drifts={},
                reference_stats={},
                current_stats={},
                window_size=window_size,
            )

        records_list = list(self._records)
        current_records = records_list[-window_size:] if len(records_list) >= window_size else records_list

        if not current_records:
            return DriftReport(
                is_drifted=False,
                drift_score=0.0,
                feature_drifts={},
                reference_stats=self._summarise(self._reference_records),
                current_stats={},
                window_size=window_size,
            )

        feature_drifts: dict[str, dict[str, Any]] = {}
        psi_scores: list[float] = []

        numeric_features = ["doc_length", "word_count", "vocab_diversity", "abbrev_density", "section_count"]

        for feat in numeric_features:
            ref_vals = np.array([r[feat] for r in self._reference_records])
            cur_vals = np.array([r[feat] for r in current_records])
            psi = _compute_psi(ref_vals, cur_vals)
            psi_scores.append(psi)

            feature_drifts[feat] = {
                "psi": round(psi, 6),
                "is_drifted": psi >= PSI_WARNING_THRESHOLD,
                "severity": _classify_psi(psi),
                "reference_mean": round(float(ref_vals.mean()), 4),
                "current_mean": round(float(cur_vals.mean()), 4),
            }

        overall_psi = float(np.mean(psi_scores)) if psi_scores else 0.0

        return DriftReport(
            is_drifted=overall_psi >= PSI_WARNING_THRESHOLD,
            drift_score=round(overall_psi, 6),
            feature_drifts=feature_drifts,
            reference_stats=self._summarise(self._reference_records),
            current_stats=self._summarise(current_records),
            window_size=window_size,
        )

    @property
    def n_tracked(self) -> int:
        """Total number of documents tracked so far."""
        return len(self._records)

    @property
    def reference_frozen(self) -> bool:
        """Whether the reference distribution has been locked in."""
        return self._reference_frozen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_stats(self, text: str) -> dict[str, float]:
        """Extract numeric statistics from a single document."""
        doc_length = float(len(text))
        words = text.split()
        word_count = float(len(words))

        # Type-token ratio (capped to avoid noise on very short docs)
        if word_count > 0:
            unique_words = len({w.lower() for w in words})
            vocab_diversity = unique_words / word_count
        else:
            vocab_diversity = 0.0

        # Medical abbreviation density
        abbrev_matches = _MEDICAL_ABBREVIATIONS.findall(text)
        abbrev_density = len(abbrev_matches) / max(word_count, 1)

        # Section heading count
        section_count = float(
            sum(1 for pat in _SECTION_PATTERNS.values() if pat.search(text))
        )

        return {
            "doc_length": doc_length,
            "word_count": word_count,
            "vocab_diversity": round(vocab_diversity, 6),
            "abbrev_density": round(abbrev_density, 6),
            "section_count": section_count,
        }

    def _summarise(self, records: list[dict[str, float]]) -> dict[str, Any]:
        """Produce mean/std summary statistics over a list of record dicts."""
        if not records:
            return {}
        summary: dict[str, Any] = {}
        keys = records[0].keys()
        for key in keys:
            vals = np.array([r[key] for r in records])
            summary[key] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "n": len(vals),
            }
        return summary


# ---------------------------------------------------------------------------
# Prediction distribution monitor
# ---------------------------------------------------------------------------


class PredictionMonitor:
    """Track model prediction distributions and detect output drift.

    Separately monitors confidence score distributions and label/code
    distributions so clinicians receive targeted alerts about what kind of
    drift is occurring.

    Parameters
    ----------
    alert_threshold:
        PSI threshold above which a drift alert is raised.  Defaults to
        ``0.1`` (warning level).  Raise to ``0.2`` for critical-only alerts.
    reference_size:
        Minimum number of predictions required to freeze the reference.
    max_history:
        Maximum number of prediction records kept in memory.
    """

    alert_threshold: float = PSI_WARNING_THRESHOLD

    def __init__(
        self,
        alert_threshold: float = PSI_WARNING_THRESHOLD,
        reference_size: int = 500,
        max_history: int = 10000,
    ) -> None:
        self.alert_threshold = alert_threshold
        self.reference_size = reference_size
        self.max_history = max_history

        # Records keyed by model_name
        self._confidence_records: dict[str, deque[float]] = {}
        self._label_records: dict[str, deque[str]] = {}
        self._reference_confidences: dict[str, list[float]] = {}
        self._reference_labels: dict[str, list[str]] = {}
        self._reference_frozen: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def track_prediction(
        self,
        model_name: str,
        predictions: list[str] | list[dict[str, Any]],
        confidence: float | list[float],
    ) -> None:
        """Record a prediction event.

        Parameters
        ----------
        model_name:
            Logical model identifier (e.g. ``"sklearn-baseline"``).
        predictions:
            List of predicted labels/codes, or list of dicts with a
            ``"code"`` key (as returned by
            :meth:`~app.ml.icd.model.ICDCodePrediction.to_dict`).
        confidence:
            Scalar or per-prediction confidence score(s).
        """
        self._ensure_model_buffers(model_name)

        # Normalise confidence to a scalar (average for multi-label)
        if isinstance(confidence, list):
            conf_value = float(np.mean(confidence)) if confidence else 0.0
        else:
            conf_value = float(confidence)

        self._confidence_records[model_name].append(conf_value)

        # Normalise predictions to label strings
        labels: list[str] = []
        for pred in predictions:
            if isinstance(pred, dict):
                labels.append(str(pred.get("code", "UNKNOWN")))
            else:
                labels.append(str(pred))

        self._label_records[model_name].extend(labels)

        # Populate reference until frozen
        if not self._reference_frozen.get(model_name, False):
            self._reference_confidences[model_name].append(conf_value)
            self._reference_labels[model_name].extend(labels)

            n_conf = len(self._reference_confidences[model_name])
            if n_conf >= self.reference_size:
                self._reference_frozen[model_name] = True
                logger.info(
                    "PredictionMonitor: reference frozen for '%s' at %d predictions",
                    model_name,
                    n_conf,
                )

    def detect_confidence_drift(self, window_size: int = 100, model_name: str | None = None) -> DriftReport:
        """Detect drift in confidence score distribution.

        Parameters
        ----------
        window_size:
            Number of recent predictions to compare against the reference.
        model_name:
            Model to analyse.  If ``None`` the first registered model is used.

        Returns
        -------
        DriftReport
        """
        model_name = self._resolve_model(model_name)
        if model_name is None:
            return self._empty_report(window_size)

        ref = self._reference_confidences.get(model_name, [])
        recent_deque = self._confidence_records.get(model_name, deque())
        recent = list(recent_deque)[-window_size:]

        if not ref or not recent:
            return self._empty_report(window_size)

        ref_arr = np.array(ref)
        cur_arr = np.array(recent)
        psi = _compute_psi(ref_arr, cur_arr)

        feature_drifts = {
            "confidence": {
                "psi": round(psi, 6),
                "is_drifted": psi >= self.alert_threshold,
                "severity": _classify_psi(psi),
                "reference_mean": round(float(ref_arr.mean()), 4),
                "current_mean": round(float(cur_arr.mean()), 4),
                "reference_std": round(float(ref_arr.std()), 4),
                "current_std": round(float(cur_arr.std()), 4),
            }
        }

        return DriftReport(
            is_drifted=psi >= self.alert_threshold,
            drift_score=round(psi, 6),
            feature_drifts=feature_drifts,
            reference_stats={"confidence": {"mean": round(float(ref_arr.mean()), 4), "n": len(ref)}},
            current_stats={"confidence": {"mean": round(float(cur_arr.mean()), 4), "n": len(recent)}},
            window_size=window_size,
        )

    def detect_prediction_drift(self, window_size: int = 100, model_name: str | None = None) -> DriftReport:
        """Detect drift in predicted label/code distribution.

        Parameters
        ----------
        window_size:
            Number of recent label records to compare against the reference.
        model_name:
            Model to analyse.  If ``None`` the first registered model is used.

        Returns
        -------
        DriftReport
        """
        model_name = self._resolve_model(model_name)
        if model_name is None:
            return self._empty_report(window_size)

        ref_labels = self._reference_labels.get(model_name, [])
        recent_deque = self._label_records.get(model_name, deque())
        recent_labels = list(recent_deque)[-window_size:]

        if not ref_labels or not recent_labels:
            return self._empty_report(window_size)

        psi = _compute_categorical_psi(ref_labels, recent_labels)

        # Top-label frequency for context
        from collections import Counter

        ref_top = Counter(ref_labels).most_common(5)
        cur_top = Counter(recent_labels).most_common(5)

        feature_drifts = {
            "label_distribution": {
                "psi": round(psi, 6),
                "is_drifted": psi >= self.alert_threshold,
                "severity": _classify_psi(psi),
                "reference_top_labels": ref_top,
                "current_top_labels": cur_top,
            }
        }

        ref_stats = {
            "n_labels": len(ref_labels),
            "n_unique": len(set(ref_labels)),
            "top_labels": ref_top,
        }
        cur_stats = {
            "n_labels": len(recent_labels),
            "n_unique": len(set(recent_labels)),
            "top_labels": cur_top,
        }

        return DriftReport(
            is_drifted=psi >= self.alert_threshold,
            drift_score=round(psi, 6),
            feature_drifts=feature_drifts,
            reference_stats=ref_stats,
            current_stats=cur_stats,
            window_size=window_size,
        )

    @property
    def registered_models(self) -> list[str]:
        """List of model names that have received at least one prediction."""
        return list(self._confidence_records.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model_buffers(self, model_name: str) -> None:
        """Initialise per-model deques and reference lists on first use."""
        if model_name not in self._confidence_records:
            self._confidence_records[model_name] = deque(maxlen=self.max_history)
            self._label_records[model_name] = deque(maxlen=self.max_history)
            self._reference_confidences[model_name] = []
            self._reference_labels[model_name] = []
            self._reference_frozen[model_name] = False

    def _resolve_model(self, model_name: str | None) -> str | None:
        """Return *model_name* or the first registered model when ``None``."""
        if model_name is not None:
            return model_name
        models = self.registered_models
        return models[0] if models else None

    def _empty_report(self, window_size: int) -> DriftReport:
        """Return an empty no-drift report when insufficient data is available."""
        return DriftReport(
            is_drifted=False,
            drift_score=0.0,
            feature_drifts={},
            reference_stats={},
            current_stats={},
            window_size=window_size,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _classify_psi(psi: float) -> str:
    """Map a PSI score to a human-readable severity string."""
    if psi < PSI_WARNING_THRESHOLD:
        return "stable"
    if psi < PSI_CRITICAL_THRESHOLD:
        return "warning"
    return "critical"
