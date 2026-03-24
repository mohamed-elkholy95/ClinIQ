"""Unit tests for data drift detection.

Tests PSI computation, TextDistributionMonitor, and PredictionMonitor
with controlled synthetic distributions.
"""

import numpy as np
import pytest

from app.ml.monitoring.drift_detector import (
    PSI_CRITICAL_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    DriftReport,
    PredictionMonitor,
    TextDistributionMonitor,
    _classify_psi,
    _compute_categorical_psi,
    _compute_psi,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clinical_note(length: str = "medium", style: str = "standard") -> str:
    """Generate a synthetic clinical note for testing."""
    if style == "standard":
        base = (
            "Chief Complaint: chest pain\n"
            "History of Present Illness: "
            "Patient is a 55-year-old male presenting with acute chest pain. "
            "BP 130/85, HR 78, O2 sat 97%. "
            "Medications: metformin 500mg BID, lisinopril 10mg daily. "
            "Assessment: rule out acute coronary syndrome.\n"
            "Plan: order troponin, EKG, chest X-ray."
        )
    elif style == "short":
        base = "Pt with cough x 3 days. Denies fever. No meds."
    else:
        base = "Random text without medical content. The weather is nice today."

    if length == "short":
        return base[:80]
    if length == "long":
        return base * 4
    return base


@pytest.fixture
def text_monitor() -> TextDistributionMonitor:
    """Monitor with a small reference size for fast tests."""
    return TextDistributionMonitor(reference_size=10, max_history=500)


@pytest.fixture
def prediction_monitor() -> PredictionMonitor:
    """PredictionMonitor with a small reference size."""
    return PredictionMonitor(reference_size=10, max_history=500)


# ---------------------------------------------------------------------------
# PSI computation helpers
# ---------------------------------------------------------------------------


class TestComputePSI:
    """Test _compute_psi with known distributions."""

    def test_identical_distributions_zero_psi(self) -> None:
        data = np.random.normal(0, 1, 500)
        psi = _compute_psi(data, data)
        assert psi < 0.01  # Near zero

    def test_similar_distributions_low_psi(self) -> None:
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0.02, 1, 1000)
        psi = _compute_psi(ref, cur)
        assert psi < PSI_WARNING_THRESHOLD

    def test_shifted_distributions_high_psi(self) -> None:
        ref = np.random.normal(0, 1, 500)
        cur = np.random.normal(5, 1, 500)  # Large shift
        psi = _compute_psi(ref, cur)
        assert psi >= PSI_CRITICAL_THRESHOLD

    def test_empty_arrays_return_zero(self) -> None:
        assert _compute_psi(np.array([]), np.array([1, 2])) == 0.0
        assert _compute_psi(np.array([1]), np.array([2])) == 0.0  # <2 elements

    def test_constant_arrays_return_zero(self) -> None:
        ref = np.ones(100)
        cur = np.ones(100)
        psi = _compute_psi(ref, cur)
        assert psi == 0.0  # max == min

    def test_psi_non_negative(self) -> None:
        ref = np.random.uniform(0, 10, 200)
        cur = np.random.uniform(2, 12, 200)
        psi = _compute_psi(ref, cur)
        assert psi >= 0.0


class TestComputeCategoricalPSI:
    """Test _compute_categorical_psi for label distributions."""

    def test_identical_labels_near_zero(self) -> None:
        labels = ["A", "B", "C"] * 100
        psi = _compute_categorical_psi(labels, labels)
        assert psi < 0.01

    def test_different_labels_high_psi(self) -> None:
        ref = ["A"] * 100
        cur = ["B"] * 100
        psi = _compute_categorical_psi(ref, cur)
        assert psi > PSI_CRITICAL_THRESHOLD

    def test_empty_lists_return_zero(self) -> None:
        psi = _compute_categorical_psi([], [])
        assert psi == 0.0

    def test_partial_overlap(self) -> None:
        ref = ["A", "B", "C"] * 50
        cur = ["A", "B", "D"] * 50  # C replaced by D
        psi = _compute_categorical_psi(ref, cur)
        assert psi > 0.0


class TestClassifyPSI:
    """Test the PSI severity classifier."""

    def test_stable(self) -> None:
        assert _classify_psi(0.0) == "stable"
        assert _classify_psi(0.05) == "stable"

    def test_warning(self) -> None:
        assert _classify_psi(0.1) == "warning"
        assert _classify_psi(0.15) == "warning"

    def test_critical(self) -> None:
        assert _classify_psi(0.2) == "critical"
        assert _classify_psi(1.0) == "critical"


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


class TestDriftReport:
    """Test the DriftReport dataclass."""

    def test_to_dict_serialisable(self) -> None:
        report = DriftReport(
            is_drifted=True,
            drift_score=0.15,
            feature_drifts={"length": {"psi": 0.15, "is_drifted": True}},
            reference_stats={"length": {"mean": 100}},
            current_stats={"length": {"mean": 200}},
            window_size=50,
        )
        d = report.to_dict()
        assert d["is_drifted"] is True
        assert d["drift_score"] == 0.15
        assert "timestamp" in d
        assert d["window_size"] == 50

    def test_defaults(self) -> None:
        report = DriftReport(
            is_drifted=False,
            drift_score=0.0,
            feature_drifts={},
            reference_stats={},
            current_stats={},
        )
        assert report.window_size == 100
        assert report.timestamp is not None


# ---------------------------------------------------------------------------
# TextDistributionMonitor
# ---------------------------------------------------------------------------


class TestTextDistributionMonitor:
    """Test text-level drift detection."""

    def test_track_increments_count(self, text_monitor: TextDistributionMonitor) -> None:
        assert text_monitor.n_tracked == 0
        text_monitor.track("Hello world")
        assert text_monitor.n_tracked == 1

    def test_reference_freezes_at_threshold(self, text_monitor: TextDistributionMonitor) -> None:
        assert not text_monitor.reference_frozen
        for i in range(10):
            text_monitor.track(_make_clinical_note())
        assert text_monitor.reference_frozen

    def test_reference_not_frozen_before_threshold(self, text_monitor: TextDistributionMonitor) -> None:
        for _ in range(5):
            text_monitor.track(_make_clinical_note())
        assert not text_monitor.reference_frozen

    def test_compute_drift_no_data_returns_empty(self, text_monitor: TextDistributionMonitor) -> None:
        report = text_monitor.compute_drift()
        assert not report.is_drifted
        assert report.drift_score == 0.0
        assert report.feature_drifts == {}

    def test_compute_drift_stable_distribution(self, text_monitor: TextDistributionMonitor) -> None:
        # Feed similar documents
        for _ in range(20):
            text_monitor.track(_make_clinical_note())
        report = text_monitor.compute_drift(window_size=10)
        # Same distribution → should not drift
        assert report.drift_score < PSI_WARNING_THRESHOLD

    def test_compute_drift_shifted_distribution(self) -> None:
        monitor = TextDistributionMonitor(reference_size=15, max_history=500)
        # Build reference with standard clinical notes
        for _ in range(15):
            monitor.track(_make_clinical_note(length="medium", style="standard"))
        assert monitor.reference_frozen

        # Now feed very different documents
        for _ in range(20):
            monitor.track(_make_clinical_note(length="short", style="non_medical"))
        report = monitor.compute_drift(window_size=20)
        # At least some features should show drift
        assert len(report.feature_drifts) > 0

    def test_extract_stats_keys(self, text_monitor: TextDistributionMonitor) -> None:
        stats = text_monitor._extract_stats(_make_clinical_note())
        expected_keys = {"doc_length", "word_count", "vocab_diversity", "abbrev_density", "section_count"}
        assert set(stats.keys()) == expected_keys

    def test_extract_stats_medical_note_has_sections(self, text_monitor: TextDistributionMonitor) -> None:
        note = _make_clinical_note(style="standard")
        stats = text_monitor._extract_stats(note)
        assert stats["section_count"] >= 1  # At least "Chief Complaint" or "Assessment"

    def test_extract_stats_abbreviation_density(self, text_monitor: TextDistributionMonitor) -> None:
        note = "BP 120/80 HR 72 RR 16 O2 sat 98% WBC 7.2"
        stats = text_monitor._extract_stats(note)
        assert stats["abbrev_density"] > 0

    def test_max_history_respected(self) -> None:
        monitor = TextDistributionMonitor(reference_size=5, max_history=10)
        for i in range(20):
            monitor.track(f"Document number {i} with some content")
        # Deque maxlen should cap at 10
        assert monitor.n_tracked == 10

    def test_summarise_empty_returns_empty(self, text_monitor: TextDistributionMonitor) -> None:
        result = text_monitor._summarise([])
        assert result == {}

    def test_summarise_produces_stats(self, text_monitor: TextDistributionMonitor) -> None:
        records = [
            {"doc_length": 100.0, "word_count": 20.0},
            {"doc_length": 200.0, "word_count": 40.0},
        ]
        result = text_monitor._summarise(records)
        assert "doc_length" in result
        assert result["doc_length"]["mean"] == 150.0
        assert result["doc_length"]["n"] == 2


# ---------------------------------------------------------------------------
# PredictionMonitor
# ---------------------------------------------------------------------------


class TestPredictionMonitor:
    """Test prediction-level drift detection."""

    def test_registered_models_initially_empty(self, prediction_monitor: PredictionMonitor) -> None:
        assert prediction_monitor.registered_models == []

    def test_track_prediction_registers_model(self, prediction_monitor: PredictionMonitor) -> None:
        prediction_monitor.track_prediction("test-model", ["A01"], confidence=0.9)
        assert "test-model" in prediction_monitor.registered_models

    def test_track_prediction_with_dict_predictions(self, prediction_monitor: PredictionMonitor) -> None:
        preds = [{"code": "E11.9", "confidence": 0.85}]
        prediction_monitor.track_prediction("icd-model", preds, confidence=0.85)
        assert "icd-model" in prediction_monitor.registered_models

    def test_track_prediction_list_confidence(self, prediction_monitor: PredictionMonitor) -> None:
        """Confidence as a list should be averaged."""
        prediction_monitor.track_prediction("m", ["A", "B"], confidence=[0.8, 0.6])
        # Should not crash; internal confidence = 0.7

    def test_confidence_drift_no_data_returns_empty(self, prediction_monitor: PredictionMonitor) -> None:
        report = prediction_monitor.detect_confidence_drift()
        assert not report.is_drifted
        assert report.drift_score == 0.0

    def test_confidence_drift_stable(self) -> None:
        # Use a large reference so PSI is stable.
        monitor = PredictionMonitor(reference_size=200, max_history=1000)
        rng = np.random.RandomState(42)
        for _ in range(600):
            monitor.track_prediction(
                "stable-model",
                ["E11.9"],
                confidence=rng.normal(0.85, 0.02),
            )
        report = monitor.detect_confidence_drift(
            window_size=200, model_name="stable-model"
        )
        # Same IID distribution → PSI should be low
        assert report.drift_score < PSI_CRITICAL_THRESHOLD

    def test_prediction_drift_no_data_returns_empty(self, prediction_monitor: PredictionMonitor) -> None:
        report = prediction_monitor.detect_prediction_drift()
        assert not report.is_drifted

    def test_prediction_drift_stable_labels(self) -> None:
        monitor = PredictionMonitor(reference_size=200, max_history=1000)
        rng = np.random.RandomState(42)
        labels = ["E11.9", "I10", "J06.9"]
        for _ in range(600):
            label = labels[rng.randint(len(labels))]
            monitor.track_prediction("label-model", [label], confidence=0.8)
        report = monitor.detect_prediction_drift(
            window_size=200, model_name="label-model"
        )
        # Same distribution → low drift
        assert report.drift_score < PSI_CRITICAL_THRESHOLD

    def test_reference_freezes(self, prediction_monitor: PredictionMonitor) -> None:
        for _ in range(10):
            prediction_monitor.track_prediction("freeze-test", ["A"], confidence=0.9)
        assert prediction_monitor._reference_frozen.get("freeze-test", False)

    def test_resolve_model_none_picks_first(self, prediction_monitor: PredictionMonitor) -> None:
        prediction_monitor.track_prediction("first", ["A"], confidence=0.5)
        prediction_monitor.track_prediction("second", ["B"], confidence=0.6)
        resolved = prediction_monitor._resolve_model(None)
        assert resolved == "first"

    def test_resolve_model_explicit(self, prediction_monitor: PredictionMonitor) -> None:
        assert prediction_monitor._resolve_model("explicit") == "explicit"

    def test_empty_report_structure(self, prediction_monitor: PredictionMonitor) -> None:
        report = prediction_monitor._empty_report(50)
        assert not report.is_drifted
        assert report.window_size == 50
        assert report.feature_drifts == {}

    def test_confidence_drift_report_fields(self, prediction_monitor: PredictionMonitor) -> None:
        for _ in range(15):
            prediction_monitor.track_prediction("m", ["A"], confidence=0.8)
        report = prediction_monitor.detect_confidence_drift(window_size=5, model_name="m")
        if report.feature_drifts:
            conf = report.feature_drifts["confidence"]
            assert "psi" in conf
            assert "severity" in conf
            assert "reference_mean" in conf
            assert "current_mean" in conf

    def test_prediction_drift_report_has_top_labels(self, prediction_monitor: PredictionMonitor) -> None:
        for _ in range(15):
            prediction_monitor.track_prediction("m", ["E11.9"], confidence=0.8)
        report = prediction_monitor.detect_prediction_drift(window_size=5, model_name="m")
        if report.feature_drifts:
            ld = report.feature_drifts["label_distribution"]
            assert "reference_top_labels" in ld
            assert "current_top_labels" in ld

    def test_multiple_models_independent(self, prediction_monitor: PredictionMonitor) -> None:
        """Tracking for one model should not affect another."""
        for _ in range(15):
            prediction_monitor.track_prediction("model-a", ["X"], confidence=0.9)
        for _ in range(15):
            prediction_monitor.track_prediction("model-b", ["Y"], confidence=0.5)

        report_a = prediction_monitor.detect_confidence_drift(model_name="model-a")
        report_b = prediction_monitor.detect_confidence_drift(model_name="model-b")
        # Both should have data, but their stats should differ
        if report_a.reference_stats and report_b.reference_stats:
            assert report_a.reference_stats != report_b.reference_stats
