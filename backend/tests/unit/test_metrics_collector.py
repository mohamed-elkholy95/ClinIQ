"""Unit tests for the Prometheus-compatible metrics collector.

Tests ModelMetrics, the fallback in-memory histogram/counter/gauge
implementations, and the inference timer context manager.
"""

import time

import pytest

from app.ml.monitoring.metrics_collector import (
    ModelMetrics,
    _SimpleCounter,
    _SimpleGauge,
    _SimpleHistogram,
)


# ---------------------------------------------------------------------------
# Internal primitives
# ---------------------------------------------------------------------------


class TestSimpleHistogram:
    """Test _SimpleHistogram bucket tracking."""

    def test_empty_histogram(self) -> None:
        h = _SimpleHistogram(buckets=(1.0, 5.0, 10.0))
        d = h.to_dict()
        assert d["count"] == 0
        assert d["sum"] == 0.0

    def test_observe_single_value(self) -> None:
        h = _SimpleHistogram(buckets=(1.0, 5.0, 10.0))
        h.observe(3.0)
        d = h.to_dict()
        assert d["count"] == 1
        assert d["sum"] == 3.0

    def test_bucket_assignment(self) -> None:
        h = _SimpleHistogram(buckets=(1.0, 5.0, 10.0))
        h.observe(0.5)  # → bucket ≤1.0
        h.observe(3.0)  # → bucket ≤5.0
        h.observe(7.0)  # → bucket ≤10.0
        h.observe(15.0)  # → overflow bucket
        d = h.to_dict()
        assert d["count"] == 4
        assert d["buckets"]["1.0"] == 1
        assert d["buckets"]["5.0"] == 1
        assert d["buckets"]["10.0"] == 1
        assert d["buckets"]["+Inf"] == 4  # Cumulative total

    def test_boundary_value_goes_to_bucket(self) -> None:
        h = _SimpleHistogram(buckets=(1.0, 5.0))
        h.observe(1.0)  # Exactly on boundary → ≤1.0 bucket
        d = h.to_dict()
        assert d["buckets"]["1.0"] == 1

    def test_sum_accuracy(self) -> None:
        h = _SimpleHistogram(buckets=(10.0,))
        for v in [1.0, 2.0, 3.0, 4.0]:
            h.observe(v)
        assert h.to_dict()["sum"] == 10.0


class TestSimpleCounter:
    """Test _SimpleCounter increment behaviour."""

    def test_starts_at_zero(self) -> None:
        c = _SimpleCounter()
        assert c.value == 0.0

    def test_inc_default(self) -> None:
        c = _SimpleCounter()
        c.inc()
        assert c.value == 1.0

    def test_inc_custom_amount(self) -> None:
        c = _SimpleCounter()
        c.inc(5.0)
        assert c.value == 5.0

    def test_multiple_increments(self) -> None:
        c = _SimpleCounter()
        c.inc(2.0)
        c.inc(3.0)
        assert c.value == 5.0


class TestSimpleGauge:
    """Test _SimpleGauge set/inc/dec."""

    def test_starts_at_zero(self) -> None:
        g = _SimpleGauge()
        assert g.value == 0.0

    def test_set(self) -> None:
        g = _SimpleGauge()
        g.set(42.0)
        assert g.value == 42.0

    def test_inc(self) -> None:
        g = _SimpleGauge()
        g.set(10.0)
        g.inc(5.0)
        assert g.value == 15.0

    def test_dec(self) -> None:
        g = _SimpleGauge()
        g.set(10.0)
        g.dec(3.0)
        assert g.value == 7.0

    def test_inc_default(self) -> None:
        g = _SimpleGauge()
        g.inc()
        assert g.value == 1.0


# ---------------------------------------------------------------------------
# ModelMetrics — fallback store
# ---------------------------------------------------------------------------


class TestModelMetricsFallback:
    """Test ModelMetrics using the in-memory fallback store."""

    @pytest.fixture
    def metrics(self) -> ModelMetrics:
        """Force fallback store (no Prometheus)."""
        return ModelMetrics(use_prometheus=False)

    def test_record_inference(self, metrics: ModelMetrics) -> None:
        metrics.record_inference("test-model", latency_ms=42.3, prediction_type="icd10")
        data = metrics.get_metrics()
        assert "test-model" in data["inference_latency_seconds"]
        assert data["inference_latency_seconds"]["test-model"]["count"] == 1

    def test_record_multiple_inferences(self, metrics: ModelMetrics) -> None:
        for i in range(5):
            metrics.record_inference("m", latency_ms=float(i * 10), prediction_type="ner")
        data = metrics.get_metrics()
        assert data["inference_latency_seconds"]["m"]["count"] == 5

    def test_prediction_count(self, metrics: ModelMetrics) -> None:
        metrics.record_inference("m", latency_ms=10, prediction_type="icd10")
        metrics.record_inference("m", latency_ms=20, prediction_type="icd10")
        metrics.record_inference("m", latency_ms=30, prediction_type="ner")
        data = metrics.get_metrics()
        assert data["prediction_count_total"]["m__icd10"] == 2.0
        assert data["prediction_count_total"]["m__ner"] == 1.0

    def test_record_batch(self, metrics: ModelMetrics) -> None:
        metrics.record_batch("m", batch_size=10)
        data = metrics.get_metrics()
        assert "m" in data["batch_size"]
        assert data["batch_size"]["m"]["count"] == 1

    def test_record_error(self, metrics: ModelMetrics) -> None:
        metrics.record_error("m", "timeout")
        metrics.record_error("m", "timeout")
        metrics.record_error("m", "oom")
        data = metrics.get_metrics()
        assert data["error_count_total"]["m__timeout"] == 2.0
        assert data["error_count_total"]["m__oom"] == 1.0

    def test_set_model_load_time(self, metrics: ModelMetrics) -> None:
        metrics.set_model_load_time("m", 2.5)
        data = metrics.get_metrics()
        assert data["model_load_time_seconds"]["m"] == 2.5

    def test_set_active_models(self, metrics: ModelMetrics) -> None:
        metrics.set_active_models(3)
        data = metrics.get_metrics()
        assert data["active_models"] == 3.0

    def test_latency_converted_to_seconds(self, metrics: ModelMetrics) -> None:
        metrics.record_inference("m", latency_ms=1000.0)
        data = metrics.get_metrics()
        # sum should be 1.0 second
        assert data["inference_latency_seconds"]["m"]["sum"] == 1.0

    def test_get_metrics_structure(self, metrics: ModelMetrics) -> None:
        data = metrics.get_metrics()
        expected_keys = {
            "inference_latency_seconds",
            "prediction_count_total",
            "model_load_time_seconds",
            "active_models",
            "error_count_total",
            "batch_size",
        }
        assert set(data.keys()) == expected_keys

    def test_multiple_models(self, metrics: ModelMetrics) -> None:
        metrics.record_inference("model-a", latency_ms=10, prediction_type="ner")
        metrics.record_inference("model-b", latency_ms=20, prediction_type="icd")
        data = metrics.get_metrics()
        assert "model-a" in data["inference_latency_seconds"]
        assert "model-b" in data["inference_latency_seconds"]


# ---------------------------------------------------------------------------
# ModelMetrics — inference timer context manager
# ---------------------------------------------------------------------------


class TestInferenceTimer:
    """Test the time_inference context manager."""

    def test_timer_records_latency(self) -> None:
        metrics = ModelMetrics(use_prometheus=False)
        with metrics.time_inference("timer-model", "icd10"):
            time.sleep(0.01)  # 10ms
        data = metrics.get_metrics()
        assert data["inference_latency_seconds"]["timer-model"]["count"] == 1
        assert data["inference_latency_seconds"]["timer-model"]["sum"] > 0

    def test_timer_records_error_on_exception(self) -> None:
        metrics = ModelMetrics(use_prometheus=False)
        with pytest.raises(ValueError):
            with metrics.time_inference("err-model", "ner"):
                raise ValueError("test error")
        data = metrics.get_metrics()
        assert data["error_count_total"]["err-model__ValueError"] == 1.0
        # Should NOT record inference latency on error
        assert data["inference_latency_seconds"].get("err-model") is None or \
               data["inference_latency_seconds"]["err-model"]["count"] == 0

    def test_timer_does_not_swallow_exception(self) -> None:
        metrics = ModelMetrics(use_prometheus=False)
        with pytest.raises(RuntimeError, match="boom"):
            with metrics.time_inference("m"):
                raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# ModelMetrics — namespace
# ---------------------------------------------------------------------------


class TestModelMetricsNamespace:
    """Verify custom namespace is accepted."""

    def test_custom_namespace(self) -> None:
        metrics = ModelMetrics(namespace="myapp", use_prometheus=False)
        assert metrics._namespace == "myapp"
        # Should still work normally
        metrics.record_inference("m", latency_ms=5.0)
        data = metrics.get_metrics()
        assert data["inference_latency_seconds"]["m"]["count"] == 1
