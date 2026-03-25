"""Advanced tests for ModelMetrics — Prometheus initialisation, collect methods, and time_inference.

Covers the Prometheus-backed code paths (mocked prometheus_client),
_collect_fallback_metrics serialisation, _collect_prometheus_metrics,
and the _InferenceTimer context manager.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ml.monitoring.metrics_collector import ModelMetrics

# ---------------------------------------------------------------------------
# Prometheus init path
# ---------------------------------------------------------------------------


class TestModelMetricsPrometheusInit:
    """Test _try_init_prometheus when prometheus_client is available."""

    def test_prometheus_init_success(self) -> None:
        """When prometheus_client is importable, Prometheus objects are created."""
        mock_prom = MagicMock()
        mock_prom.Histogram.return_value = MagicMock()
        mock_prom.Counter.return_value = MagicMock()
        mock_prom.Gauge.return_value = MagicMock()

        with patch.dict("sys.modules", {"prometheus_client": mock_prom}):
            # Need a unique namespace to avoid Prometheus re-registration errors
            m = ModelMetrics(namespace="test_prom_init", use_prometheus=True)

        assert m._prometheus_available is True
        assert "inference_latency" in m._prom

    def test_prometheus_init_failure_falls_back(self) -> None:
        """When prometheus_client import fails, fallback store is used."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            m = ModelMetrics(namespace="test_prom_fail", use_prometheus=True)

        assert m._prometheus_available is False
        assert hasattr(m, "_fb_latency")

    def test_prometheus_disabled(self) -> None:
        """When use_prometheus=False, fallback store is always used."""
        m = ModelMetrics(namespace="test_disabled", use_prometheus=False)
        assert m._prometheus_available is False
        assert hasattr(m, "_fb_latency")


# ---------------------------------------------------------------------------
# Prometheus record paths
# ---------------------------------------------------------------------------


class TestModelMetricsPrometheusRecord:
    """Test record methods when Prometheus is active."""

    @pytest.fixture()
    def prom_metrics(self) -> ModelMetrics:
        m = ModelMetrics(namespace="test_rec", use_prometheus=False)
        # Simulate Prometheus available by injecting mock objects
        m._prometheus_available = True
        m._prom = {
            "inference_latency": MagicMock(),
            "prediction_count": MagicMock(),
            "model_load_time": MagicMock(),
            "active_models": MagicMock(),
            "error_count": MagicMock(),
            "batch_size": MagicMock(),
        }
        return m

    def test_record_inference_prometheus(self, prom_metrics: ModelMetrics) -> None:
        prom_metrics.record_inference("ner", 150.0, "entity")
        prom_metrics._prom["inference_latency"].labels.assert_called_with(model="ner")
        prom_metrics._prom["inference_latency"].labels().observe.assert_called_with(0.15)
        prom_metrics._prom["prediction_count"].labels.assert_called_with(
            model="ner", prediction_type="entity"
        )
        prom_metrics._prom["prediction_count"].labels().inc.assert_called_once()

    def test_record_batch_prometheus(self, prom_metrics: ModelMetrics) -> None:
        prom_metrics.record_batch("icd", 32)
        prom_metrics._prom["batch_size"].labels.assert_called_with(model="icd")
        prom_metrics._prom["batch_size"].labels().observe.assert_called_with(32)

    def test_record_error_prometheus(self, prom_metrics: ModelMetrics) -> None:
        prom_metrics.record_error("ner", "InferenceError")
        prom_metrics._prom["error_count"].labels.assert_called_with(
            model="ner", error_type="InferenceError"
        )
        prom_metrics._prom["error_count"].labels().inc.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback collect
# ---------------------------------------------------------------------------


class TestCollectFallbackMetrics:
    """Test _collect_fallback_metrics serialisation."""

    def test_empty_metrics(self) -> None:
        m = ModelMetrics(namespace="test_empty", use_prometheus=False)
        result = m._collect_fallback_metrics()
        assert "inference_latency_seconds" in result
        assert "prediction_count_total" in result
        assert "active_models" in result
        assert result["active_models"] == 0

    def test_metrics_after_recording(self) -> None:
        m = ModelMetrics(namespace="test_after", use_prometheus=False)
        m.record_inference("ner", 100.0, "entity")
        m.record_inference("ner", 200.0, "entity")
        m.record_batch("icd", 10)
        m.record_error("ner", "timeout")

        result = m._collect_fallback_metrics()
        assert "ner" in result["inference_latency_seconds"]
        assert result["prediction_count_total"]["ner__entity"] == 2
        assert "ner__timeout" in result["error_count_total"]

    def test_batch_size_in_collect(self) -> None:
        m = ModelMetrics(namespace="test_batch", use_prometheus=False)
        m.record_batch("summarizer", 25)

        result = m._collect_fallback_metrics()
        assert "summarizer" in result["batch_size"]


# ---------------------------------------------------------------------------
# Prometheus collect
# ---------------------------------------------------------------------------


class TestCollectPrometheusMetrics:
    """Test _collect_prometheus_metrics."""

    def test_collect_prometheus_success(self) -> None:
        """Mocked Prometheus REGISTRY returns metric families."""
        mock_sample = MagicMock()
        mock_sample.name = "test_ns_inference_latency_seconds_bucket"
        mock_sample.labels = {"model": "ner", "le": "0.1"}
        mock_sample.value = 5.0

        mock_metric = MagicMock()
        mock_metric.name = "test_ns_inference_latency_seconds"
        mock_metric.type = "histogram"
        mock_metric.documentation = "Model inference latency"
        mock_metric.samples = [mock_sample]

        mock_prom_module = MagicMock()
        mock_prom_module.REGISTRY.collect.return_value = [mock_metric]

        m = ModelMetrics(namespace="test_ns", use_prometheus=False)
        m._prometheus_available = True
        m._namespace = "test_ns"

        with patch.dict("sys.modules", {"prometheus_client": mock_prom_module, "prometheus_client.exposition": MagicMock()}):
            result = m._collect_prometheus_metrics()

        assert "test_ns_inference_latency_seconds" in result
        assert result["test_ns_inference_latency_seconds"]["type"] == "histogram"

    def test_collect_prometheus_exception_returns_empty(self) -> None:
        """When REGISTRY.collect fails, returns empty dict."""
        m = ModelMetrics(namespace="test_fail", use_prometheus=False)
        m._prometheus_available = True
        m._namespace = "test_fail"

        mock_prom = MagicMock()
        mock_prom.REGISTRY.collect.side_effect = RuntimeError("registry error")

        with patch.dict("sys.modules", {"prometheus_client": mock_prom, "prometheus_client.exposition": MagicMock()}):
            result = m._collect_prometheus_metrics()

        assert result == {}

    def test_collect_filters_by_namespace(self) -> None:
        """Only metrics matching the namespace prefix are included."""
        matching = MagicMock()
        matching.name = "myns_prediction_count"
        matching.type = "counter"
        matching.documentation = "Count"
        matching.samples = []

        other = MagicMock()
        other.name = "other_metric"
        other.type = "gauge"
        other.documentation = "Other"
        other.samples = []

        mock_prom = MagicMock()
        mock_prom.REGISTRY.collect.return_value = [matching, other]

        m = ModelMetrics(namespace="myns", use_prometheus=False)
        m._prometheus_available = True
        m._namespace = "myns"

        with patch.dict("sys.modules", {"prometheus_client": mock_prom, "prometheus_client.exposition": MagicMock()}):
            result = m._collect_prometheus_metrics()

        assert "myns_prediction_count" in result
        assert "other_metric" not in result


# ---------------------------------------------------------------------------
# time_inference context manager
# ---------------------------------------------------------------------------


class TestTimeInference:
    """Test the _InferenceTimer context manager."""

    def test_timer_records_inference(self) -> None:
        m = ModelMetrics(namespace="test_timer", use_prometheus=False)

        with m.time_inference("ner", "entity"):
            pass  # Simulate fast inference

        result = m._collect_fallback_metrics()
        assert "ner" in result["inference_latency_seconds"]
        assert result["prediction_count_total"]["ner__entity"] == 1

    def test_timer_records_error_on_exception(self) -> None:
        """Timer records an error (not inference) when exception occurs."""
        m = ModelMetrics(namespace="test_timer_exc", use_prometheus=False)

        with pytest.raises(ValueError), m.time_inference("ner", "entity"):
            raise ValueError("inference failed")

        result = m._collect_fallback_metrics()
        assert result["error_count_total"]["ner__ValueError"] == 1
        # Inference count should NOT have been incremented
        assert result["prediction_count_total"].get("ner__entity", 0) == 0
