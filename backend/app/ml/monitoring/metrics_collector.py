"""Prometheus-compatible metrics collection for ClinIQ ML models.

Exposes model performance counters and histograms through
``prometheus_client`` when available, with a transparent dict-based fallback
that maintains identical semantics so the rest of the codebase never needs to
branch on whether Prometheus is installed.

Usage
-----
>>> from app.ml.monitoring.metrics_collector import ModelMetrics
>>> metrics = ModelMetrics()
>>> metrics.record_inference("sklearn-baseline", latency_ms=42.3, prediction_type="icd10")
>>> print(metrics.get_metrics())
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Histogram bucket boundaries (in seconds) for inference latency
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


# ---------------------------------------------------------------------------
# Internal fallback histogram / counter / gauge
# ---------------------------------------------------------------------------


class _SimpleHistogram:
    """Minimal histogram that tracks count, sum, and bucket counts."""

    def __init__(self, buckets: tuple[float, ...] = _LATENCY_BUCKETS) -> None:
        self._buckets = sorted(buckets)
        self._bucket_counts: list[int] = [0] * (len(self._buckets) + 1)
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for i, upper in enumerate(self._buckets):
            if value <= upper:
                self._bucket_counts[i] += 1
                return
        self._bucket_counts[-1] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "sum": round(self._sum, 6),
            "buckets": {
                **{str(b): self._bucket_counts[i] for i, b in enumerate(self._buckets)},
                "+Inf": sum(self._bucket_counts),
            },
        }


class _SimpleCounter:
    def __init__(self) -> None:
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


class _SimpleGauge:
    def __init__(self) -> None:
        self._value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


# ---------------------------------------------------------------------------
# ModelMetrics — public API
# ---------------------------------------------------------------------------


class ModelMetrics:
    """Collect and expose operational metrics for ClinIQ ML models.

    When ``prometheus_client`` is installed the metrics are registered with
    the global Prometheus registry and can be scraped by a Prometheus server.
    Otherwise a lightweight in-process store is used with identical semantics.

    Metrics
    -------
    ``cliniq_inference_latency_seconds``
        Histogram.  Latency per inference call, labelled by model name.
    ``cliniq_prediction_count_total``
        Counter.  Total predictions, labelled by model name and prediction type.
    ``cliniq_model_load_time_seconds``
        Gauge.  Time taken to load each model (seconds).
    ``cliniq_active_models``
        Gauge.  Number of currently loaded models.
    ``cliniq_error_count_total``
        Counter.  Total errors, labelled by model name and error type.
    ``cliniq_batch_size``
        Histogram.  Distribution of batch sizes submitted to the models.

    Parameters
    ----------
    namespace:
        Prometheus metric name prefix (default ``"cliniq"``).
    use_prometheus:
        If ``True`` (default) attempt to import ``prometheus_client``.
        Set ``False`` to force the fallback store even when Prometheus is
        installed (useful in tests).
    """

    def __init__(
        self,
        namespace: str = "cliniq",
        use_prometheus: bool = True,
    ) -> None:
        self._namespace = namespace
        self._prometheus_available = False
        self._prom: dict[str, Any] = {}

        if use_prometheus:
            self._try_init_prometheus()

        if not self._prometheus_available:
            self._init_fallback()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_inference(
        self,
        model_name: str,
        latency_ms: float,
        prediction_type: str = "default",
    ) -> None:
        """Record a completed inference call.

        Parameters
        ----------
        model_name:
            Logical model identifier (e.g. ``"sklearn-baseline"``).
        latency_ms:
            Inference latency in **milliseconds** (converted to seconds
            internally).
        prediction_type:
            Label for the kind of prediction (e.g. ``"icd10"``, ``"ner"``).
        """
        latency_s = latency_ms / 1000.0

        if self._prometheus_available:
            self._prom["inference_latency"].labels(model=model_name).observe(latency_s)
            self._prom["prediction_count"].labels(
                model=model_name, prediction_type=prediction_type
            ).inc()
        else:
            key = f"{model_name}"
            self._fb_latency[key].observe(latency_s)
            self._fb_pred_count[(model_name, prediction_type)].inc()

    def record_batch(self, model_name: str, batch_size: int) -> None:
        """Record the size of a batch submitted to *model_name*.

        Parameters
        ----------
        model_name:
            Logical model identifier.
        batch_size:
            Number of documents in the batch.
        """
        if self._prometheus_available:
            self._prom["batch_size"].labels(model=model_name).observe(float(batch_size))
        else:
            self._fb_batch[model_name].observe(float(batch_size))

    def record_error(self, model_name: str, error_type: str) -> None:
        """Increment the error counter for a model.

        Parameters
        ----------
        model_name:
            Logical model identifier.
        error_type:
            Short error category string (e.g. ``"inference_error"``,
            ``"timeout"``, ``"oom"``).
        """
        if self._prometheus_available:
            self._prom["error_count"].labels(
                model=model_name, error_type=error_type
            ).inc()
        else:
            self._fb_errors[(model_name, error_type)].inc()

    def set_model_load_time(self, model_name: str, load_time_s: float) -> None:
        """Record how long it took to load *model_name*.

        Parameters
        ----------
        model_name:
            Logical model identifier.
        load_time_s:
            Load time in seconds.
        """
        if self._prometheus_available:
            self._prom["model_load_time"].labels(model=model_name).set(load_time_s)
        else:
            self._fb_load_time[model_name].set(load_time_s)

    def set_active_models(self, count: int) -> None:
        """Set the number of currently active (loaded) models.

        Parameters
        ----------
        count:
            Absolute count of loaded models.
        """
        if self._prometheus_available:
            self._prom["active_models"].set(float(count))
        else:
            self._fb_active_models.set(float(count))

    def get_metrics(self) -> dict[str, Any]:
        """Return all current metrics as a plain dictionary.

        When Prometheus is active this serialises the in-process registry
        state.  When using the fallback store it returns the stored values
        directly.

        Returns
        -------
        dict
            Nested dictionary with metric names as top-level keys.
        """
        if self._prometheus_available:
            return self._collect_prometheus_metrics()
        return self._collect_fallback_metrics()

    # ------------------------------------------------------------------
    # Context-manager helper for timing blocks
    # ------------------------------------------------------------------

    class _InferenceTimer:
        """Context manager that records inference latency on exit."""

        def __init__(
            self,
            metrics: "ModelMetrics",
            model_name: str,
            prediction_type: str,
        ) -> None:
            self._metrics = metrics
            self._model_name = model_name
            self._prediction_type = prediction_type
            self._start: float = 0.0

        def __enter__(self) -> "ModelMetrics._InferenceTimer":
            self._start = time.monotonic()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            elapsed_ms = (time.monotonic() - self._start) * 1000
            if exc_type is not None:
                self._metrics.record_error(self._model_name, exc_type.__name__)
            else:
                self._metrics.record_inference(
                    self._model_name, elapsed_ms, self._prediction_type
                )

    def time_inference(
        self, model_name: str, prediction_type: str = "default"
    ) -> "ModelMetrics._InferenceTimer":
        """Return a context manager that auto-records inference timing.

        Example
        -------
        >>> with metrics.time_inference("sklearn-baseline", "icd10"):
        ...     result = model.predict(text)
        """
        return self._InferenceTimer(self, model_name, prediction_type)

    # ------------------------------------------------------------------
    # Prometheus initialisation
    # ------------------------------------------------------------------

    def _try_init_prometheus(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY  # noqa: F401

            prefix = self._namespace

            self._prom["inference_latency"] = Histogram(
                f"{prefix}_inference_latency_seconds",
                "Model inference latency in seconds",
                labelnames=["model"],
                buckets=_LATENCY_BUCKETS,
            )
            self._prom["prediction_count"] = Counter(
                f"{prefix}_prediction_count_total",
                "Total number of predictions made",
                labelnames=["model", "prediction_type"],
            )
            self._prom["model_load_time"] = Gauge(
                f"{prefix}_model_load_time_seconds",
                "Time taken to load the model in seconds",
                labelnames=["model"],
            )
            self._prom["active_models"] = Gauge(
                f"{prefix}_active_models",
                "Number of currently loaded models",
            )
            self._prom["error_count"] = Counter(
                f"{prefix}_error_count_total",
                "Total number of errors encountered",
                labelnames=["model", "error_type"],
            )
            self._prom["batch_size"] = Histogram(
                f"{prefix}_batch_size",
                "Distribution of batch sizes",
                labelnames=["model"],
                buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
            )

            self._prometheus_available = True
            logger.info("ModelMetrics: prometheus_client initialised")

        except Exception as exc:
            logger.info("prometheus_client not available (%s) — using fallback store", exc)

    # ------------------------------------------------------------------
    # Fallback in-memory store initialisation
    # ------------------------------------------------------------------

    def _init_fallback(self) -> None:
        self._fb_latency: dict[str, _SimpleHistogram] = defaultdict(
            lambda: _SimpleHistogram(_LATENCY_BUCKETS)
        )
        self._fb_pred_count: dict[tuple[str, str], _SimpleCounter] = defaultdict(
            _SimpleCounter
        )
        self._fb_load_time: dict[str, _SimpleGauge] = defaultdict(_SimpleGauge)
        self._fb_active_models: _SimpleGauge = _SimpleGauge()
        self._fb_errors: dict[tuple[str, str], _SimpleCounter] = defaultdict(_SimpleCounter)
        self._fb_batch: dict[str, _SimpleHistogram] = defaultdict(
            lambda: _SimpleHistogram((1, 2, 5, 10, 20, 50, 100, 200, 500))
        )

    # ------------------------------------------------------------------
    # Metric serialisation helpers
    # ------------------------------------------------------------------

    def _collect_fallback_metrics(self) -> dict[str, Any]:
        """Serialise fallback store to a plain dict."""
        return {
            "inference_latency_seconds": {
                model: hist.to_dict() for model, hist in self._fb_latency.items()
            },
            "prediction_count_total": {
                f"{model}__{ptype}": counter.value
                for (model, ptype), counter in self._fb_pred_count.items()
            },
            "model_load_time_seconds": {
                model: gauge.value for model, gauge in self._fb_load_time.items()
            },
            "active_models": self._fb_active_models.value,
            "error_count_total": {
                f"{model}__{etype}": counter.value
                for (model, etype), counter in self._fb_errors.items()
            },
            "batch_size": {
                model: hist.to_dict() for model, hist in self._fb_batch.items()
            },
        }

    def _collect_prometheus_metrics(self) -> dict[str, Any]:
        """Read Prometheus metric samples and serialise to a plain dict."""
        try:
            from prometheus_client import REGISTRY
            from prometheus_client.exposition import choose_encoder

            output: dict[str, Any] = {}

            for metric in REGISTRY.collect():
                if not metric.name.startswith(self._namespace):
                    continue

                samples: dict[str, Any] = {}
                for sample in metric.samples:
                    label_str = "_".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
                    key = f"{sample.name}__{label_str}" if label_str else sample.name
                    samples[key] = sample.value

                output[metric.name] = {
                    "type": metric.type,
                    "documentation": metric.documentation,
                    "samples": samples,
                }

            return output

        except Exception as exc:
            logger.warning("Failed to collect Prometheus metrics: %s", exc)
            return {}
