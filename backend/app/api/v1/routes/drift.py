"""Data and model drift monitoring endpoints.

Exposes drift detection results via REST so that Grafana dashboards and
alerting pipelines can query drift status without scraping Prometheus.
The underlying computation is delegated to
:mod:`app.ml.monitoring.drift_detector`.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["monitoring"])

# Module-level singletons so state accumulates across requests.
_text_monitor: Any = None
_pred_monitor: Any = None


def _get_text_monitor() -> Any:
    """Lazy-init the text distribution monitor singleton."""
    global _text_monitor
    if _text_monitor is None:
        from app.ml.monitoring.drift_detector import TextDistributionMonitor

        _text_monitor = TextDistributionMonitor()
    return _text_monitor


def _get_pred_monitor() -> Any:
    """Lazy-init the prediction monitor singleton."""
    global _pred_monitor
    if _pred_monitor is None:
        from app.ml.monitoring.drift_detector import PredictionMonitor

        _pred_monitor = PredictionMonitor()
    return _pred_monitor


@router.get(
    "/status",
    summary="Overall drift status",
    description=(
        "Returns the current data-drift and prediction-drift status for "
        "all monitored models.  Each model section includes PSI scores, "
        "vocabulary diversity, label-distribution shift, and confidence "
        "drift indicators."
    ),
    response_model=dict[str, Any],
)
async def drift_status() -> dict[str, Any]:
    """Return aggregated drift metrics for all monitored models.

    Returns
    -------
    dict
        Top-level keys: ``text_distribution``, ``prediction_drift``,
        ``overall_status`` (``'stable'`` | ``'warning'`` | ``'drifted'``).
    """
    text_monitor = _get_text_monitor()
    pred_monitor = _get_pred_monitor()

    # Text distribution: compute_drift returns a DriftReport dataclass
    text_report = text_monitor.compute_drift()
    text_metrics = {
        "drift_detected": text_report.is_drifted,
        "drift_score": text_report.drift_score,
        "n_tracked": text_monitor.n_tracked,
        "reference_frozen": text_monitor.reference_frozen,
        "feature_drifts": text_report.feature_drifts,
    }

    # Prediction confidence drift per model
    pred_metrics: dict[str, Any] = {}
    for model_name in list(pred_monitor._confidence_records.keys()):
        conf_report = pred_monitor.detect_confidence_drift(model_name=model_name)
        pred_metrics[model_name] = {
            "drift_detected": conf_report.is_drifted,
            "drift_score": conf_report.drift_score,
            "feature_drifts": conf_report.feature_drifts,
        }

    # Derive overall status
    has_text_drift = text_metrics.get("drift_detected", False)
    has_pred_drift = any(
        m.get("drift_detected", False)
        for m in pred_metrics.values()
        if isinstance(m, dict)
    )

    if has_text_drift and has_pred_drift:
        overall = "drifted"
    elif has_text_drift or has_pred_drift:
        overall = "warning"
    else:
        overall = "stable"

    return {
        "overall_status": overall,
        "text_distribution": text_metrics,
        "prediction_drift": pred_metrics,
    }


@router.post(
    "/record",
    summary="Record a prediction for drift tracking",
    description=(
        "Ingest a prediction record into the drift monitors.  Call this "
        "after each inference to keep drift statistics up to date.  In "
        "production, this is called internally by the pipeline; the "
        "endpoint exists for testing and batch back-fill."
    ),
    response_model=dict[str, str],
)
async def record_prediction(
    model_name: str,
    predicted_label: str,
    confidence: float,
    text: str | None = None,
) -> dict[str, str]:
    """Record a single prediction for drift monitoring.

    Parameters
    ----------
    model_name : str
        Identifier of the model that produced the prediction.
    predicted_label : str
        The label/class predicted by the model.
    confidence : float
        Model confidence score (0–1).
    text : str, optional
        Input text — recorded by the text distribution monitor when provided.

    Returns
    -------
    dict
        ``{"status": "recorded"}`` on success.
    """
    pred_monitor = _get_pred_monitor()
    pred_monitor.track_prediction(
        model_name=model_name,
        predictions=[predicted_label],
        confidence=confidence,
    )

    if text:
        text_monitor = _get_text_monitor()
        text_monitor.track(text)

    return {"status": "recorded"}
