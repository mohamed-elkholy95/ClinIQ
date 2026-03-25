"""Prometheus-compatible metrics endpoint.

Exposes model inference latencies, error counts, batch sizes, and
application-level gauges in Prometheus text exposition format when
``prometheus_client`` is installed, or a JSON fallback otherwise.

This endpoint is intended for scraping by Prometheus and should be
excluded from authentication middleware in production.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


def _get_global_metrics() -> Any:
    """Lazily import the global metrics singleton.

    The model registry exposes a module-level ``ModelMetrics`` instance
    through ``get_metrics()``.  Importing at call-time avoids circular
    dependency issues and lets the route work even if the ML subsystem
    hasn't fully initialised yet.
    """
    from app.ml.monitoring.metrics_collector import ModelMetrics

    # Return a fresh instance if no global singleton is available yet —
    # in production the pipeline sets up a shared one.
    return ModelMetrics()


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description=(
        "Returns application and ML model metrics in Prometheus text exposition "
        "format (``text/plain; version=0.0.4``) when ``prometheus_client`` is "
        "installed.  Falls back to a JSON representation otherwise.\n\n"
        "Scraped by the Prometheus instance defined in "
        "``infra/prometheus/prometheus.yml`` under the ``cliniq-api`` job."
    ),
    responses={
        200: {
            "description": "Metrics payload",
            "content": {
                "text/plain": {"schema": {"type": "string"}},
                "application/json": {},
            },
        },
    },
)
async def prometheus_metrics() -> Response:
    """Serve Prometheus metrics for the /metrics scrape endpoint.

    Returns
    -------
    Response
        ``text/plain`` with Prometheus exposition format when the client
        library is available; ``application/json`` fallback otherwise.

    Notes
    -----
    The endpoint does **not** require authentication so that Prometheus
    can scrape without a bearer token.  Network-level access control
    (e.g. internal-only service mesh) is the recommended protection.
    """
    try:
        from prometheus_client import (  # type: ignore[import-untyped]
            REGISTRY,
            generate_latest,
        )

        metrics_output = generate_latest(REGISTRY)
        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except ImportError:
        logger.debug("prometheus_client not installed — returning JSON fallback")
        metrics = _get_global_metrics()
        data = metrics.get_metrics()
        return Response(
            content=_json_encode(data),
            media_type="application/json",
        )


@router.get(
    "/metrics/models",
    summary="Per-model metrics summary",
    description=(
        "Returns a JSON summary of per-model inference counts, latencies, "
        "error rates, and batch sizes.  Useful for dashboards and alerting "
        "rules that operate on structured data rather than Prometheus queries."
    ),
    response_model=dict[str, Any],
)
async def model_metrics_summary() -> dict[str, Any]:
    """Return per-model metrics as structured JSON.

    Returns
    -------
    dict
        Keyed by model name, each value contains ``inference_count``,
        ``error_count``, ``avg_latency_ms``, and ``last_batch_size``.
    """
    metrics = _get_global_metrics()
    raw = metrics.get_metrics()

    # Reshape the raw dict into a more dashboard-friendly structure.
    summary: dict[str, Any] = {
        "status": "ok",
        "prometheus_available": raw.get("prometheus_available", False),
        "models": raw.get("models", {}),
    }
    return summary


def _json_encode(data: Any) -> str:
    """Encode data to JSON string, handling non-serialisable types."""
    import json
    from datetime import datetime

    class _Encoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            if isinstance(o, set):
                return list(o)
            return super().default(o)

    return json.dumps(data, cls=_Encoder, indent=2)
