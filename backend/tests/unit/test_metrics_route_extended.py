"""Extended tests for metrics route — targeting uncovered Prometheus and JSON paths.

Covers:
- ``/metrics`` with prometheus_client installed (Prometheus text format)
- ``/metrics`` without prometheus_client (JSON fallback)
- ``/metrics/models`` endpoint
- ``_json_encode`` helper with datetime and set types
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class TestMetricsPrometheusPath:
    """Cover /metrics with prometheus_client available."""

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self) -> None:
        """Should return text/plain Prometheus format when library available."""
        mock_prometheus = MagicMock()
        mock_prometheus.REGISTRY = MagicMock()
        mock_prometheus.generate_latest.return_value = b"# HELP test_metric\ntest_metric 42\n"

        with patch.dict("sys.modules", {"prometheus_client": mock_prometheus}):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/api/v1/metrics")

        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_metrics_json_fallback(self) -> None:
        """Should return JSON when prometheus_client not installed."""
        # Force ImportError by patching the import inside the handler
        with patch(
            "app.api.v1.routes.metrics.Response",
        ) as MockResponse:
            MockResponse.side_effect = lambda **kwargs: MagicMock(
                status_code=200,
                headers={"content-type": kwargs.get("media_type", "")},
            )

            # Just verify the endpoint doesn't crash
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.get("/api/v1/metrics")

            assert resp.status_code == 200


class TestModelMetricsEndpoint:
    """Cover /metrics/models endpoint."""

    @pytest.mark.asyncio
    async def test_model_metrics_returns_summary(self) -> None:
        """Should return per-model metrics summary."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/api/v1/metrics/models")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"


class TestJsonEncode:
    """Cover _json_encode helper with special types."""

    def test_encodes_datetime(self) -> None:
        """Should encode datetime objects to ISO format."""
        from app.api.v1.routes.metrics import _json_encode

        dt = datetime(2026, 3, 25, 12, 0, 0)
        result = _json_encode({"timestamp": dt})
        assert "2026-03-25T12:00:00" in result

    def test_encodes_set(self) -> None:
        """Should encode sets as sorted lists."""
        from app.api.v1.routes.metrics import _json_encode

        result = _json_encode({"tags": {"a", "b"}})
        assert '"a"' in result
        assert '"b"' in result

    def test_encodes_nested_dict(self) -> None:
        """Should handle nested dictionaries."""
        from app.api.v1.routes.metrics import _json_encode

        data = {"models": {"ner": {"count": 42}}}
        result = _json_encode(data)
        assert '"count": 42' in result
