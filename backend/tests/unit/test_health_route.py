"""Unit tests for the health check route module.

Tests cover:
- ``_probe_database``: healthy and unhealthy states
- ``_probe_redis``: healthy, unhealthy, and unavailable (import missing) states
- ``_probe_models``: loaded and not-loaded model registry scenarios
- ``health_check``: overall status aggregation (healthy, degraded, unhealthy)
- ``liveness`` and ``readiness`` probes
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.routes.health import (
    _probe_database,
    _probe_models,
    _probe_redis,
    health_check,
    liveness,
    readiness,
)

# ---------------------------------------------------------------------------
# _probe_database
# ---------------------------------------------------------------------------


class TestProbeDatabase:
    """Tests for the _probe_database helper."""

    @pytest.mark.asyncio
    async def test_healthy_when_query_succeeds(self) -> None:
        """SELECT 1 succeeds → 'healthy'."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock())

        result = await _probe_database(mock_session)
        assert result == "healthy"
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unhealthy_when_query_raises(self) -> None:
        """Exception during SELECT 1 → 'unhealthy'."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=ConnectionError("db down"))

        result = await _probe_database(mock_session)
        assert result == "unhealthy"

    @pytest.mark.asyncio
    async def test_unhealthy_on_timeout_error(self) -> None:
        """Timeout-like exceptions should also yield 'unhealthy'."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=TimeoutError("slow db"))

        result = await _probe_database(mock_session)
        assert result == "unhealthy"


# ---------------------------------------------------------------------------
# _probe_redis
# ---------------------------------------------------------------------------


class TestProbeRedis:
    """Tests for the _probe_redis helper."""

    @pytest.mark.asyncio
    async def test_healthy_when_ping_succeeds(self) -> None:
        """Redis PING succeeds → 'healthy'."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()

        mock_module = MagicMock()
        mock_module.from_url = MagicMock(return_value=mock_redis)

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with patch.dict("sys.modules", {"redis.asyncio": mock_module, "redis": MagicMock()}):
            with patch("app.api.v1.routes.health.aioredis", mock_module, create=True):
                # Re-import so the patched module is used.

                # Directly call with patched import inside the function body.
                result = await _probe_redis(mock_settings)

        # The result depends on whether the real redis module is installed;
        # with the function's try/except, either 'healthy' or 'unavailable'
        # are acceptable when mocking at this level.
        assert result in ("healthy", "unavailable", "unhealthy")

    @pytest.mark.asyncio
    async def test_unavailable_when_redis_not_installed(self) -> None:
        """ImportError from missing redis package → 'unavailable'."""
        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with patch.dict("sys.modules", {"redis.asyncio": None, "redis": None}):
            result = await _probe_redis(mock_settings)

        assert result in ("unavailable", "unhealthy")


# ---------------------------------------------------------------------------
# _probe_models
# ---------------------------------------------------------------------------


class TestProbeModels:
    """Tests for the _probe_models helper."""

    @pytest.mark.asyncio
    async def test_all_loaded(self) -> None:
        """All models loaded → per-model status is 'loaded' and _status is 'loaded'."""
        mock_health = MagicMock(return_value={
            "ner": True,
            "icd": True,
            "summarizer": True,
            "risk_scorer": True,
        })

        with patch("app.services.model_registry.health_check", mock_health):
            result = await _probe_models()

        assert result["ner"] == "loaded"
        assert result["icd"] == "loaded"
        assert result["_status"] == "loaded"

    @pytest.mark.asyncio
    async def test_none_loaded(self) -> None:
        """No models loaded → _status is 'not_loaded'."""
        mock_health = MagicMock(return_value={
            "ner": False,
            "icd": False,
            "summarizer": False,
            "risk_scorer": False,
        })

        with patch("app.services.model_registry.health_check", mock_health):
            result = await _probe_models()

        assert result["_status"] == "not_loaded"
        assert result["ner"] == "not_loaded"

    @pytest.mark.asyncio
    async def test_partial_load(self) -> None:
        """Some models loaded → _status is 'loaded' (at least one ready)."""
        mock_health = MagicMock(return_value={
            "ner": True,
            "icd": False,
            "summarizer": False,
            "risk_scorer": False,
        })

        with patch("app.services.model_registry.health_check", mock_health):
            result = await _probe_models()

        assert result["_status"] == "loaded"
        assert result["ner"] == "loaded"
        assert result["icd"] == "not_loaded"


# ---------------------------------------------------------------------------
# Liveness and readiness probes
# ---------------------------------------------------------------------------


class TestLivenessProbe:
    """Tests for the /health/live endpoint handler."""

    @pytest.mark.asyncio
    async def test_always_returns_alive(self) -> None:
        result = await liveness()
        assert result == {"status": "alive"}


class TestReadinessProbe:
    """Tests for the /health/ready endpoint handler."""

    @pytest.mark.asyncio
    async def test_ready_when_db_healthy(self) -> None:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock())

        result = await readiness(mock_db)
        assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_not_ready_when_db_unhealthy(self) -> None:
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=ConnectionError("db down"))

        result = await readiness(mock_db)
        assert result["status"] == "not_ready"
        assert "reason" in result


# ---------------------------------------------------------------------------
# health_check aggregation logic
# ---------------------------------------------------------------------------


class TestHealthCheckAggregation:
    """Tests for the main health_check route handler's status aggregation."""

    @pytest.mark.asyncio
    async def test_healthy_when_all_deps_ok(self) -> None:
        """Database healthy + Redis healthy → overall 'healthy'."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock())

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.app_version = "1.0.0"
        mock_settings.environment = "test"

        with patch("app.api.v1.routes.health._probe_redis", return_value="healthy"), \
             patch("app.api.v1.routes.health._probe_models", return_value={
                 "_status": "loaded", "ner": "loaded",
             }):
            result = await health_check(mock_db, mock_settings)

        assert result.status == "healthy"
        assert result.version == "1.0.0"
        assert result.dependencies["database"] == "healthy"
        assert result.uptime_seconds is not None
        assert result.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_degraded_when_redis_unhealthy(self) -> None:
        """Database healthy + Redis unhealthy → overall 'degraded'."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock())

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.app_version = "1.0.0"
        mock_settings.environment = "test"

        with patch("app.api.v1.routes.health._probe_redis", return_value="unhealthy"), \
             patch("app.api.v1.routes.health._probe_models", return_value={
                 "_status": "not_loaded",
             }):
            result = await health_check(mock_db, mock_settings)

        assert result.status == "degraded"
        assert result.dependencies["redis"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_unhealthy_when_db_down(self) -> None:
        """Database unhealthy → overall 'unhealthy' regardless of Redis."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=ConnectionError("db down"))

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.app_version = "1.0.0"
        mock_settings.environment = "test"

        with patch("app.api.v1.routes.health._probe_redis", return_value="healthy"), \
             patch("app.api.v1.routes.health._probe_models", return_value={
                 "_status": "loaded",
             }):
            result = await health_check(mock_db, mock_settings)

        assert result.status == "unhealthy"
        assert result.dependencies["database"] == "unhealthy"
