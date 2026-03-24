"""Unit tests for the Redis-backed rate limiting path.

Targets uncovered lines 118–132 in rate_limit.py (_check_redis and
_get_redis connection).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.middleware.rate_limit import RateLimitMiddleware


def _make_redis_pipe(zcard_count: int) -> MagicMock:
    """Create a mock Redis pipeline where methods chain and execute is async."""
    pipe = MagicMock()
    # Pipeline methods are sync (they chain), but execute() is async
    pipe.zremrangebyscore.return_value = pipe
    pipe.zadd.return_value = pipe
    pipe.zcard.return_value = pipe
    pipe.expire.return_value = pipe
    pipe.execute = AsyncMock(return_value=[None, None, zcard_count, None])
    return pipe


class TestCheckRedis:
    """Tests for the _check_redis sliding-window implementation."""

    @pytest.fixture
    def middleware(self) -> RateLimitMiddleware:
        """Create a RateLimitMiddleware instance without a real app."""
        with patch("app.middleware.rate_limit.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                rate_limit_requests=10, rate_limit_period=60
            )
            app = MagicMock()
            mw = RateLimitMiddleware(app, redis_url=None)
        return mw

    @pytest.mark.asyncio
    async def test_redis_allows_within_limit(self, middleware: RateLimitMiddleware) -> None:
        """Requests within limit should be allowed."""
        mock_redis = MagicMock()
        mock_pipe = _make_redis_pipe(zcard_count=5)
        mock_redis.pipeline.return_value = mock_pipe

        allowed, remaining, reset_at = await middleware._check_redis(
            mock_redis, "rate_limit:ip:1.2.3.4", 10, 60
        )

        assert allowed is True
        assert remaining == 5
        assert reset_at > time.time()

    @pytest.mark.asyncio
    async def test_redis_blocks_over_limit(self, middleware: RateLimitMiddleware) -> None:
        """Requests over limit should be blocked."""
        mock_redis = MagicMock()
        mock_pipe = _make_redis_pipe(zcard_count=15)
        mock_redis.pipeline.return_value = mock_pipe

        allowed, remaining, reset_at = await middleware._check_redis(
            mock_redis, "rate_limit:ip:1.2.3.4", 10, 60
        )

        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_redis_exact_limit_allowed(self, middleware: RateLimitMiddleware) -> None:
        """Exactly at the limit should still be allowed (<=)."""
        mock_redis = MagicMock()
        mock_pipe = _make_redis_pipe(zcard_count=10)
        mock_redis.pipeline.return_value = mock_pipe

        allowed, remaining, reset_at = await middleware._check_redis(
            mock_redis, "rate_limit:ip:1.2.3.4", 10, 60
        )

        assert allowed is True
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_redis_pipeline_operations(self, middleware: RateLimitMiddleware) -> None:
        """Verify Redis pipeline receives correct operations."""
        mock_redis = MagicMock()
        mock_pipe = _make_redis_pipe(zcard_count=1)
        mock_redis.pipeline.return_value = mock_pipe

        await middleware._check_redis(
            mock_redis, "rate_limit:ip:test", 10, 60
        )

        # Should call zremrangebyscore, zadd, zcard, expire
        mock_pipe.zremrangebyscore.assert_called_once()
        mock_pipe.zadd.assert_called_once()
        mock_pipe.zcard.assert_called_once()
        mock_pipe.expire.assert_called_once()
        mock_pipe.execute.assert_awaited_once()


class TestCheckLocal:
    """Edge cases for the in-memory fallback."""

    @pytest.fixture
    def middleware(self) -> RateLimitMiddleware:
        with patch("app.middleware.rate_limit.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                rate_limit_requests=10, rate_limit_period=60
            )
            app = MagicMock()
            mw = RateLimitMiddleware(app, redis_url=None)
        return mw

    def test_local_first_request_allowed(self, middleware: RateLimitMiddleware) -> None:
        """First request to a new key should be allowed."""
        allowed, remaining, reset_at = middleware._check_local(
            "rate_limit:ip:new-client", 5, 60
        )
        assert allowed is True
        assert remaining == 4

    def test_local_exact_limit_blocked(self, middleware: RateLimitMiddleware) -> None:
        """Requests at the exact limit should be blocked."""
        key = "rate_limit:ip:exact"
        for _ in range(5):
            middleware._check_local(key, 5, 60)

        allowed, remaining, reset_at = middleware._check_local(key, 5, 60)
        assert allowed is False
        assert remaining == 0


class TestGetClientKey:
    """Tests for _get_client_key extraction."""

    @pytest.fixture
    def middleware(self) -> RateLimitMiddleware:
        with patch("app.middleware.rate_limit.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                rate_limit_requests=10, rate_limit_period=60
            )
            app = MagicMock()
            mw = RateLimitMiddleware(app, redis_url=None)
        return mw

    def test_api_key_takes_precedence(self, middleware: RateLimitMiddleware) -> None:
        """When X-API-Key is present, use it for the rate limit key."""
        request = MagicMock()
        request.headers = {"X-API-Key": "sk-1234567890abcdef"}
        key = middleware._get_client_key(request)
        assert "apikey:" in key

    def _make_request(self, headers: dict, client_host: str | None = "127.0.0.1"):
        """Build a mock request with real dict-like headers."""
        request = MagicMock()
        _h = dict(headers)
        request.headers = MagicMock()
        request.headers.get = lambda k, d="": _h.get(k, d)
        if client_host is not None:
            request.client = MagicMock(host=client_host)
        else:
            request.client = None
        return request

    def test_forwarded_ip_used(self, middleware: RateLimitMiddleware) -> None:
        """X-Forwarded-For should be preferred over client.host."""
        request = self._make_request(
            {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}, client_host="127.0.0.1"
        )
        key = middleware._get_client_key(request)
        assert "10.0.0.1" in key

    def test_client_host_fallback(self, middleware: RateLimitMiddleware) -> None:
        """When no X-Forwarded-For, fall back to client.host."""
        request = self._make_request({}, client_host="192.168.1.50")
        key = middleware._get_client_key(request)
        assert "192.168.1.50" in key

    def test_no_client_uses_unknown(self, middleware: RateLimitMiddleware) -> None:
        """No client info → 'unknown'."""
        request = self._make_request({}, client_host=None)
        key = middleware._get_client_key(request)
        assert "unknown" in key
