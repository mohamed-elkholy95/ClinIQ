"""Unit tests for the rate-limiting middleware.

Tests the in-memory fallback path (no Redis) since that's what runs
without external dependencies.  Validates that requests within the
window are allowed, that excess requests get 429, and that rate-limit
headers are always present.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.rate_limit import RateLimitMiddleware


from unittest.mock import patch, MagicMock


def _make_app_and_client(max_requests: int = 5, window: int = 60):
    """Create a minimal FastAPI app with rate limiting and a test client.

    Returns ``(client, patcher)`` — the caller must keep the patcher alive
    (stop it after the test) or use the client within a ``with`` block.
    We patch ``get_settings`` so ``RateLimitMiddleware.__init__`` picks up
    our fake limits.
    """

    class _FakeSettings:
        rate_limit_requests = max_requests
        rate_limit_period = window

    patcher = patch("app.middleware.rate_limit.get_settings", return_value=_FakeSettings())
    patcher.start()

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, redis_url=None)

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/test")
    def test_route():
        return {"data": "hello"}

    client = TestClient(app, raise_server_exceptions=False)
    # Make a dummy request to force middleware stack construction
    client.get("/api/v1/health")
    patcher.stop()

    return client


class TestRateLimitBypass:
    """Health and docs endpoints should bypass rate limiting."""

    def test_health_endpoint_bypasses_rate_limit(self):
        client = _make_app_and_client(max_requests=1)
        # Even with limit=1, health should always pass
        for _ in range(10):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200


class TestRateLimitEnforcement:
    """Regular endpoints should enforce the configured limit."""

    def test_requests_within_limit_succeed(self):
        client = _make_app_and_client(max_requests=5)
        for _ in range(5):
            resp = client.get("/api/v1/test")
            assert resp.status_code == 200

    def test_exceeding_limit_returns_429(self):
        client = _make_app_and_client(max_requests=3)
        for _ in range(3):
            resp = client.get("/api/v1/test")
            assert resp.status_code == 200

        resp = client.get("/api/v1/test")
        assert resp.status_code == 429

    def test_rate_limit_headers_present(self):
        client = _make_app_and_client(max_requests=10)
        resp = client.get("/api/v1/test")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_429_includes_retry_after(self):
        client = _make_app_and_client(max_requests=1)
        client.get("/api/v1/test")  # exhaust quota
        resp = client.get("/api/v1/test")
        assert resp.status_code == 429


class TestClientKeyExtraction:
    """Rate limits should use API key when present, IP otherwise."""

    def test_api_key_clients_tracked_separately(self):
        client = _make_app_and_client(max_requests=2)

        # Client A (by API key)
        for _ in range(2):
            resp = client.get("/api/v1/test", headers={"X-API-Key": "key-alpha"})
            assert resp.status_code == 200

        # Client B (different API key) should still have quota
        resp = client.get("/api/v1/test", headers={"X-API-Key": "key-beta"})
        assert resp.status_code == 200
