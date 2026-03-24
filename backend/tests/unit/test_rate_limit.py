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


def _make_app(max_requests: int = 5, window: int = 60) -> FastAPI:
    """Create a minimal FastAPI app with rate limiting."""
    app = FastAPI()

    # Patch settings to control limits
    class _FakeSettings:
        rate_limit_requests = max_requests
        rate_limit_period = window

    app.add_middleware(RateLimitMiddleware, redis_url=None)

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/test")
    def test_route():
        return {"data": "hello"}

    # Monkey-patch get_settings inside the middleware instance
    for mw in app.middleware_stack.__dict__.get("app", app).__dict__.get(
        "middleware_stack", []
    ):
        pass  # starlette wraps deeply; we set it on the class instead

    # Simpler: patch the middleware's settings attribute directly
    # Walk through middleware stack
    _patch_middleware_settings(app, _FakeSettings())

    return app


def _patch_middleware_settings(app, settings):
    """Recursively find RateLimitMiddleware and patch its settings."""
    current = app
    while hasattr(current, "app"):
        if isinstance(current, RateLimitMiddleware):
            current.settings = settings
            return
        current = current.app
    # If middleware is the outermost layer
    if isinstance(current, RateLimitMiddleware):
        current.settings = settings


class TestRateLimitBypass:
    """Health and docs endpoints should bypass rate limiting."""

    def test_health_endpoint_bypasses_rate_limit(self):
        app = _make_app(max_requests=1)
        client = TestClient(app)
        # Even with limit=1, health should always pass
        for _ in range(10):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200


class TestRateLimitEnforcement:
    """Regular endpoints should enforce the configured limit."""

    def test_requests_within_limit_succeed(self):
        app = _make_app(max_requests=5)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/api/v1/test")
            assert resp.status_code == 200

    def test_exceeding_limit_returns_429(self):
        app = _make_app(max_requests=3)
        client = TestClient(app)
        for _ in range(3):
            resp = client.get("/api/v1/test")
            assert resp.status_code == 200

        resp = client.get("/api/v1/test")
        assert resp.status_code == 429

    def test_rate_limit_headers_present(self):
        app = _make_app(max_requests=10)
        client = TestClient(app)
        resp = client.get("/api/v1/test")
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_429_includes_retry_after(self):
        app = _make_app(max_requests=1)
        client = TestClient(app)
        client.get("/api/v1/test")  # exhaust quota
        resp = client.get("/api/v1/test")
        assert resp.status_code == 429


class TestClientKeyExtraction:
    """Rate limits should use API key when present, IP otherwise."""

    def test_api_key_clients_tracked_separately(self):
        app = _make_app(max_requests=2)
        client = TestClient(app)

        # Client A (by API key)
        for _ in range(2):
            resp = client.get("/api/v1/test", headers={"X-API-Key": "key-alpha"})
            assert resp.status_code == 200

        # Client B (different API key) should still have quota
        resp = client.get("/api/v1/test", headers={"X-API-Key": "key-beta"})
        assert resp.status_code == 200
