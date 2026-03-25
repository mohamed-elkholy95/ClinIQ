"""Unit tests for the structured logging middleware.

Validates that every response includes X-Request-ID and X-Process-Time
headers, and that the logging configuration function runs without error.
"""

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.logging import RequestLoggingMiddleware, configure_logging


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with logging middleware."""
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ok")
    def ok_route():
        return {"status": "ok"}

    @app.get("/error")
    def error_route():
        raise ValueError("intentional boom")

    return app


class TestResponseHeaders:
    """Every response should have request-ID and timing headers."""

    def test_request_id_header_present(self):
        client = TestClient(_make_app())
        resp = client.get("/ok")
        assert "X-Request-ID" in resp.headers

    def test_request_id_is_valid_uuid(self):
        client = TestClient(_make_app())
        resp = client.get("/ok")
        rid = resp.headers["X-Request-ID"]
        # Should be parseable as UUID
        uuid.UUID(rid)

    def test_process_time_header_present(self):
        client = TestClient(_make_app())
        resp = client.get("/ok")
        assert "X-Process-Time" in resp.headers

    def test_process_time_is_numeric(self):
        client = TestClient(_make_app())
        resp = client.get("/ok")
        # Format: "1.23ms"
        raw = resp.headers["X-Process-Time"].replace("ms", "")
        float(raw)  # should not raise


class TestConfigureLogging:
    """configure_logging() should not raise for any supported format."""

    def test_json_format(self):
        configure_logging(log_level="INFO", log_format="json")

    def test_console_format(self):
        configure_logging(log_level="DEBUG", log_format="console")

    def test_default_arguments(self):
        configure_logging()
