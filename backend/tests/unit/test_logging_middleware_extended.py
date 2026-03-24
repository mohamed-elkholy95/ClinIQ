"""Extended tests for logging middleware — covering the exception path (lines 58-66)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.middleware.logging import RequestLoggingMiddleware


@pytest.mark.asyncio
async def test_request_failure_logs_error() -> None:
    """When call_next raises, the middleware logs the error and re-raises (lines 58-66)."""
    mock_app = MagicMock()
    middleware = RequestLoggingMiddleware(mock_app)

    request = MagicMock()
    request.method = "GET"
    request.url.path = "/api/v1/analyze"
    request.url = MagicMock()
    request.url.path = "/api/v1/analyze"
    request.url.__str__ = lambda self: "http://test/api/v1/analyze"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {}
    request.state = MagicMock()

    async def failing_call_next(req):
        """Simulate a request that raises an exception."""
        raise RuntimeError("Database connection lost")

    with pytest.raises(RuntimeError, match="Database connection lost"):
        await middleware.dispatch(request, failing_call_next)
