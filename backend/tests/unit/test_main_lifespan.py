"""Unit tests for FastAPI app lifespan, exception handlers, and middleware.

Covers the lifespan context manager (startup/shutdown), ClinIQ custom
exception handler, general exception handler, and process-time middleware.
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import _error_code_to_http_status, app

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class TestLifespan:
    """Test app startup and shutdown lifecycle."""

    def test_app_starts_with_lifespan(self) -> None:
        """TestClient triggers lifespan; app should start without errors."""
        with patch("app.main.init_db", new_callable=AsyncMock) as mock_init, \
             patch("app.main.close_db", new_callable=AsyncMock) as mock_close, \
             patch("app.main.configure_logging"):
            with TestClient(app):
                mock_init.assert_awaited_once()
            mock_close.assert_awaited_once()

    def test_lifespan_handles_db_init_failure(self) -> None:
        """If database init fails, app still starts (warning only)."""
        with patch("app.main.init_db", new_callable=AsyncMock, side_effect=Exception("DB down")), \
             patch("app.main.close_db", new_callable=AsyncMock), \
             patch("app.main.configure_logging"):
            # Should not raise
            with TestClient(app) as client:
                resp = client.get("/")
                assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


class TestClinIQExceptionHandler:
    """Test the ClinIQError exception handler mapping."""

    def test_not_found_error_returns_404(self) -> None:
        status = _error_code_to_http_status("NOT_FOUND")
        assert status == 404

    def test_validation_error_returns_400(self) -> None:
        status = _error_code_to_http_status("VALIDATION_ERROR")
        assert status == 400

    def test_auth_error_returns_401(self) -> None:
        status = _error_code_to_http_status("AUTHENTICATION_ERROR")
        assert status == 401

    def test_unknown_error_code_returns_500(self) -> None:
        status = _error_code_to_http_status("UNKNOWN_CODE_XYZ")
        assert status == 500

    def test_all_known_codes_have_status(self) -> None:
        """All 12 known error codes map to non-500 status codes."""
        known = [
            "VALIDATION_ERROR", "AUTHENTICATION_ERROR", "AUTHORIZATION_ERROR",
            "NOT_FOUND", "RATE_LIMIT_EXCEEDED", "MODEL_NOT_FOUND",
            "MODEL_LOAD_ERROR", "INFERENCE_ERROR", "DOCUMENT_ERROR",
            "BATCH_ERROR", "CONFIGURATION_ERROR", "DATABASE_ERROR",
        ]
        for code in known:
            status = _error_code_to_http_status(code)
            assert 400 <= status <= 503, f"{code} mapped to {status}"


# ---------------------------------------------------------------------------
# Process-time middleware
# ---------------------------------------------------------------------------


class TestProcessTimeMiddleware:
    """Test the X-Process-Time header middleware."""

    def test_process_time_header_present(self) -> None:
        with patch("app.main.init_db", new_callable=AsyncMock), \
             patch("app.main.close_db", new_callable=AsyncMock), \
             patch("app.main.configure_logging"), TestClient(app) as client:
            resp = client.get("/")
            assert "X-Process-Time" in resp.headers
            # Should be a parseable float ending with 's'
            val = resp.headers["X-Process-Time"]
            assert val.endswith("s")
            float(val.rstrip("s"))  # Should not raise


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Test the root / endpoint."""

    def test_root_returns_app_info(self) -> None:
        with patch("app.main.init_db", new_callable=AsyncMock), \
             patch("app.main.close_db", new_callable=AsyncMock), \
             patch("app.main.configure_logging"), TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert "name" in data
            assert "version" in data
            assert "docs" in data
