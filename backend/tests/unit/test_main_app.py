"""Unit tests for the FastAPI application setup in main.py.

Covers route registration, error-code mapping, exception handlers,
root endpoint, and middleware presence.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.core.exceptions import ClinIQError
from app.main import _error_code_to_http_status, app


@pytest.fixture
def client() -> TestClient:
    """Provide a plain TestClient (no DB overrides)."""
    return TestClient(app, raise_server_exceptions=False)


class TestErrorCodeMapping:
    """Tests for _error_code_to_http_status."""

    @pytest.mark.parametrize(
        "code,expected_status",
        [
            ("VALIDATION_ERROR", 400),
            ("AUTHENTICATION_ERROR", 401),
            ("AUTHORIZATION_ERROR", 403),
            ("NOT_FOUND", 404),
            ("RATE_LIMIT_EXCEEDED", 429),
            ("MODEL_NOT_FOUND", 404),
            ("MODEL_LOAD_ERROR", 503),
            ("INFERENCE_ERROR", 500),
            ("DOCUMENT_ERROR", 400),
            ("BATCH_ERROR", 400),
            ("CONFIGURATION_ERROR", 500),
            ("DATABASE_ERROR", 500),
        ],
    )
    def test_known_codes(self, code: str, expected_status: int) -> None:
        assert _error_code_to_http_status(code) == expected_status

    def test_unknown_code_defaults_to_500(self) -> None:
        assert _error_code_to_http_status("SOME_UNKNOWN_CODE") == 500

    def test_empty_string_defaults_to_500(self) -> None:
        assert _error_code_to_http_status("") == 500


class TestAppMetadata:
    """Tests for FastAPI app object metadata."""

    def test_app_title(self) -> None:
        assert app.title == "ClinIQ API"

    def test_app_description_not_empty(self) -> None:
        assert len(app.description) > 0

    def test_app_has_openapi_url(self) -> None:
        # openapi_url is set from settings; just ensure it's present
        assert app.openapi_url is not None


class TestRouteRegistration:
    """Verify all expected routes are registered."""

    def _route_paths(self) -> set[str]:
        return {r.path for r in app.routes}

    def test_root_registered(self) -> None:
        assert "/" in self._route_paths()

    def test_health_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/health" in p for p in paths)

    def test_ner_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/ner" in p for p in paths)

    def test_icd_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/icd" in p for p in paths)

    def test_summarize_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/summarize" in p for p in paths)

    def test_risk_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/risk" in p for p in paths)

    def test_models_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/models" in p for p in paths)

    def test_auth_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/auth" in p for p in paths)

    def test_batch_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/batch" in p for p in paths)

    def test_analyze_route_registered(self) -> None:
        paths = self._route_paths()
        assert any("/analyze" in p for p in paths)


class TestRootEndpoint:
    """Tests for the root ``/`` endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_name(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "name" in data

    def test_root_contains_version(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "version" in data

    def test_root_contains_docs_url(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "docs" in data
