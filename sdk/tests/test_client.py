"""Tests for the ClinIQ SDK HTTP client.

Uses httpx mock transport to verify request construction, header injection,
error handling, and response parsing without hitting a real server.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cliniq_client.client import ClinIQClient
from cliniq_client.models import AnalysisResult, BatchJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a synthetic httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "http://test"),
    )


class _MockTransport(httpx.BaseTransport):
    """Transport that returns canned responses keyed by URL path suffix."""

    def __init__(self, responses: dict[str, dict]) -> None:
        self._responses = responses

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Return the pre-configured response for this request path."""
        path = request.url.path
        for suffix, data in self._responses.items():
            if path.endswith(suffix):
                return httpx.Response(200, json=data, request=request)
        return httpx.Response(404, json={"error": "not found"}, request=request)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestClientInit:
    """Tests for ClinIQClient initialisation and header setup."""

    def test_default_base_url(self) -> None:
        client = ClinIQClient()
        assert client.base_url == "http://localhost:8000"
        assert client.api_prefix == "http://localhost:8000/api/v1"
        client.close()

    def test_custom_base_url_strips_trailing_slash(self) -> None:
        client = ClinIQClient(base_url="http://example.com:9000/")
        assert client.base_url == "http://example.com:9000"
        client.close()

    def test_api_key_header(self) -> None:
        client = ClinIQClient(api_key="cliniq_test123")
        assert client._client.headers["X-API-Key"] == "cliniq_test123"
        client.close()

    def test_bearer_token_header(self) -> None:
        client = ClinIQClient(token="jwt.token.here")
        assert client._client.headers["Authorization"] == "Bearer jwt.token.here"
        client.close()

    def test_context_manager(self) -> None:
        with ClinIQClient() as client:
            assert client.base_url == "http://localhost:8000"
        # After exit, the underlying httpx client should be closed
        assert client._client.is_closed


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for the analyze() method."""

    def test_returns_analysis_result(self) -> None:
        """analyze() deserialises the response into an AnalysisResult."""
        api_response = {
            "entities": [
                {
                    "text": "metformin",
                    "entity_type": "MEDICATION",
                    "start_char": 0,
                    "end_char": 9,
                    "confidence": 0.95,
                }
            ],
            "icd_predictions": [],
            "summary": None,
            "risk_assessment": None,
            "processing_time_ms": 42.0,
            "model_versions": {"ner": "1.0.0"},
        }
        transport = _MockTransport({"/analyze": api_response})
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)

        result = client.analyze("Patient takes metformin.")
        assert isinstance(result, AnalysisResult)
        assert len(result.entities) == 1
        assert result.entities[0].text == "metformin"
        client.close()

    def test_pipeline_flags_forwarded(self) -> None:
        """Pipeline stage toggles appear in the request payload."""
        captured: dict[str, Any] = {}

        class _CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={}, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(
            transport=_CapturingTransport(), headers=client._client.headers
        )
        client.analyze(
            "test",
            enable_ner=False,
            enable_icd=True,
            enable_summarization=False,
            enable_risk=True,
            detail_level="detailed",
        )
        cfg = captured["body"]["pipeline_config"]
        assert cfg["ner"]["enabled"] is False
        assert cfg["icd"]["enabled"] is True
        assert cfg["summary"]["enabled"] is False
        assert cfg["risk"]["enabled"] is True
        assert cfg["summary"]["detail_level"] == "detailed"
        client.close()


class TestEntityExtraction:
    """Tests for extract_entities()."""

    def test_returns_entity_list(self) -> None:
        transport = _MockTransport(
            {"/ner": {"entities": [{"text": "aspirin", "type": "MED"}]}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        entities = client.extract_entities("Patient on aspirin.")
        assert len(entities) == 1
        assert entities[0]["text"] == "aspirin"
        client.close()

    def test_empty_entities(self) -> None:
        transport = _MockTransport({"/ner": {"entities": []}})
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        assert client.extract_entities("nothing here") == []
        client.close()


class TestICDPredict:
    """Tests for predict_icd()."""

    def test_returns_predictions(self) -> None:
        transport = _MockTransport(
            {"/icd-predict": {"predictions": [{"code": "I21.9", "confidence": 0.87}]}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        preds = client.predict_icd("chest pain", top_k=5)
        assert len(preds) == 1
        assert preds[0]["code"] == "I21.9"
        client.close()


class TestSummarize:
    """Tests for summarize()."""

    def test_returns_dict(self) -> None:
        transport = _MockTransport(
            {"/summarize": {"summary": "Patient stable.", "word_count": 2}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.summarize("Long clinical note...", detail_level="brief")
        assert result["summary"] == "Patient stable."
        client.close()


class TestRiskScore:
    """Tests for assess_risk()."""

    def test_returns_dict(self) -> None:
        transport = _MockTransport(
            {"/risk-score": {"score": 0.72, "category": "high"}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.assess_risk("Patient with MI and CKD.")
        assert result["score"] == 0.72
        client.close()


# ---------------------------------------------------------------------------
# Batch endpoints
# ---------------------------------------------------------------------------


class TestBatch:
    """Tests for batch submission and status polling."""

    def test_submit_batch(self) -> None:
        transport = _MockTransport(
            {
                "/batch": {
                    "job_id": "job-001",
                    "status": "pending",
                    "total_documents": 3,
                }
            }
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        job = client.submit_batch(
            [{"text": "note 1"}, {"text": "note 2"}, {"text": "note 3"}]
        )
        assert isinstance(job, BatchJob)
        assert job.job_id == "job-001"
        assert job.total_documents == 3
        client.close()

    def test_get_batch_status(self) -> None:
        transport = _MockTransport(
            {
                "/batch/job-001": {
                    "job_id": "job-001",
                    "status": "completed",
                    "total_documents": 3,
                    "processed_documents": 3,
                    "progress": 1.0,
                }
            }
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        job = client.get_batch_status("job-001")
        assert job.status == "completed"
        assert job.progress == 1.0
        client.close()

    def test_wait_for_batch_completes(self) -> None:
        """wait_for_batch returns immediately when status is 'completed'."""
        transport = _MockTransport(
            {
                "/batch/job-002": {
                    "job_id": "job-002",
                    "status": "completed",
                    "total_documents": 1,
                    "processed_documents": 1,
                    "progress": 1.0,
                }
            }
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        job = client.wait_for_batch("job-002", poll_interval=0.01, timeout=1.0)
        assert job.status == "completed"
        client.close()

    def test_wait_for_batch_timeout(self) -> None:
        """wait_for_batch raises TimeoutError when job stays pending."""
        transport = _MockTransport(
            {
                "/batch/job-003": {
                    "job_id": "job-003",
                    "status": "pending",
                    "total_documents": 1,
                }
            }
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        with pytest.raises(TimeoutError, match="job-003"):
            client.wait_for_batch("job-003", poll_interval=0.01, timeout=0.05)
        client.close()


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


class TestUtilityEndpoints:
    """Tests for health and model listing."""

    def test_health(self) -> None:
        transport = _MockTransport({"/health": {"status": "healthy"}})
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        assert client.health()["status"] == "healthy"
        client.close()

    def test_list_models(self) -> None:
        transport = _MockTransport(
            {"/models": {"models": [{"name": "ner"}, {"name": "icd"}]}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        models = client.list_models()
        assert len(models) == 2
        client.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestLogin:
    """Tests for the login() method."""

    def test_login_sets_bearer_header(self) -> None:
        transport = _MockTransport(
            {"/auth/token": {"access_token": "jwt.abc.xyz"}}
        )
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        token = client.login("user@example.com", "password123")
        assert token == "jwt.abc.xyz"
        assert client._client.headers["Authorization"] == "Bearer jwt.abc.xyz"
        client.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error propagation."""

    def test_404_raises(self) -> None:
        """Non-existent endpoints raise httpx.HTTPStatusError."""

        class _NotFoundTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(404, json={"error": "not found"}, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(
            transport=_NotFoundTransport(), headers=client._client.headers
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.health()
        client.close()

    def test_500_raises(self) -> None:
        """Server errors raise httpx.HTTPStatusError."""

        class _ServerErrorTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                return httpx.Response(500, json={"error": "internal"}, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(
            transport=_ServerErrorTransport(), headers=client._client.headers
        )
        with pytest.raises(httpx.HTTPStatusError):
            client.analyze("test")
        client.close()
