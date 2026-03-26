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
from cliniq_client.models import (
    AbbreviationResult,
    AllergyResult,
    AnalysisResult,
    AUPRCResult,
    BatchJob,
    ClassificationEvalResult,
    ClassificationResult,
    ComorbidityResult,
    ConversationContext,
    ConversationSessionInfo,
    ConversationStats,
    ConversationTurnResult,
    EnhancedAnalysisResult,
    ICDEvalResult,
    KappaResult,
    MedicationResult,
    NEREvalResult,
    QualityReport,
    RelationResult,
    ROUGEEvalResult,
    SDoHResult,
    SearchResult,
    SectionResult,
    VitalSignResult,
)


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


# ---------------------------------------------------------------------------
# Enhanced analysis
# ---------------------------------------------------------------------------


class TestEnhancedAnalyze:
    """Tests for the analyze_enhanced() method."""

    def test_returns_enhanced_result(self) -> None:
        transport = _MockTransport({
            "/analyze/enhanced": {
                "classification": {"predicted_type": "progress_note"},
                "sections": {"section_count": 5},
                "medications": {"medication_count": 3},
                "processing_time_ms": 250.0,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.analyze_enhanced("Clinical note text...")
        assert isinstance(result, EnhancedAnalysisResult)
        assert result.classification["predicted_type"] == "progress_note"
        assert result.processing_time_ms == 250.0
        client.close()

    def test_module_toggles_forwarded(self) -> None:
        captured: dict = {}

        class _Capture(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={}, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(transport=_Capture(), headers=client._client.headers)
        client.analyze_enhanced(
            "test", enable_deidentification=True, enable_medications=False,
        )
        cfg = captured["body"]["config"]
        assert cfg["enable_deidentification"] is True
        assert cfg["enable_medications"] is False
        client.close()


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


class TestClassifyDocument:
    def test_returns_classification_result(self) -> None:
        transport = _MockTransport({
            "/classify": {
                "predicted_type": "discharge_summary",
                "scores": [{"document_type": "discharge_summary", "confidence": 0.92}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.classify_document("Discharge Summary: ...")
        assert isinstance(result, ClassificationResult)
        assert result.predicted_type == "discharge_summary"
        client.close()

    def test_list_document_types(self) -> None:
        transport = _MockTransport({
            "/classify/types": {"types": [{"name": "progress_note"}]},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        types = client.list_document_types()
        assert len(types) == 1
        client.close()


# ---------------------------------------------------------------------------
# Medication extraction
# ---------------------------------------------------------------------------


class TestMedications:
    def test_extract_medications(self) -> None:
        transport = _MockTransport({
            "/medications": {
                "medication_count": 1,
                "medications": [{"drug_name": "metformin", "confidence": 0.9}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.extract_medications("Metformin 1000mg BID")
        assert isinstance(result, MedicationResult)
        assert result.medications[0].drug_name == "metformin"
        client.close()

    def test_lookup_medication(self) -> None:
        transport = _MockTransport({
            "/medications/lookup/aspirin": {"generic_name": "aspirin", "brands": ["Bayer"]},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.lookup_medication("aspirin")
        assert result["generic_name"] == "aspirin"
        client.close()


# ---------------------------------------------------------------------------
# Allergy extraction
# ---------------------------------------------------------------------------


class TestAllergies:
    def test_extract_allergies(self) -> None:
        transport = _MockTransport({
            "/allergies": {
                "allergy_count": 1,
                "no_known_allergies": False,
                "allergies": [{"allergen": "penicillin", "severity": "severe"}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.extract_allergies("Allergic to penicillin")
        assert isinstance(result, AllergyResult)
        assert result.allergies[0].allergen == "penicillin"
        client.close()


# ---------------------------------------------------------------------------
# Vital signs extraction
# ---------------------------------------------------------------------------


class TestVitals:
    def test_extract_vitals(self) -> None:
        transport = _MockTransport({
            "/vitals": {
                "vital_count": 1,
                "vitals": [{"vital_type": "heart_rate", "value": 92, "unit": "bpm"}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.extract_vitals("HR 92 bpm")
        assert isinstance(result, VitalSignResult)
        assert result.vital_count == 1
        client.close()

    def test_list_vital_types(self) -> None:
        transport = _MockTransport({
            "/vitals/types": {"types": [{"name": "heart_rate", "unit": "bpm"}]},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        types = client.list_vital_types()
        assert len(types) == 1
        client.close()


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


class TestSections:
    def test_parse_sections(self) -> None:
        transport = _MockTransport({
            "/sections": {
                "section_count": 2,
                "sections": [
                    {"category": "chief_complaint", "header": "CC:", "confidence": 1.0},
                ],
                "categories_found": ["chief_complaint", "hpi"],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.parse_sections("CC: Chest pain")
        assert isinstance(result, SectionResult)
        assert result.section_count == 2
        client.close()

    def test_list_categories(self) -> None:
        transport = _MockTransport({
            "/sections/categories": {"categories": [{"name": "chief_complaint"}]},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        cats = client.list_section_categories()
        assert len(cats) == 1
        client.close()


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------


class TestAbbreviations:
    def test_expand_abbreviations(self) -> None:
        transport = _MockTransport({
            "/abbreviations": {
                "total_found": 1,
                "expanded_text": "hypertension",
                "matches": [{"abbreviation": "HTN", "expansion": "hypertension"}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.expand_abbreviations("Patient has HTN")
        assert isinstance(result, AbbreviationResult)
        assert result.total_found == 1
        client.close()

    def test_lookup_abbreviation(self) -> None:
        transport = _MockTransport({
            "/abbreviations/lookup/HTN": {"abbreviation": "HTN", "expansion": "hypertension"},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.lookup_abbreviation("HTN")
        assert result["expansion"] == "hypertension"
        client.close()


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------


class TestQuality:
    def test_analyze_quality(self) -> None:
        transport = _MockTransport({
            "/quality": {
                "overall_score": 85.0, "grade": "B",
                "dimensions": [], "recommendation_count": 1,
                "top_recommendations": ["Add vitals"],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.analyze_quality("Full clinical note...")
        assert isinstance(result, QualityReport)
        assert result.grade == "B"
        client.close()


# ---------------------------------------------------------------------------
# SDoH extraction
# ---------------------------------------------------------------------------


class TestSDoH:
    def test_extract_sdoh(self) -> None:
        transport = _MockTransport({
            "/sdoh": {
                "extraction_count": 1, "adverse_count": 1, "protective_count": 0,
                "extractions": [{"domain": "substance_use", "text": "current smoker"}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.extract_sdoh("Current smoker, 1 PPD")
        assert isinstance(result, SDoHResult)
        assert result.extraction_count == 1
        client.close()

    def test_list_domains(self) -> None:
        transport = _MockTransport({
            "/sdoh/domains": {"domains": [{"name": "housing"}, {"name": "employment"}]},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        domains = client.list_sdoh_domains()
        assert len(domains) == 2
        client.close()


# ---------------------------------------------------------------------------
# Comorbidity scoring
# ---------------------------------------------------------------------------


class TestComorbidity:
    def test_calculate(self) -> None:
        transport = _MockTransport({
            "/comorbidity": {
                "raw_score": 3, "risk_group": "moderate",
                "ten_year_mortality": 0.52, "category_count": 2,
                "matched_categories": [],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.calculate_comorbidity(
            icd_codes=["E11.9", "I10"], text="diabetes and HTN", age=65,
        )
        assert isinstance(result, ComorbidityResult)
        assert result.raw_score == 3
        client.close()


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------


class TestRelations:
    def test_extract_relations(self) -> None:
        transport = _MockTransport({
            "/relations": {
                "relation_count": 1, "pair_count": 1,
                "relations": [
                    {"subject": "metoprolol", "subject_type": "MEDICATION",
                     "object": "HTN", "object_type": "DISEASE",
                     "relation_type": "treats", "confidence": 0.87},
                ],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.extract_relations(
            "metoprolol treats HTN",
            entities=[
                {"text": "metoprolol", "entity_type": "MEDICATION", "start_char": 0, "end_char": 10},
                {"text": "HTN", "entity_type": "DISEASE", "start_char": 18, "end_char": 21},
            ],
        )
        assert isinstance(result, RelationResult)
        assert result.relations[0].relation_type == "treats"
        client.close()


# ---------------------------------------------------------------------------
# Document search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search(self) -> None:
        transport = _MockTransport({
            "/search": {
                "hits": [{"document_id": "doc-1", "score": 0.92, "snippet": "..."}],
                "total": 1, "reranked": True,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.search("diabetes management")
        assert isinstance(result, SearchResult)
        assert result.total == 1
        assert result.reranked is True
        client.close()


# ---------------------------------------------------------------------------
# Concept normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_concept(self) -> None:
        transport = _MockTransport({
            "/normalize": {
                "matched": True, "cui": "C0020538",
                "preferred_term": "Hypertensive disease",
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.normalize_concept("hypertension", entity_type="DISEASE")
        assert result["cui"] == "C0020538"
        client.close()


# ---------------------------------------------------------------------------
# Infrastructure endpoints
# ---------------------------------------------------------------------------


class TestInfrastructure:
    def test_get_metrics(self) -> None:
        transport = _MockTransport({"/metrics": {"inference_count": 1234}})
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.get_metrics()
        assert result["inference_count"] == 1234
        client.close()

    def test_get_drift_status(self) -> None:
        transport = _MockTransport({"/drift/status": {"status": "stable"}})
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.get_drift_status()
        assert result["status"] == "stable"
        client.close()


# ---------------------------------------------------------------------------
# Evaluation endpoints
# ---------------------------------------------------------------------------


class TestEvaluateClassification:
    """Tests for evaluate_classification()."""

    def test_returns_result_without_calibration(self) -> None:
        transport = _MockTransport({
            "/evaluate/classification": {
                "mcc": 0.85, "tp": 40, "fp": 5, "fn": 3, "tn": 52,
                "calibration": None, "processing_time_ms": 1.2,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_classification(
            y_true=[1, 0, 1], y_pred=[1, 0, 1],
        )
        assert isinstance(result, ClassificationEvalResult)
        assert result.mcc == 0.85
        assert result.tp == 40
        assert result.calibration is None
        client.close()

    def test_returns_result_with_calibration(self) -> None:
        transport = _MockTransport({
            "/evaluate/classification": {
                "mcc": 0.72, "tp": 30, "fp": 8, "fn": 5, "tn": 57,
                "calibration": {
                    "expected_calibration_error": 0.03,
                    "brier_score": 0.12,
                    "n_bins": 10,
                },
                "processing_time_ms": 2.5,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_classification(
            y_true=[1, 0, 1], y_pred=[1, 0, 0],
            y_prob=[0.9, 0.2, 0.6], n_calibration_bins=10,
        )
        assert result.calibration is not None
        assert result.calibration["brier_score"] == 0.12
        client.close()

    def test_payload_includes_y_prob(self) -> None:
        captured: dict = {}

        class _Capture(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={"mcc": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(transport=_Capture(), headers=client._client.headers)
        client.evaluate_classification(
            y_true=[1, 0], y_pred=[1, 1], y_prob=[0.8, 0.6],
        )
        assert "y_prob" in captured["body"]
        assert captured["body"]["y_prob"] == [0.8, 0.6]
        client.close()


class TestEvaluateAgreement:
    """Tests for evaluate_agreement()."""

    def test_returns_kappa_result(self) -> None:
        transport = _MockTransport({
            "/evaluate/agreement": {
                "kappa": 0.82, "observed_agreement": 0.91,
                "expected_agreement": 0.50, "n_items": 100,
                "processing_time_ms": 0.5,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_agreement(
            rater_a=["A", "B", "A"], rater_b=["A", "B", "B"],
        )
        assert isinstance(result, KappaResult)
        assert result.kappa == 0.82
        assert result.n_items == 100
        client.close()


class TestEvaluateNER:
    """Tests for evaluate_ner()."""

    def test_returns_ner_eval_result(self) -> None:
        transport = _MockTransport({
            "/evaluate/ner": {
                "exact_f1": 0.75, "partial_f1": 0.88,
                "type_weighted_f1": 0.82, "mean_overlap": 0.91,
                "n_gold": 10, "n_pred": 12,
                "n_exact_matches": 7, "n_partial_matches": 2,
                "n_unmatched_pred": 3, "n_unmatched_gold": 1,
                "processing_time_ms": 0.8,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_ner(
            gold_entities=[{"entity_type": "DISEASE", "start": 0, "end": 8}],
            pred_entities=[{"entity_type": "DISEASE", "start": 0, "end": 10}],
            overlap_threshold=0.5,
        )
        assert isinstance(result, NEREvalResult)
        assert result.exact_f1 == 0.75
        assert result.partial_f1 == 0.88
        assert result.n_gold == 10
        client.close()

    def test_overlap_threshold_in_payload(self) -> None:
        captured: dict = {}

        class _Capture(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={
                    "exact_f1": 0, "partial_f1": 0, "type_weighted_f1": 0,
                    "mean_overlap": 0, "n_gold": 0, "n_pred": 0,
                    "n_exact_matches": 0, "n_partial_matches": 0,
                    "n_unmatched_pred": 0, "n_unmatched_gold": 0,
                }, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(transport=_Capture(), headers=client._client.headers)
        client.evaluate_ner([], [], overlap_threshold=0.7)
        assert captured["body"]["overlap_threshold"] == 0.7
        client.close()


class TestEvaluateROUGE:
    """Tests for evaluate_rouge()."""

    def test_returns_rouge_result(self) -> None:
        transport = _MockTransport({
            "/evaluate/rouge": {
                "rouge1": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
                "rouge2": {"precision": 0.6, "recall": 0.55, "f1": 0.57},
                "rougeL": {"precision": 0.7, "recall": 0.65, "f1": 0.67},
                "reference_length": 50, "hypothesis_length": 45,
                "length_ratio": 0.9, "processing_time_ms": 1.0,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_rouge(
            reference="The patient has diabetes.",
            hypothesis="Patient has diabetes.",
        )
        assert isinstance(result, ROUGEEvalResult)
        assert result.rouge1.f1 == 0.77
        assert result.rouge2.precision == 0.6
        assert result.rougeL.recall == 0.65
        assert result.length_ratio == 0.9
        client.close()


class TestEvaluateICD:
    """Tests for evaluate_icd()."""

    def test_returns_icd_eval_result(self) -> None:
        transport = _MockTransport({
            "/evaluate/icd": {
                "full_code_accuracy": 0.65, "block_accuracy": 0.80,
                "chapter_accuracy": 0.95, "n_samples": 100,
                "full_code_matches": 65, "block_matches": 80,
                "chapter_matches": 95, "processing_time_ms": 0.3,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_icd(
            gold_codes=["E11.65", "I10"], pred_codes=["E11.9", "I10"],
        )
        assert isinstance(result, ICDEvalResult)
        assert result.full_code_accuracy == 0.65
        assert result.chapter_accuracy == 0.95
        assert result.n_samples == 100
        client.close()


class TestEvaluateAUPRC:
    """Tests for evaluate_auprc()."""

    def test_returns_auprc_result(self) -> None:
        transport = _MockTransport({
            "/evaluate/auprc": {
                "label": "disease", "auprc": 0.92,
                "n_positive": 30, "n_total": 100,
                "processing_time_ms": 0.4,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.evaluate_auprc(
            y_true=[1, 0, 1], y_scores=[0.9, 0.1, 0.8], label="disease",
        )
        assert isinstance(result, AUPRCResult)
        assert result.auprc == 0.92
        assert result.label == "disease"
        assert result.n_positive == 30
        client.close()

    def test_default_label(self) -> None:
        captured: dict = {}

        class _Capture(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={
                    "label": "positive", "auprc": 0.5, "n_positive": 0, "n_total": 0,
                }, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(transport=_Capture(), headers=client._client.headers)
        client.evaluate_auprc(y_true=[0, 1], y_scores=[0.3, 0.7])
        assert captured["body"]["label"] == "positive"
        client.close()


class TestListEvaluationMetrics:
    """Tests for list_evaluation_metrics()."""

    def test_returns_metric_list(self) -> None:
        transport = _MockTransport({
            "/evaluate/metrics": {
                "metrics": [
                    {"name": "classification", "endpoint": "/evaluate/classification"},
                    {"name": "agreement", "endpoint": "/evaluate/agreement"},
                ],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        metrics = client.list_evaluation_metrics()
        assert len(metrics) == 2
        assert metrics[0]["name"] == "classification"
        client.close()


# ---------------------------------------------------------------------------
# Conversation memory endpoints
# ---------------------------------------------------------------------------


class TestAddConversationTurn:
    """Tests for add_conversation_turn()."""

    def test_returns_turn_result(self) -> None:
        transport = _MockTransport({
            "/conversation/turns": {
                "session_id": "sess-001", "turn_id": 1, "turn_count": 1,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.add_conversation_turn(
            session_id="sess-001",
            text="Patient presents with chest pain.",
        )
        assert isinstance(result, ConversationTurnResult)
        assert result.session_id == "sess-001"
        assert result.turn_id == 1
        client.close()

    def test_optional_fields_forwarded(self) -> None:
        captured: dict = {}

        class _Capture(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={
                    "session_id": "s1", "turn_id": 1, "turn_count": 1,
                }, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(transport=_Capture(), headers=client._client.headers)
        client.add_conversation_turn(
            session_id="s1", text="test",
            entities=[{"text": "HTN", "entity_type": "DISEASE", "confidence": 0.9}],
            icd_codes=[{"code": "I10", "description": "HTN"}],
            risk_score=0.65,
            summary="Hypertension noted.",
            document_id="doc-42",
            metadata={"source": "ER"},
        )
        body = captured["body"]
        assert body["entities"][0]["text"] == "HTN"
        assert body["risk_score"] == 0.65
        assert body["document_id"] == "doc-42"
        assert body["metadata"]["source"] == "ER"
        client.close()


class TestGetConversationContext:
    """Tests for get_conversation_context()."""

    def test_returns_context(self) -> None:
        transport = _MockTransport({
            "/conversation/context": {
                "session_id": "sess-001", "turn_count": 3,
                "unique_entities": ["diabetes", "metformin"],
                "unique_icd_codes": ["E11.9"],
                "overall_risk_trend": [0.3, 0.5, 0.7],
                "context": [{"turn_id": 1}],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.get_conversation_context("sess-001", last_n=3)
        assert isinstance(result, ConversationContext)
        assert result.turn_count == 3
        assert "diabetes" in result.unique_entities
        assert len(result.overall_risk_trend) == 3
        client.close()


class TestClearConversation:
    """Tests for clear_conversation()."""

    def test_returns_confirmation(self) -> None:
        class _DeleteTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                assert request.method == "DELETE"
                return httpx.Response(200, json={
                    "session_id": "sess-001", "turns_cleared": 5,
                }, request=request)

        client = ClinIQClient()
        client._client = httpx.Client(
            transport=_DeleteTransport(), headers=client._client.headers
        )
        result = client.clear_conversation("sess-001")
        assert result["turns_cleared"] == 5
        client.close()


class TestConversationStats:
    """Tests for get_conversation_stats()."""

    def test_returns_stats(self) -> None:
        transport = _MockTransport({
            "/conversation/stats": {
                "active_sessions": 12, "total_turns": 87,
                "max_turns_per_session": 50, "session_ttl_seconds": 7200.0,
                "max_sessions": 5000,
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        result = client.get_conversation_stats()
        assert isinstance(result, ConversationStats)
        assert result.active_sessions == 12
        assert result.total_turns == 87
        client.close()


class TestListConversationSessions:
    """Tests for list_conversation_sessions()."""

    def test_returns_sessions(self) -> None:
        transport = _MockTransport({
            "/conversation/sessions": {
                "sessions": [
                    {
                        "session_id": "sess-001", "turn_count": 5,
                        "oldest_turn_id": 1, "newest_turn_id": 5,
                        "last_access": "2026-03-26T10:00:00Z",
                    },
                    {
                        "session_id": "sess-002", "turn_count": 2,
                        "oldest_turn_id": 1, "newest_turn_id": 2,
                        "last_access": "2026-03-26T09:30:00Z",
                    },
                ],
            },
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        sessions = client.list_conversation_sessions()
        assert len(sessions) == 2
        assert isinstance(sessions[0], ConversationSessionInfo)
        assert sessions[0].session_id == "sess-001"
        assert sessions[0].turn_count == 5
        assert sessions[1].session_id == "sess-002"
        client.close()

    def test_empty_sessions(self) -> None:
        transport = _MockTransport({
            "/conversation/sessions": {"sessions": []},
        })
        client = ClinIQClient()
        client._client = httpx.Client(transport=transport, headers=client._client.headers)
        sessions = client.list_conversation_sessions()
        assert sessions == []
        client.close()
