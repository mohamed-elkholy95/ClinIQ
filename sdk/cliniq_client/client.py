"""ClinIQ API client.

Provides typed Python methods for all 29 ClinIQ API endpoint groups,
including core analysis, specialized clinical extraction modules,
document search, and infrastructure endpoints.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from cliniq_client.models import (
    AbbreviationResult,
    AllergyResult,
    AnalysisResult,
    BatchJob,
    ClassificationResult,
    ComorbidityResult,
    EnhancedAnalysisResult,
    MedicationResult,
    QualityReport,
    RelationResult,
    SDoHResult,
    SearchResult,
    SectionResult,
    VitalSignResult,
)


class ClinIQClient:
    """Typed Python client for the ClinIQ Clinical NLP API.

    Usage::

        client = ClinIQClient(base_url="http://localhost:8000", api_key="cliniq_xxx")
        result = client.analyze("Patient has diabetes and takes metformin.")
        print(result.entities)
        print(result.icd_predictions)

    All methods return typed dataclass instances for easy programmatic use.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_prefix = f"{self.base_url}/api/v1"
        self.timeout = timeout

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> ClinIQClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ===================================================================
    # Core analysis endpoints
    # ===================================================================

    def analyze(
        self,
        text: str,
        enable_ner: bool = True,
        enable_icd: bool = True,
        enable_summarization: bool = True,
        enable_risk: bool = True,
        detail_level: str = "standard",
    ) -> AnalysisResult:
        """Run full pipeline analysis on clinical text.

        Parameters
        ----------
        text:
            Raw clinical document text.
        enable_ner:
            Enable named entity recognition.
        enable_icd:
            Enable ICD-10 prediction.
        enable_summarization:
            Enable clinical summarization.
        enable_risk:
            Enable risk scoring.
        detail_level:
            Summary detail level (``brief``, ``standard``, ``detailed``).

        Returns
        -------
        AnalysisResult
        """
        payload = {
            "text": text,
            "pipeline_config": {
                "ner": {"enabled": enable_ner},
                "icd": {"enabled": enable_icd},
                "summary": {"enabled": enable_summarization, "detail_level": detail_level},
                "risk": {"enabled": enable_risk},
            },
        }
        resp = self._post("/analyze", payload)
        return AnalysisResult.from_dict(resp)

    def analyze_enhanced(
        self,
        text: str,
        *,
        enable_classification: bool = True,
        enable_sections: bool = True,
        enable_quality: bool = True,
        enable_deidentification: bool = False,
        enable_abbreviations: bool = True,
        enable_medications: bool = True,
        enable_allergies: bool = True,
        enable_vitals: bool = True,
        enable_temporal: bool = True,
        enable_assertions: bool = True,
        enable_normalization: bool = True,
        enable_sdoh: bool = True,
        enable_relations: bool = True,
        enable_comorbidity: bool = True,
    ) -> EnhancedAnalysisResult:
        """Run the enhanced pipeline with all 14+ clinical NLP modules.

        Parameters
        ----------
        text:
            Raw clinical document text.
        enable_*:
            Toggle individual modules.

        Returns
        -------
        EnhancedAnalysisResult
        """
        payload: dict[str, Any] = {
            "text": text,
            "config": {
                "enable_classification": enable_classification,
                "enable_sections": enable_sections,
                "enable_quality": enable_quality,
                "enable_deidentification": enable_deidentification,
                "enable_abbreviations": enable_abbreviations,
                "enable_medications": enable_medications,
                "enable_allergies": enable_allergies,
                "enable_vitals": enable_vitals,
                "enable_temporal": enable_temporal,
                "enable_assertions": enable_assertions,
                "enable_normalization": enable_normalization,
                "enable_sdoh": enable_sdoh,
                "enable_relations": enable_relations,
                "enable_comorbidity": enable_comorbidity,
            },
        }
        resp = self._post("/analyze/enhanced", payload)
        return EnhancedAnalysisResult.from_dict(resp)

    def extract_entities(self, text: str, min_confidence: float = 0.5) -> list[dict]:
        """Extract medical entities from text.

        Parameters
        ----------
        text:
            Clinical text to analyze.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        list[dict]
            List of entity dicts.
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/ner", payload)
        return resp.get("entities", [])

    def predict_icd(self, text: str, top_k: int = 10) -> list[dict]:
        """Predict ICD-10 codes from clinical text.

        Parameters
        ----------
        text:
            Clinical text.
        top_k:
            Maximum number of predictions to return.

        Returns
        -------
        list[dict]
            List of ICD-10 prediction dicts.
        """
        payload = {"text": text, "top_k": top_k}
        resp = self._post("/icd-predict", payload)
        return resp.get("predictions", [])

    def summarize(self, text: str, detail_level: str = "standard") -> dict[str, Any]:
        """Generate clinical summary.

        Parameters
        ----------
        text:
            Clinical text to summarize.
        detail_level:
            Summary detail level.

        Returns
        -------
        dict
        """
        payload = {"text": text, "detail_level": detail_level}
        return self._post("/summarize", payload)

    def assess_risk(self, text: str) -> dict[str, Any]:
        """Calculate risk score.

        Parameters
        ----------
        text:
            Clinical text.

        Returns
        -------
        dict
        """
        payload = {"text": text}
        return self._post("/risk-score", payload)

    # ===================================================================
    # Document classification
    # ===================================================================

    def classify_document(
        self, text: str, *, min_confidence: float = 0.1, top_k: int = 3,
    ) -> ClassificationResult:
        """Classify clinical document type.

        Parameters
        ----------
        text:
            Document text.
        min_confidence:
            Minimum confidence for inclusion.
        top_k:
            Number of top scores to return.

        Returns
        -------
        ClassificationResult
        """
        payload = {"text": text, "min_confidence": min_confidence, "top_k": top_k}
        resp = self._post("/classify", payload)
        return ClassificationResult.from_dict(resp)

    def list_document_types(self) -> list[dict]:
        """List available document type categories.

        Returns
        -------
        list[dict]
        """
        resp = self._get("/classify/types")
        return resp.get("types", [])

    # ===================================================================
    # Medication extraction
    # ===================================================================

    def extract_medications(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> MedicationResult:
        """Extract structured medication information.

        Parameters
        ----------
        text:
            Clinical text.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        MedicationResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/medications", payload)
        return MedicationResult.from_dict(resp)

    def lookup_medication(self, drug_name: str) -> dict[str, Any]:
        """Look up a drug in the medication dictionary.

        Parameters
        ----------
        drug_name:
            Drug name (brand or generic).

        Returns
        -------
        dict
        """
        return self._get(f"/medications/lookup/{drug_name}")

    # ===================================================================
    # Allergy extraction
    # ===================================================================

    def extract_allergies(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> AllergyResult:
        """Extract allergens with reactions and severity.

        Parameters
        ----------
        text:
            Clinical text.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        AllergyResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/allergies", payload)
        return AllergyResult.from_dict(resp)

    # ===================================================================
    # Vital signs extraction
    # ===================================================================

    def extract_vitals(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> VitalSignResult:
        """Extract vital sign measurements.

        Parameters
        ----------
        text:
            Clinical text.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        VitalSignResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/vitals", payload)
        return VitalSignResult.from_dict(resp)

    def list_vital_types(self) -> list[dict]:
        """List vital sign types with standard units.

        Returns
        -------
        list[dict]
        """
        resp = self._get("/vitals/types")
        return resp.get("types", [])

    # ===================================================================
    # Section parsing
    # ===================================================================

    def parse_sections(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> SectionResult:
        """Parse clinical document into sections.

        Parameters
        ----------
        text:
            Document text.
        min_confidence:
            Minimum confidence for section detection.

        Returns
        -------
        SectionResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/sections", payload)
        return SectionResult.from_dict(resp)

    def list_section_categories(self) -> list[dict]:
        """List available section categories.

        Returns
        -------
        list[dict]
        """
        resp = self._get("/sections/categories")
        return resp.get("categories", [])

    # ===================================================================
    # Abbreviation expansion
    # ===================================================================

    def expand_abbreviations(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> AbbreviationResult:
        """Detect and expand clinical abbreviations.

        Parameters
        ----------
        text:
            Clinical text.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        AbbreviationResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/abbreviations", payload)
        return AbbreviationResult.from_dict(resp)

    def lookup_abbreviation(self, abbreviation: str) -> dict[str, Any]:
        """Look up an abbreviation in the dictionary.

        Parameters
        ----------
        abbreviation:
            Abbreviation to look up.

        Returns
        -------
        dict
        """
        return self._get(f"/abbreviations/lookup/{abbreviation}")

    # ===================================================================
    # Quality analysis
    # ===================================================================

    def analyze_quality(
        self, text: str, *, expected_sections: list[str] | None = None,
    ) -> QualityReport:
        """Analyze clinical note quality.

        Parameters
        ----------
        text:
            Clinical note text.
        expected_sections:
            Custom expected sections for completeness scoring.

        Returns
        -------
        QualityReport
        """
        payload: dict[str, Any] = {"text": text}
        if expected_sections:
            payload["expected_sections"] = expected_sections
        resp = self._post("/quality", payload)
        return QualityReport.from_dict(resp)

    # ===================================================================
    # SDoH extraction
    # ===================================================================

    def extract_sdoh(
        self, text: str, *, min_confidence: float = 0.5,
    ) -> SDoHResult:
        """Extract Social Determinants of Health.

        Parameters
        ----------
        text:
            Clinical text.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        SDoHResult
        """
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/sdoh", payload)
        return SDoHResult.from_dict(resp)

    def list_sdoh_domains(self) -> list[dict]:
        """List SDoH domain categories.

        Returns
        -------
        list[dict]
        """
        resp = self._get("/sdoh/domains")
        return resp.get("domains", [])

    # ===================================================================
    # Comorbidity scoring
    # ===================================================================

    def calculate_comorbidity(
        self,
        *,
        icd_codes: list[str] | None = None,
        text: str | None = None,
        age: int | None = None,
    ) -> ComorbidityResult:
        """Calculate Charlson Comorbidity Index.

        Parameters
        ----------
        icd_codes:
            ICD-10-CM codes.
        text:
            Clinical text for text-based extraction.
        age:
            Patient age for age-adjusted score.

        Returns
        -------
        ComorbidityResult
        """
        payload: dict[str, Any] = {}
        if icd_codes:
            payload["icd_codes"] = icd_codes
        if text:
            payload["text"] = text
        if age is not None:
            payload["age"] = age
        resp = self._post("/comorbidity", payload)
        return ComorbidityResult.from_dict(resp)

    # ===================================================================
    # Relation extraction
    # ===================================================================

    def extract_relations(
        self,
        text: str,
        entities: list[dict[str, Any]],
        *,
        min_confidence: float = 0.3,
    ) -> RelationResult:
        """Extract clinical relations between entities.

        Parameters
        ----------
        text:
            Source clinical text.
        entities:
            Pre-extracted entities with character offsets.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        RelationResult
        """
        payload = {
            "text": text,
            "entities": entities,
            "min_confidence": min_confidence,
        }
        resp = self._post("/relations", payload)
        return RelationResult.from_dict(resp)

    # ===================================================================
    # Document search
    # ===================================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        expand_query: bool = True,
        rerank: bool = True,
    ) -> SearchResult:
        """Search indexed clinical documents.

        Parameters
        ----------
        query:
            Search query.
        top_k:
            Maximum results to return.
        expand_query:
            Enable medical query expansion.
        rerank:
            Enable re-ranking of results.

        Returns
        -------
        SearchResult
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "expand_query": expand_query,
            "rerank": rerank,
        }
        resp = self._post("/search", payload)
        return SearchResult.from_dict(resp)

    # ===================================================================
    # Concept normalization
    # ===================================================================

    def normalize_concept(
        self,
        text: str,
        *,
        entity_type: str | None = None,
        min_similarity: float = 0.8,
    ) -> dict[str, Any]:
        """Normalize a medical concept to ontology codes.

        Parameters
        ----------
        text:
            Entity text to normalize.
        entity_type:
            Entity type hint for type-aware filtering.
        min_similarity:
            Minimum fuzzy match similarity.

        Returns
        -------
        dict
        """
        payload: dict[str, Any] = {
            "text": text,
            "min_similarity": min_similarity,
        }
        if entity_type:
            payload["entity_type"] = entity_type
        return self._post("/normalize", payload)

    # ===================================================================
    # Batch processing
    # ===================================================================

    def submit_batch(
        self, documents: list[dict[str, str]], config: dict | None = None,
    ) -> BatchJob:
        """Submit a batch processing job.

        Parameters
        ----------
        documents:
            List of document dicts with ``text`` and optional ``id`` keys.
        config:
            Pipeline configuration overrides.

        Returns
        -------
        BatchJob
        """
        payload = {
            "documents": documents,
            "pipeline_config": config or {},
        }
        resp = self._post("/batch", payload)
        return BatchJob(**resp)

    def get_batch_status(self, job_id: str) -> BatchJob:
        """Get batch job status.

        Parameters
        ----------
        job_id:
            Batch job identifier.

        Returns
        -------
        BatchJob
        """
        resp = self._get(f"/batch/{job_id}")
        return BatchJob(**resp)

    def wait_for_batch(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> BatchJob:
        """Wait for a batch job to complete.

        Parameters
        ----------
        job_id:
            Batch job identifier.
        poll_interval:
            Seconds between status polls.
        timeout:
            Maximum wait time in seconds.

        Returns
        -------
        BatchJob

        Raises
        ------
        TimeoutError
            If the job does not complete within *timeout*.
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_batch_status(job_id)
            if status.status in ("completed", "failed"):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Batch job {job_id} did not complete within {timeout}s")

    # ===================================================================
    # Infrastructure & utility endpoints
    # ===================================================================

    def health(self) -> dict:
        """Check API health status.

        Returns
        -------
        dict
        """
        return self._get("/health")

    def list_models(self) -> list[dict]:
        """List available ML models.

        Returns
        -------
        list[dict]
        """
        resp = self._get("/models")
        return resp.get("models", [])

    def get_metrics(self) -> dict[str, Any]:
        """Get Prometheus-style metrics.

        Returns
        -------
        dict
        """
        return self._get("/metrics")

    def get_drift_status(self) -> dict[str, Any]:
        """Get data drift monitoring status.

        Returns
        -------
        dict
        """
        return self._get("/drift/status")

    # ===================================================================
    # Authentication
    # ===================================================================

    def login(self, email: str, password: str) -> str:
        """Login and return access token.

        Parameters
        ----------
        email:
            User email.
        password:
            User password.

        Returns
        -------
        str
            Access token.
        """
        resp = self._post("/auth/token", {"email": email, "password": password})
        token = resp.get("access_token", "")
        self._client.headers["Authorization"] = f"Bearer {token}"
        return token

    # ===================================================================
    # Internal HTTP helpers
    # ===================================================================

    def _get(self, path: str) -> dict:
        """Send GET request and return parsed JSON.

        Parameters
        ----------
        path:
            API path relative to ``/api/v1``.

        Returns
        -------
        dict

        Raises
        ------
        httpx.HTTPStatusError
            On non-2xx response.
        """
        resp = self._client.get(f"{self.api_prefix}{path}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        """Send POST request with JSON body and return parsed JSON.

        Parameters
        ----------
        path:
            API path relative to ``/api/v1``.
        payload:
            JSON-serialisable request body.

        Returns
        -------
        dict

        Raises
        ------
        httpx.HTTPStatusError
            On non-2xx response.
        """
        resp = self._client.post(f"{self.api_prefix}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
