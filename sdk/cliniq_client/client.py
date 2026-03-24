"""ClinIQ API client."""

import time
from typing import Any

import httpx

from cliniq_client.models import AnalysisResult, BatchJob


class ClinIQClient:
    """Typed Python client for the ClinIQ Clinical NLP API.

    Usage:
        client = ClinIQClient(base_url="http://localhost:8000", api_key="cliniq_xxx")
        result = client.analyze("Patient has diabetes and takes metformin.")
        print(result.entities)
        print(result.icd_predictions)
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Core endpoints ---

    def analyze(
        self,
        text: str,
        enable_ner: bool = True,
        enable_icd: bool = True,
        enable_summarization: bool = True,
        enable_risk: bool = True,
        detail_level: str = "standard",
    ) -> AnalysisResult:
        """Run full pipeline analysis on clinical text."""
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

    def extract_entities(self, text: str, min_confidence: float = 0.5) -> list[dict]:
        """Extract medical entities from text."""
        payload = {"text": text, "min_confidence": min_confidence}
        resp = self._post("/ner", payload)
        return resp.get("entities", [])

    def predict_icd(self, text: str, top_k: int = 10) -> list[dict]:
        """Predict ICD-10 codes from clinical text."""
        payload = {"text": text, "top_k": top_k}
        resp = self._post("/icd-predict", payload)
        return resp.get("predictions", [])

    def summarize(
        self, text: str, detail_level: str = "standard"
    ) -> dict[str, Any]:
        """Generate clinical summary."""
        payload = {"text": text, "detail_level": detail_level}
        return self._post("/summarize", payload)

    def assess_risk(self, text: str) -> dict[str, Any]:
        """Calculate risk score."""
        payload = {"text": text}
        return self._post("/risk-score", payload)

    def submit_batch(
        self, documents: list[dict[str, str]], config: dict | None = None
    ) -> BatchJob:
        """Submit a batch processing job."""
        payload = {
            "documents": documents,
            "pipeline_config": config or {},
        }
        resp = self._post("/batch", payload)
        return BatchJob(**resp)

    def get_batch_status(self, job_id: str) -> BatchJob:
        """Get batch job status."""
        resp = self._get(f"/batch/{job_id}")
        return BatchJob(**resp)

    def wait_for_batch(
        self, job_id: str, poll_interval: float = 2.0, timeout: float = 600.0
    ) -> BatchJob:
        """Wait for a batch job to complete."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_batch_status(job_id)
            if status.status in ("completed", "failed"):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Batch job {job_id} did not complete within {timeout}s")

    def list_models(self) -> list[dict]:
        """List available models."""
        resp = self._get("/models")
        return resp.get("models", [])

    def health(self) -> dict:
        """Check API health."""
        return self._get("/health")

    # --- Auth ---

    def login(self, email: str, password: str) -> str:
        """Login and return access token."""
        resp = self._post("/auth/token", {"email": email, "password": password})
        token = resp.get("access_token", "")
        self._client.headers["Authorization"] = f"Bearer {token}"
        return token

    # --- Internal ---

    def _get(self, path: str) -> dict:
        resp = self._client.get(f"{self.api_prefix}{path}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        resp = self._client.post(f"{self.api_prefix}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()
