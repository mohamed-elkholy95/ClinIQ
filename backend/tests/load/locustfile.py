"""Locust load test suite for the ClinIQ API.

Usage
-----
Run all tasks against a local dev server:

    locust -f backend/tests/load/locustfile.py \
        --host http://localhost:8000 \
        --users 20 \
        --spawn-rate 2 \
        --run-time 60s \
        --headless

Or open the web UI (default http://localhost:8089):

    locust -f backend/tests/load/locustfile.py --host http://localhost:8000

Task weights reflect approximate real-world traffic distribution:
  - Health checks            → lightweight, high frequency
  - NER extraction           → most common ML endpoint
  - ICD-10 prediction        → common ML endpoint
  - Full analysis pipeline   → heaviest, less frequent
  - Summarization            → moderate frequency
  - Risk scoring             → moderate frequency
  - Batch processing         → low frequency, high payload
"""

from __future__ import annotations

import random
from typing import Any

from locust import HttpUser, between, task


# ---------------------------------------------------------------------------
# Sample clinical texts used in load tasks
# ---------------------------------------------------------------------------

_SHORT_NOTE = (
    "Patient is a 45-year-old male with hypertension on lisinopril 10mg daily. "
    "Blood pressure today is 132/82 mmHg. Continue current regimen."
)

_MEDIUM_NOTE = """
CHIEF COMPLAINT: Routine follow-up for diabetes mellitus.

HISTORY OF PRESENT ILLNESS:
Patient is a 55-year-old female with a 10-year history of type 2 diabetes
mellitus, hypertension, and hyperlipidemia. Reports good compliance with
metformin 1000mg twice daily and atorvastatin 20mg nightly. HbA1c was 7.2%
at last visit. No recent hypoglycaemic episodes. Denies chest pain or
shortness of breath.

MEDICATIONS:
- Metformin 1000mg PO BID
- Lisinopril 10mg PO daily
- Atorvastatin 20mg PO nightly
- Aspirin 81mg PO daily

ASSESSMENT AND PLAN:
1. Type 2 DM — HbA1c trending down. Continue current regimen.
2. Hypertension — well controlled on lisinopril.
3. Hyperlipidemia — LDL at goal. Continue statin therapy.

Follow up in 3 months.
"""

_LONG_NOTE = """
HISTORY & PHYSICAL NOTE

PATIENT: John Doe | DOB: 1958-03-12 | MRN: 123456
DATE: 2026-03-24 | PROVIDER: Dr. Smith, MD

CHIEF COMPLAINT: Chest pain, shortness of breath.

HISTORY OF PRESENT ILLNESS:
Mr. Doe is a 68-year-old male with an extensive cardiac history including
coronary artery disease with prior PCI (2019), hypertension, type 2 diabetes
mellitus, hyperlipidemia, and moderate COPD (FEV1 60%). He presents with
acute onset of substernal chest pressure rated 8/10, radiating to the left
shoulder and jaw, accompanied by diaphoresis and shortness of breath. Onset
was approximately 2 hours prior to arrival while shovelling snow.

PAST MEDICAL HISTORY:
1. Coronary artery disease — PCI to LAD 2019
2. Hypertension — on ACE inhibitor
3. Type 2 diabetes mellitus — on insulin glargine 20 units nightly
4. Hyperlipidemia — on high-intensity statin
5. COPD — on tiotropium and albuterol PRN
6. Atrial fibrillation — on warfarin, INR therapeutic range 2–3
7. CKD stage 3b — creatinine 2.1 mg/dL

MEDICATIONS:
- Metformin HELD peri-procedure (CKD/contrast risk)
- Insulin glargine 20 units SQ nightly
- Lisinopril 10mg PO daily
- Atorvastatin 80mg PO nightly (high-intensity)
- Aspirin 81mg PO daily
- Clopidogrel 75mg PO daily (dual antiplatelet post-PCI)
- Warfarin 5mg PO daily — INR 2.4 on admission
- Tiotropium 18mcg inhaled daily
- Albuterol 90mcg MDI PRN
- Metoprolol succinate 50mg PO daily
- Furosemide 40mg PO daily

ALLERGIES: Penicillin — anaphylaxis; Sulfa — rash

REVIEW OF SYSTEMS:
Positive for chest pain, diaphoresis, dyspnoea. Denies nausea, vomiting,
syncope, or leg swelling.

PHYSICAL EXAMINATION:
General: Diaphoretic, distressed male.
Vitals: BP 168/98, HR 104, RR 24, SpO2 90% on RA, Temp 37.1°C.
Cardiac: Tachycardic, no murmurs.
Respiratory: Bilateral wheezing, reduced air entry at bases.

DIAGNOSTIC RESULTS:
ECG: ST elevation in leads II, III, aVF — inferior STEMI pattern.
Troponin I: 1.8 ng/mL (elevated, reference <0.04).
BNP: 480 pg/mL (elevated).
BMP: Na 138, K 4.2, BUN 28, Cr 2.1, Glucose 210.
CBC: WBC 11.2, Hgb 10.8, Plt 198.
CXR: Cardiomegaly, mild pulmonary vascular congestion.

ASSESSMENT AND PLAN:
1. ACUTE INFERIOR STEMI — Activate cath lab. STAT cardiology consult.
   Heparin drip per ACS protocol. Hold warfarin. Aspirin 325mg load given.
   Target door-to-balloon < 90 minutes.
2. ATRIAL FIBRILLATION — Currently rate-controlled. Warfarin held pre-procedure.
   Bridge with heparin. Resume post-procedure per EP guidance.
3. ACUTE ON CHRONIC KIDNEY DISEASE — IV contrast pre-hydration per protocol.
   Hold metformin. Monitor renal function post-procedure.
4. COPD — Continue tiotropium. Albuterol nebulisations Q4h PRN.
5. HYPERLIPIDEMIA — Continue high-intensity statin.
6. DIABETES — Insulin sliding scale while NPO. Glucose monitoring Q4h.
7. DISPOSITION — ICU/CCU admission. Emergent PCI.

ATTENDING: Dr. Smith, MD
"""

_CLINICAL_TEXTS = [_SHORT_NOTE, _MEDIUM_NOTE, _LONG_NOTE]


def _random_text() -> str:
    """Return a randomly chosen clinical text."""
    return random.choice(_CLINICAL_TEXTS)


def _random_short_or_medium() -> str:
    """Return a short or medium text (for lower-weight endpoints)."""
    return random.choice([_SHORT_NOTE, _MEDIUM_NOTE])


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class HealthCheckUser(HttpUser):
    """Simulates clients that poll health check endpoints frequently.

    Represents monitoring agents and load balancer probes.
    """

    weight = 3
    wait_time = between(0.5, 2.0)

    @task(5)
    def liveness(self) -> None:
        """Poll /health/live — the lightest possible endpoint."""
        with self.client.get("/api/v1/health/live", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def health(self) -> None:
        """Poll the full health check endpoint."""
        with self.client.get("/api/v1/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "healthy":
                    resp.success()
                else:
                    resp.failure(f"Unhealthy response: {data}")
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def readiness(self) -> None:
        """Poll /health/ready for Kubernetes readiness probe simulation."""
        self.client.get("/api/v1/health/ready")


class NERUser(HttpUser):
    """Simulates clients that frequently call the NER endpoint.

    NER is the most lightweight ML call and the most commonly used.
    """

    weight = 4
    wait_time = between(0.5, 3.0)

    @task(10)
    def extract_entities(self) -> None:
        """POST to /ner with a clinical text."""
        payload = {"text": _random_text()}

        with self.client.post(
            "/api/v1/ner",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "entities" in data:
                    resp.success()
                else:
                    resp.failure("Response missing 'entities' key")
            elif resp.status_code == 422:
                # Validation error — expected for short texts in some configs
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(3)
    def extract_entities_short(self) -> None:
        """POST to /ner with a short clinical text."""
        payload = {"text": _SHORT_NOTE}
        self.client.post("/api/v1/ner", json=payload)

    @task(1)
    def extract_entities_long(self) -> None:
        """POST to /ner with the long clinical note."""
        payload = {"text": _LONG_NOTE}
        self.client.post("/api/v1/ner", json=payload)


class ICDUser(HttpUser):
    """Simulates clients that call the ICD-10 prediction endpoint."""

    weight = 3
    wait_time = between(1.0, 4.0)

    @task(5)
    def predict_icd_codes(self) -> None:
        """POST to /icd/predict with a clinical text."""
        payload = {
            "text": _random_text(),
            "top_k": random.choice([3, 5, 10]),
        }
        with self.client.post(
            "/api/v1/icd/predict",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "predictions" in data:
                    resp.success()
                else:
                    resp.failure("Response missing 'predictions' key")
            elif resp.status_code in (422, 503):
                resp.success()  # Acceptable non-200 codes
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def predict_icd_top3(self) -> None:
        """POST to /icd/predict requesting top 3 codes."""
        payload = {"text": _SHORT_NOTE, "top_k": 3}
        self.client.post("/api/v1/icd/predict", json=payload)


class SummarizationUser(HttpUser):
    """Simulates clients that call the summarization endpoint."""

    weight = 2
    wait_time = between(1.5, 5.0)

    @task(5)
    def summarize_standard(self) -> None:
        """POST to /summarize at standard detail level."""
        payload = {
            "text": _random_text(),
            "detail_level": "standard",
            "max_length": 200,
        }
        with self.client.post(
            "/api/v1/summarize",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "summary" in data:
                    resp.success()
                else:
                    resp.failure("Response missing 'summary' key")
            elif resp.status_code in (422, 503):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def summarize_brief(self) -> None:
        """POST to /summarize at brief detail level."""
        payload = {
            "text": _MEDIUM_NOTE,
            "detail_level": "brief",
            "max_length": 100,
        }
        self.client.post("/api/v1/summarize", json=payload)

    @task(1)
    def summarize_detailed(self) -> None:
        """POST to /summarize at detailed level — heaviest summarization call."""
        payload = {
            "text": _LONG_NOTE,
            "detail_level": "detailed",
            "max_length": 500,
        }
        self.client.post("/api/v1/summarize", json=payload)


class RiskScoringUser(HttpUser):
    """Simulates clients that call the risk scoring endpoint."""

    weight = 2
    wait_time = between(1.0, 4.0)

    @task(5)
    def calculate_risk(self) -> None:
        """POST to /risk/score."""
        payload = {"text": _random_text()}

        with self.client.post(
            "/api/v1/risk/score",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "risk_score" in data or "overall_score" in data:
                    resp.success()
                else:
                    resp.failure("Response missing risk score field")
            elif resp.status_code in (422, 503):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def calculate_risk_long_note(self) -> None:
        """POST to /risk/score with a long, complex clinical note."""
        payload = {"text": _LONG_NOTE}
        self.client.post("/api/v1/risk/score", json=payload)


class FullAnalysisUser(HttpUser):
    """Simulates clients that invoke the full /analyze pipeline.

    This is the most expensive endpoint and should be called less frequently.
    """

    weight = 1
    wait_time = between(3.0, 10.0)

    @task(5)
    def full_analysis(self) -> None:
        """POST to /analyze with all components enabled."""
        payload = {
            "text": _random_text(),
            "enable_ner": True,
            "enable_icd": True,
            "enable_summarization": True,
            "enable_risk": True,
        }

        with self.client.post(
            "/api/v1/analyze",
            json=payload,
            catch_response=True,
            timeout=30,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "entities" in data:
                    resp.success()
                else:
                    resp.failure("Response missing 'entities' key")
            elif resp.status_code in (422, 503):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def ner_only_analysis(self) -> None:
        """POST to /analyze with only NER enabled — lighter load."""
        payload = {
            "text": _random_short_or_medium(),
            "enable_ner": True,
            "enable_icd": False,
            "enable_summarization": False,
            "enable_risk": False,
        }
        self.client.post("/api/v1/analyze", json=payload)

    @task(1)
    def full_analysis_long_note(self) -> None:
        """POST to /analyze with the longest clinical note — stress test."""
        payload = {
            "text": _LONG_NOTE,
            "enable_ner": True,
            "enable_icd": True,
            "enable_summarization": True,
            "enable_risk": True,
        }
        self.client.post("/api/v1/analyze", json=payload, timeout=60)


class BatchProcessingUser(HttpUser):
    """Simulates clients submitting batch analysis jobs.

    Batch requests are infrequent but carry large payloads.
    """

    weight = 1
    wait_time = between(5.0, 15.0)

    @task(3)
    def submit_batch_job(self) -> None:
        """POST to /batch — submit a small batch job."""
        payload = {
            "documents": [
                {"text": _SHORT_NOTE, "document_id": f"doc-{i:03d}"}
                for i in range(random.randint(2, 5))
            ],
            "enable_ner": True,
            "enable_icd": False,
            "enable_summarization": False,
            "enable_risk": True,
        }

        with self.client.post(
            "/api/v1/batch",
            json=payload,
            catch_response=True,
            timeout=30,
        ) as resp:
            if resp.status_code in (200, 202):
                resp.success()
            elif resp.status_code in (422, 503):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def submit_large_batch(self) -> None:
        """POST to /batch — submit a larger batch to stress the worker queue."""
        payload = {
            "documents": [
                {"text": _random_text(), "document_id": f"stress-{i:03d}"}
                for i in range(10)
            ],
            "enable_ner": True,
            "enable_icd": True,
            "enable_summarization": False,
            "enable_risk": False,
        }
        self.client.post("/api/v1/batch", json=payload, timeout=60)


# ---------------------------------------------------------------------------
# Mixed realistic user — combines all endpoint types
# ---------------------------------------------------------------------------


class RealisticAPIUser(HttpUser):
    """A single user that exercises the full API surface in realistic proportions.

    Use this class for a single-class load test that mirrors production traffic.
    """

    weight = 2
    wait_time = between(1.0, 5.0)

    @task(10)
    def health_check(self) -> None:
        """Lightweight health check."""
        self.client.get("/api/v1/health/live")

    @task(8)
    def ner_request(self) -> None:
        """NER extraction — most common ML endpoint."""
        self.client.post("/api/v1/ner", json={"text": _random_short_or_medium()})

    @task(5)
    def icd_prediction(self) -> None:
        """ICD-10 prediction."""
        self.client.post(
            "/api/v1/icd/predict",
            json={"text": _random_short_or_medium(), "top_k": 5},
        )

    @task(4)
    def risk_scoring(self) -> None:
        """Risk score calculation."""
        self.client.post("/api/v1/risk/score", json={"text": _random_text()})

    @task(3)
    def summarization(self) -> None:
        """Text summarization."""
        self.client.post(
            "/api/v1/summarize",
            json={"text": _random_text(), "detail_level": "standard"},
        )

    @task(2)
    def full_pipeline(self) -> None:
        """Full analysis pipeline — least frequent but most expensive."""
        self.client.post(
            "/api/v1/analyze",
            json={
                "text": _random_text(),
                "enable_ner": True,
                "enable_icd": True,
                "enable_summarization": True,
                "enable_risk": True,
            },
            timeout=30,
        )
