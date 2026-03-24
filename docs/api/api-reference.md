# ClinIQ API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive documentation: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

---

## Authentication

ClinIQ supports two authentication methods. All endpoints except `/health` and `/docs` require authentication.

### JWT Bearer Token

Obtain a token via the login endpoint, then include it in the `Authorization` header:

```bash
# Obtain token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "clinician@example.com", "password": "your-password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}

# Use token
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with..."}'
```

Tokens expire after 30 minutes by default (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`).

### API Key

Generate an API key (requires JWT authentication), then pass it via the `X-API-Key` header:

```bash
# Generate API key
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."

# Response
{
  "api_key": "cliniq_a1b2c3d4e5f6...",
  "created_at": "2026-03-24T12:00:00Z"
}

# Use API key
curl -X POST http://localhost:8000/api/v1/ner \
  -H "X-API-Key: cliniq_a1b2c3d4e5f6..." \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient takes metformin 1000mg BID"}'
```

---

## Endpoints

### POST /api/v1/analyze

Run the full clinical NLP pipeline (NER + ICD-10 + Summarization + Risk Scoring). Each stage can be individually enabled or disabled via the `config` object.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT: Chest pain.\n\nHPI: 72-year-old male with known coronary artery disease presents with acute onset substernal chest pain radiating to the jaw, onset 2 hours ago. History of hypertension, hyperlipidaemia, and type 2 diabetes. Current medications: atorvastatin 40 mg, metoprolol 50 mg BID, metformin 1000 mg BID, aspirin 81 mg.\n\nASSESSMENT: STEMI. PLAN: Activate cath lab, heparin bolus, clopidogrel load.",
  "config": {
    "ner": { "enabled": true, "model": "rule-based", "min_confidence": 0.5 },
    "icd": { "enabled": true, "model": "sklearn-baseline", "top_k": 5 },
    "summary": { "enabled": true, "model": "extractive", "detail_level": "brief" },
    "risk": { "enabled": true }
  },
  "document_id": "pat-20260324-001",
  "store_result": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Clinical text (1 to 100,000 characters) |
| `config` | object | No | Per-stage pipeline configuration |
| `config.ner.enabled` | bool | No | Run NER stage (default: true) |
| `config.ner.model` | string | No | `rule-based`, `spacy`, or `transformer` |
| `config.ner.min_confidence` | float | No | Minimum confidence threshold 0.0-1.0 |
| `config.icd.enabled` | bool | No | Run ICD prediction (default: true) |
| `config.icd.model` | string | No | `sklearn-baseline`, `transformer`, or `hierarchical` |
| `config.icd.top_k` | int | No | Max ICD codes to return (1-50, default: 10) |
| `config.summary.enabled` | bool | No | Run summarization (default: true) |
| `config.summary.model` | string | No | `extractive` or `abstractive` |
| `config.summary.detail_level` | string | No | `brief`, `standard`, or `detailed` |
| `config.risk.enabled` | bool | No | Run risk scoring (default: true) |
| `document_id` | string | No | Client-supplied document ID (max 36 chars) |
| `store_result` | bool | No | Persist result to database (default: false) |

**Response (200):**

```json
{
  "document_id": "pat-20260324-001",
  "result_id": null,
  "text_length": 485,
  "entities": [
    {
      "text": "chest pain",
      "entity_type": "SYMPTOM",
      "start_char": 20,
      "end_char": 30,
      "confidence": 0.92,
      "normalized_text": "chest pain",
      "umls_cui": "C0008031",
      "is_negated": false,
      "is_uncertain": false
    },
    {
      "text": "atorvastatin 40 mg",
      "entity_type": "MEDICATION",
      "start_char": 262,
      "end_char": 280,
      "confidence": 0.90,
      "normalized_text": null,
      "umls_cui": null,
      "is_negated": false,
      "is_uncertain": false
    }
  ],
  "icd_codes": [
    {
      "code": "I21.9",
      "description": "Acute myocardial infarction, unspecified",
      "confidence": 0.88,
      "chapter": "Diseases of the circulatory system",
      "category": "I21",
      "contributing_text": ["STEMI", "chest pain"]
    }
  ],
  "summary": {
    "summary": "72-year-old male with CAD presents with STEMI. Activated cath lab.",
    "key_points": ["STEMI", "Coronary artery disease", "Hypertension"],
    "original_word_count": 85,
    "summary_word_count": 14,
    "compression_ratio": 6.07,
    "summary_type": "extractive",
    "model_name": "textrank",
    "model_version": "1.0.0",
    "processing_time_ms": 19.1
  },
  "risk_score": {
    "score": 0.78,
    "category": "high",
    "top_factors": [
      {
        "name": "urgency_acute",
        "description": "Clinical urgency keyword detected: 'acute'",
        "weight": 0.7,
        "value": 0.7,
        "source": "text",
        "evidence": null
      }
    ],
    "recommendations": [
      "HIGH PRIORITY: Urgent follow-up within 24-48 hours recommended",
      "Medication reconciliation recommended"
    ]
  },
  "timing": {
    "ner_ms": 11.2,
    "icd_ms": 39.5,
    "summary_ms": 19.1,
    "risk_ms": 8.3,
    "total_ms": 78.1
  }
}
```

---

### POST /api/v1/ner

Extract medical named entities from clinical text.

**Request:**

```json
{
  "text": "Patient takes metformin 1000mg BID and lisinopril 10mg daily for type 2 diabetes and hypertension.",
  "model": "rule-based",
  "min_confidence": 0.5,
  "include_negated": true,
  "include_uncertain": true
}
```

**Response (200):**

```json
{
  "entities": [
    {
      "text": "metformin",
      "entity_type": "MEDICATION",
      "start_char": 14,
      "end_char": 23,
      "confidence": 0.90,
      "normalized_text": null,
      "umls_cui": null,
      "is_negated": false,
      "is_uncertain": false
    },
    {
      "text": "1000mg",
      "entity_type": "DOSAGE",
      "start_char": 24,
      "end_char": 30,
      "confidence": 0.90,
      "normalized_text": null,
      "umls_cui": null,
      "is_negated": false,
      "is_uncertain": false
    }
  ],
  "model_name": "rule-based",
  "model_version": "1.0.0",
  "entity_count": 6,
  "processing_time_ms": 5.3
}
```

---

### POST /api/v1/icd-predict

Predict ICD-10 diagnosis codes from clinical text.

**Request:**

```json
{
  "text": "Patient diagnosed with type 2 diabetes mellitus with diabetic chronic kidney disease, stage 3.",
  "model": "sklearn-baseline",
  "top_k": 5,
  "min_confidence": 0.1
}
```

**Response (200):**

```json
{
  "predictions": [
    {
      "code": "E11.22",
      "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
      "confidence": 0.82,
      "chapter": "Endocrine, nutritional and metabolic diseases",
      "category": "E11",
      "contributing_text": ["type 2 diabetes mellitus", "diabetic chronic kidney disease"]
    },
    {
      "code": "N18.3",
      "description": "Chronic kidney disease, stage 3",
      "confidence": 0.71,
      "chapter": "Diseases of the genitourinary system",
      "category": "N18",
      "contributing_text": ["chronic kidney disease, stage 3"]
    }
  ],
  "model_name": "sklearn-baseline",
  "model_version": "1.0.0",
  "processing_time_ms": 34.7
}
```

---

### POST /api/v1/summarize

Generate a clinical summary of the input text.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT: Shortness of breath...(full clinical note)...",
  "model": "extractive",
  "detail_level": "standard"
}
```

**Response (200):**

```json
{
  "summary": "68-year-old female presents with progressive dyspnea over 3 weeks. History of CHF and COPD. Chest X-ray shows bilateral pleural effusions. Started on IV furosemide with improvement.",
  "key_points": [
    "Progressive dyspnea x 3 weeks",
    "Known CHF and COPD",
    "Bilateral pleural effusions on CXR",
    "Responding to IV diuretics"
  ],
  "original_word_count": 342,
  "summary_word_count": 31,
  "compression_ratio": 11.03,
  "summary_type": "extractive",
  "model_name": "textrank",
  "model_version": "1.0.0",
  "processing_time_ms": 22.8
}
```

| detail_level | Sentence Retention | Max Sentences |
|---|---|---|
| `brief` | 15% of original | 5 |
| `standard` | 30% of original | 12 |
| `detailed` | 50% of original | 25 |

---

### POST /api/v1/risk-score

Assess clinical risk across medication, diagnostic, and follow-up dimensions.

**Request:**

```json
{
  "text": "Patient on warfarin and aspirin with history of GI bleeding. Non-compliant with follow-up. Diagnosed with atrial fibrillation and stage 3 CKD.",
  "entities": [],
  "icd_codes": ["I48.0", "N18.3"]
}
```

**Response (200):**

```json
{
  "overall_score": 72.5,
  "risk_level": "high",
  "factors": [
    {
      "name": "interaction_aspirin_+_warfarin",
      "score": 0.85,
      "weight": 0.9,
      "category": "medication_risk",
      "description": "Potential drug interaction: aspirin + warfarin"
    },
    {
      "name": "high_risk_med_warfarin",
      "score": 0.85,
      "weight": 0.85,
      "category": "medication_risk",
      "description": "High-risk medication detected: warfarin"
    },
    {
      "name": "follow_up_non_compliant",
      "score": 0.75,
      "weight": 0.7,
      "category": "follow_up_urgency",
      "description": "Follow-up risk indicator: 'Non-compliant'"
    }
  ],
  "recommendations": [
    "HIGH PRIORITY: Urgent follow-up within 24-48 hours recommended",
    "Medication reconciliation recommended -- review for interactions and high-risk agents",
    "Proactive care coordination indicated -- patient at risk for loss to follow-up"
  ],
  "category_scores": {
    "medication_risk": 81.3,
    "diagnostic_complexity": 55.0,
    "follow_up_urgency": 52.5
  },
  "processing_time_ms": 6.1,
  "model_name": "rule-based-risk",
  "model_version": "1.0.0"
}
```

Risk levels: `low` (0-34), `moderate` (35-59), `high` (60-79), `critical` (80-100).

---

### POST /api/v1/batch

Submit multiple documents for asynchronous processing.

**Request:**

```json
{
  "documents": [
    { "document_id": "doc-001", "text": "Patient presents with..." },
    { "document_id": "doc-002", "text": "72-year-old female..." }
  ],
  "config": {
    "ner": { "enabled": true },
    "icd": { "enabled": true },
    "summary": { "enabled": false },
    "risk": { "enabled": false }
  }
}
```

**Response (202):**

```json
{
  "job_id": "batch-a1b2c3d4",
  "status": "queued",
  "document_count": 2,
  "created_at": "2026-03-24T12:00:00Z"
}
```

### GET /api/v1/batch/{job_id}

Poll the status of a batch job.

**Response (200):**

```json
{
  "job_id": "batch-a1b2c3d4",
  "status": "completed",
  "document_count": 2,
  "completed_count": 2,
  "failed_count": 0,
  "results": [
    { "document_id": "doc-001", "entities": [...], "icd_codes": [...] },
    { "document_id": "doc-002", "entities": [...], "icd_codes": [...] }
  ],
  "created_at": "2026-03-24T12:00:00Z",
  "completed_at": "2026-03-24T12:00:05Z"
}
```

Batch status values: `queued`, `processing`, `completed`, `failed`, `partial`.

---

### GET /api/v1/models

List all available ML models and their status.

**Response (200):**

```json
{
  "models": [
    {
      "name": "rule-based-ner",
      "version": "1.0.0",
      "task": "ner",
      "status": "loaded",
      "description": "Rule-based NER with regex patterns for 14 entity types"
    },
    {
      "name": "sklearn-icd-baseline",
      "version": "1.0.0",
      "task": "icd-prediction",
      "status": "available",
      "description": "TF-IDF + multi-label classifier for ICD-10 prediction"
    },
    {
      "name": "extractive-textrank",
      "version": "1.0.0",
      "task": "summarization",
      "status": "loaded",
      "description": "TextRank with clinical relevance weighting"
    },
    {
      "name": "rule-based-risk",
      "version": "1.0.0",
      "task": "risk-scoring",
      "status": "loaded",
      "description": "Weighted rule-based clinical risk scorer"
    }
  ]
}
```

---

### GET /api/v1/health

Health check endpoint. Exempt from authentication and rate limiting.

**Response (200):**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "checks": {
    "database": "connected",
    "redis": "connected",
    "models_loaded": 3
  },
  "uptime_seconds": 3600
}
```

---

## Error Codes

All errors return a JSON body with the following structure:

```json
{
  "error": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "details": {}
}
```

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request body failed Pydantic validation |
| `AUTHENTICATION_ERROR` | 401 | Missing, invalid, or expired credentials |
| `AUTHORIZATION_ERROR` | 403 | Valid credentials but insufficient permissions |
| `NOT_FOUND` | 404 | Requested resource does not exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests in the current window |
| `MODEL_NOT_FOUND` | 404 | Requested model name/version not available |
| `MODEL_LOAD_ERROR` | 503 | Model failed to load (dependency or file issue) |
| `INFERENCE_ERROR` | 500 | Model loaded but inference failed |
| `DOCUMENT_ERROR` | 400 | Document processing error (too long, encoding) |
| `BATCH_ERROR` | 400 | Batch job submission or retrieval error |
| `CONFIGURATION_ERROR` | 500 | Server misconfiguration |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `INTERNAL_ERROR` | 500 | Unhandled server error |

---

## Rate Limiting

Rate limits are enforced per API key or per IP address using a sliding window algorithm backed by Redis.

| Tier | Requests | Window | Reset |
|------|----------|--------|-------|
| Free (default) | 100 | 24 hours | Rolling |
| Pro | 10,000 | 24 hours | Rolling |

**Response Headers:**

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests allowed in the window |
| `X-RateLimit-Remaining` | Requests remaining in the current window |
| `X-RateLimit-Reset` | Unix timestamp when the window resets |
| `Retry-After` | Seconds to wait before retrying (only on 429) |

**Exempt endpoints:** `/api/v1/health`, `/docs`, `/openapi.json`

When rate limited, the API returns:

```json
{
  "detail": "Rate limit exceeded"
}
```

with HTTP status 429 and the headers listed above.
