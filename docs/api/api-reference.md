# ClinIQ API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive documentation: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

---

## Table of Contents

1. [Authentication](#authentication)
2. [Core Analysis Endpoints](#core-analysis-endpoints)
   - [POST /analyze](#post-apiv1analyze)
   - [POST /analyze/stream](#post-apiv1analyzestream)
   - [POST /analyze/enhanced](#post-apiv1analyzeenhanced)
   - [POST /analyze/enhanced/batch](#post-apiv1analyzeenhancedbatch)
   - [GET /analyze/enhanced/modules](#get-apiv1analyzeenhancedmodules)
3. [Named Entity Recognition](#named-entity-recognition)
   - [POST /ner](#post-apiv1ner)
4. [ICD-10 Code Prediction](#icd-10-code-prediction)
   - [POST /icd-predict](#post-apiv1icd-predict)
   - [GET /icd-codes/{code}](#get-apiv1icd-codescode)
5. [Clinical Summarization](#clinical-summarization)
   - [POST /summarize](#post-apiv1summarize)
6. [Risk Scoring](#risk-scoring)
   - [POST /risk-score](#post-apiv1risk-score)
7. [Document Classification](#document-classification)
   - [POST /classify](#post-apiv1classify)
   - [POST /classify/batch](#post-apiv1classifybatch)
   - [GET /classify/types](#get-apiv1classifytypes)
8. [Medication Extraction](#medication-extraction)
   - [POST /medications](#post-apiv1medications)
   - [POST /medications/batch](#post-apiv1medicationsbatch)
   - [GET /medications/lookup/{drug_name}](#get-apiv1medicationslookupdrugname)
   - [GET /medications/dictionary/stats](#get-apiv1medicationsdictionarystats)
9. [Allergy Extraction](#allergy-extraction)
   - [POST /allergies](#post-apiv1allergies)
   - [POST /allergies/batch](#post-apiv1allergiesbatch)
   - [GET /allergies/dictionary/stats](#get-apiv1allergiesdictionarystats)
   - [GET /allergies/categories](#get-apiv1allergiescategories)
10. [Vital Signs Extraction](#vital-signs-extraction)
    - [POST /vitals](#post-apiv1vitals)
    - [POST /vitals/batch](#post-apiv1vitalsbatch)
    - [GET /vitals/types](#get-apiv1vitalstypes)
    - [GET /vitals/ranges](#get-apiv1vitalsranges)
11. [Clinical Section Parsing](#clinical-section-parsing)
    - [POST /sections](#post-apiv1sections)
    - [POST /sections/batch](#post-apiv1sectionsbatch)
    - [POST /sections/query](#post-apiv1sectionsquery)
    - [GET /sections/categories](#get-apiv1sectionscategories)
12. [Abbreviation Expansion](#abbreviation-expansion)
    - [POST /abbreviations](#post-apiv1abbreviations)
    - [POST /abbreviations/batch](#post-apiv1abbreviationsbatch)
    - [GET /abbreviations/lookup/{abbreviation}](#get-apiv1abbreviationslookupabbreviation)
    - [GET /abbreviations/dictionary/stats](#get-apiv1abbreviationsdictionarystats)
    - [GET /abbreviations/domains](#get-apiv1abbreviationsdomains)
13. [Assertion Detection](#assertion-detection)
    - [POST /assertions](#post-apiv1assertions)
    - [POST /assertions/batch](#post-apiv1assertionsbatch)
    - [GET /assertions/statuses](#get-apiv1assertionsstatuses)
    - [GET /assertions/stats](#get-apiv1assertionsstats)
14. [Concept Normalization](#concept-normalization)
    - [POST /normalize](#post-apiv1normalize)
    - [POST /normalize/batch](#post-apiv1normalizebatch)
    - [GET /normalize/lookup/{cui}](#get-apiv1normalizelookupcui)
    - [GET /normalize/dictionary/stats](#get-apiv1normalizedictionarystats)
15. [Relation Extraction](#relation-extraction)
    - [POST /relations](#post-apiv1relations)
    - [GET /relations/types](#get-apiv1relationstypes)
16. [Temporal Extraction](#temporal-extraction)
    - [POST /temporal](#post-apiv1temporal)
    - [GET /temporal/frequency-map](#get-apiv1temporalfrequency-map)
17. [PHI De-identification](#phi-de-identification)
    - [POST /deidentify](#post-apiv1deidentify)
    - [POST /deidentify/batch](#post-apiv1deidentifybatch)
18. [Clinical Note Quality](#clinical-note-quality)
    - [POST /quality](#post-apiv1quality)
    - [POST /quality/batch](#post-apiv1qualitybatch)
    - [GET /quality/dimensions](#get-apiv1qualitydimensions)
19. [Social Determinants of Health](#social-determinants-of-health)
    - [POST /sdoh](#post-apiv1sdoh)
    - [POST /sdoh/batch](#post-apiv1sdohbatch)
    - [GET /sdoh/domains](#get-apiv1sdohdomains)
    - [GET /sdoh/domains/{name}](#get-apiv1sdohdomainsname)
    - [GET /sdoh/z-codes](#get-apiv1sdohz-codes)
20. [Comorbidity Scoring](#comorbidity-scoring)
    - [POST /comorbidity](#post-apiv1comorbidity)
    - [POST /comorbidity/batch](#post-apiv1comorbiditybatch)
    - [GET /comorbidity/categories](#get-apiv1comorbiditycategories)
    - [GET /comorbidity/categories/{name}](#get-apiv1comorbiditycategoriesname)
21. [Conversation Memory](#conversation-memory)
    - [POST /conversation/turns](#post-apiv1conversationturns)
    - [POST /conversation/context](#post-apiv1conversationcontext)
    - [DELETE /conversation/{session_id}](#delete-apiv1conversationsessionid)
    - [GET /conversation/stats](#get-apiv1conversationstats)
    - [GET /conversation/sessions](#get-apiv1conversationsessions)
22. [Document Search](#document-search)
    - [POST /search](#post-apiv1search)
    - [POST /search/reindex](#post-apiv1searchreindex)
23. [Batch Processing](#batch-processing)
    - [POST /batch](#post-apiv1batch)
    - [GET /batch/{job_id}](#get-apiv1batchjobid)
24. [Model Registry](#model-registry)
    - [GET /models](#get-apiv1models)
    - [GET /models/{model_name}](#get-apiv1modelsmodelname)
25. [Infrastructure](#infrastructure)
    - [GET /health](#get-apiv1health)
    - [GET /health/live](#get-apiv1healthlive)
    - [GET /health/ready](#get-apiv1healthready)
    - [GET /metrics](#get-apiv1metrics)
    - [GET /metrics/models](#get-apiv1metricsmodels)
    - [GET /drift/status](#get-apiv1driftstatus)
    - [POST /drift/record](#post-apiv1driftrecord)
26. [Error Codes](#error-codes)
26. [Rate Limiting](#rate-limiting)

---

## Authentication

ClinIQ supports two authentication methods. All endpoints except `/health`, `/docs`, and `/metrics` require authentication.

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

### Auth Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/token` | POST | Login with username/password, returns JWT |
| `/auth/register` | POST | Register a new user account |
| `/auth/api-keys` | POST | Generate an API key (requires JWT) |
| `/auth/me` | GET | Get current authenticated user profile |

---

## Core Analysis Endpoints

### POST /api/v1/analyze

Run the full clinical NLP pipeline (NER + ICD-10 + Summarization + Risk Scoring). Each stage can be individually enabled or disabled.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT: Chest pain.\n\nHPI: 72-year-old male with known coronary artery disease...",
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
| `text` | string | Yes | Clinical text (1–100,000 characters) |
| `config.ner.enabled` | bool | No | Run NER stage (default: true) |
| `config.ner.model` | string | No | `rule-based`, `spacy`, or `transformer` |
| `config.ner.min_confidence` | float | No | Confidence threshold 0.0–1.0 |
| `config.icd.enabled` | bool | No | Run ICD prediction (default: true) |
| `config.icd.model` | string | No | `sklearn-baseline`, `transformer`, or `hierarchical` |
| `config.icd.top_k` | int | No | Max codes to return (1–50, default: 10) |
| `config.summary.enabled` | bool | No | Run summarization (default: true) |
| `config.summary.model` | string | No | `extractive` or `abstractive` |
| `config.summary.detail_level` | string | No | `brief`, `standard`, or `detailed` |
| `config.risk.enabled` | bool | No | Run risk scoring (default: true) |
| `document_id` | string | No | Client-supplied document ID |
| `store_result` | bool | No | Persist result to database (default: false) |

**Response (200):**

```json
{
  "document_id": "pat-20260324-001",
  "text_length": 485,
  "entities": [
    {
      "text": "chest pain",
      "entity_type": "SYMPTOM",
      "start_char": 20,
      "end_char": 30,
      "confidence": 0.92,
      "is_negated": false,
      "is_uncertain": false
    }
  ],
  "icd_codes": [
    {
      "code": "I21.9",
      "description": "Acute myocardial infarction, unspecified",
      "confidence": 0.88,
      "chapter": "Diseases of the circulatory system"
    }
  ],
  "summary": {
    "summary": "72-year-old male with CAD presents with STEMI.",
    "key_points": ["STEMI", "Coronary artery disease"],
    "compression_ratio": 6.07,
    "processing_time_ms": 19.1
  },
  "risk_score": {
    "score": 0.78,
    "category": "high",
    "top_factors": [...],
    "recommendations": [...]
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

### POST /api/v1/analyze/stream

Server-Sent Events endpoint for real-time stage-by-stage pipeline progress. Each stage emits its own JSON event on completion.

**Request:** Same schema as `POST /analyze`.

**Response:** SSE stream with `Content-Type: text/event-stream`.

```
event: started
data: {"document_id": "pat-001", "text_length": 485}

event: ner
data: {"entities": [...], "entity_count": 12, "processing_time_ms": 11.2}

event: icd
data: {"predictions": [...], "processing_time_ms": 39.5}

event: summary
data: {"summary": "...", "key_points": [...], "processing_time_ms": 19.1}

event: risk
data: {"score": 0.78, "category": "high", "processing_time_ms": 8.3}

event: complete
data: {"total_ms": 78.1}
```

Stage failures emit a `stage_error` event without aborting remaining stages.

---

### POST /api/v1/analyze/enhanced

Run the enhanced pipeline integrating all 14 clinical NLP modules in one request. Superset of `/analyze` — includes NER, ICD, summary, and risk plus classification, sections, quality, medications, allergies, vitals, temporal, assertions, normalization, SDoH, comorbidity, abbreviations, de-identification, and relation extraction.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT: Chest pain...",
  "document_id": "enhanced-001",
  "enable_classification": true,
  "enable_sections": true,
  "enable_quality": true,
  "enable_deidentification": false,
  "enable_abbreviations": true,
  "enable_medications": true,
  "enable_allergies": true,
  "enable_vitals": true,
  "enable_temporal": true,
  "enable_assertions": true,
  "enable_normalization": true,
  "enable_sdoh": true,
  "enable_comorbidity": true,
  "enable_relations": true
}
```

All `enable_*` fields default to `true` except `enable_deidentification` (defaults `false`). Text limit: 500,000 characters.

**Response (200):** Returns all 14 module outputs plus `component_errors` for any modules that failed gracefully.

### POST /api/v1/analyze/enhanced/batch

Batch enhanced analysis for up to 20 documents with per-document config overrides.

### GET /api/v1/analyze/enhanced/modules

Returns a catalogue of all 14 enhanced pipeline modules with descriptions and default states.

---

## Named Entity Recognition

### POST /api/v1/ner

Extract medical named entities from clinical text. Detects diseases, symptoms, medications, dosages, procedures, lab values, anatomy, temporal expressions, and more.

**Request:**

```json
{
  "text": "Patient takes metformin 1000mg BID and lisinopril 10mg daily.",
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
      "is_negated": false,
      "is_uncertain": false
    }
  ],
  "model_name": "rule-based",
  "entity_count": 6,
  "processing_time_ms": 5.3
}
```

---

## ICD-10 Code Prediction

### POST /api/v1/icd-predict

Predict ICD-10 diagnosis codes from clinical text.

**Request:**

```json
{
  "text": "Patient diagnosed with type 2 diabetes mellitus with diabetic CKD stage 3.",
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
      "chapter": "Endocrine, nutritional and metabolic diseases"
    }
  ],
  "model_name": "sklearn-baseline",
  "processing_time_ms": 34.7
}
```

### GET /api/v1/icd-codes/{code}

Look up ICD-10 code details by code string (e.g., `E11.9`). Returns 404 if not found.

---

## Clinical Summarization

### POST /api/v1/summarize

Generate a clinical summary using extractive (TextRank) or abstractive (BART/T5) methods.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT: Shortness of breath...",
  "model": "extractive",
  "detail_level": "standard"
}
```

| detail_level | Sentence Retention | Max Sentences |
|---|---|---|
| `brief` | 15% of original | 5 |
| `standard` | 30% of original | 12 |
| `detailed` | 50% of original | 25 |

**Response (200):**

```json
{
  "summary": "68-year-old female presents with progressive dyspnea over 3 weeks...",
  "key_points": ["Progressive dyspnea", "Known CHF and COPD"],
  "original_word_count": 342,
  "summary_word_count": 31,
  "compression_ratio": 11.03,
  "processing_time_ms": 22.8
}
```

---

## Risk Scoring

### POST /api/v1/risk-score

Assess clinical risk across medication, diagnostic, and follow-up dimensions.

**Request:**

```json
{
  "text": "Patient on warfarin and aspirin with history of GI bleeding.",
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
      "category": "medication_risk",
      "description": "Potential drug interaction: aspirin + warfarin"
    }
  ],
  "recommendations": [
    "HIGH PRIORITY: Urgent follow-up within 24-48 hours recommended"
  ],
  "category_scores": {
    "medication_risk": 81.3,
    "diagnostic_complexity": 55.0,
    "follow_up_urgency": 52.5
  }
}
```

Risk levels: `low` (0–34), `moderate` (35–59), `high` (60–79), `critical` (80–100).

---

## Document Classification

### POST /api/v1/classify

Classify a clinical document into one of 13 document types using weighted scoring (section headers + keyword density + structural features).

**Request:**

```json
{
  "text": "DISCHARGE SUMMARY\n\nDate of Admission: 03/20/2026...",
  "min_confidence": 0.05,
  "top_k": 5
}
```

**Response (200):**

```json
{
  "predicted_type": "discharge_summary",
  "confidence": 0.92,
  "scores": [
    {
      "document_type": "discharge_summary",
      "confidence": 0.92,
      "evidence": ["DISCHARGE SUMMARY header", "discharge medications keyword"]
    }
  ],
  "processing_time_ms": 0.8
}
```

### POST /api/v1/classify/batch

Batch classification for up to 50 documents.

### GET /api/v1/classify/types

Catalogue of all 13 classifiable document types with descriptions: `discharge_summary`, `progress_note`, `history_physical`, `operative_note`, `consultation_note`, `radiology_report`, `pathology_report`, `laboratory_report`, `nursing_note`, `emergency_note`, `dental_note`, `prescription`, `referral`.

---

## Medication Extraction

### POST /api/v1/medications

Extract structured medication data including drug name, dosage, route, frequency, duration, indication, PRN status, and active/discontinued status.

**Request:**

```json
{
  "text": "Metformin 1000mg PO BID for diabetes. Lisinopril 10mg daily. Aspirin 81mg PRN.",
  "min_confidence": 0.3,
  "include_generics": true
}
```

**Response (200):**

```json
{
  "medications": [
    {
      "drug_name": "metformin",
      "generic_name": "metformin",
      "brand_names": ["Glucophage"],
      "dosage": { "value": 1000.0, "unit": "mg" },
      "route": "PO",
      "frequency": "BID",
      "indication": "diabetes",
      "is_prn": false,
      "status": "active",
      "confidence": 0.85,
      "start": 0,
      "end": 36
    }
  ],
  "medication_count": 3,
  "processing_time_ms": 2.1
}
```

Drug dictionary covers 220+ medications across cardiology, endocrine, psychiatry, pain, pulmonary, GI, antibiotics, and dental domains with brand→generic normalization.

### POST /api/v1/medications/batch

Batch extraction for up to 50 documents.

### GET /api/v1/medications/lookup/{drug_name}

Dictionary lookup for a drug by name (brand or generic). Returns generic name, brand variants, and therapeutic category.

### GET /api/v1/medications/dictionary/stats

Coverage statistics for the drug dictionary by therapeutic domain.

---

## Allergy Extraction

### POST /api/v1/allergies

Extract drug, food, and environmental allergies with reaction detection, severity classification (life-threatening/severe/moderate/mild), and NKDA status.

**Request:**

```json
{
  "text": "ALLERGIES: Penicillin (anaphylaxis), sulfa drugs (rash). NKFA for food.",
  "min_confidence": 0.50
}
```

**Response (200):**

```json
{
  "allergies": [
    {
      "allergen": "penicillin",
      "category": "drug",
      "reactions": ["anaphylaxis"],
      "severity": "life_threatening",
      "confidence": 0.95,
      "assertion_status": "present"
    }
  ],
  "no_known_allergies": false,
  "allergy_count": 2,
  "processing_time_ms": 1.8
}
```

Covers ~150 allergen entries (80+ drug, 15 food, 10 environmental) with ~250 surface forms including brand names.

### POST /api/v1/allergies/batch

Batch extraction for up to 50 documents with aggregate category breakdown.

### GET /api/v1/allergies/dictionary/stats

Allergen coverage statistics by category.

### GET /api/v1/allergies/categories

Catalogue of 3 allergy categories (drug, food, environmental) with counts.

---

## Vital Signs Extraction

### POST /api/v1/vitals

Extract structured vital sign measurements from clinical text. Detects 9 vital types with physiological range validation and clinical interpretation.

**Request:**

```json
{
  "text": "VS: BP 142/88, HR 76, T 98.6F, RR 16, SpO2 97% on RA. Wt 82kg.",
  "min_confidence": 0.5
}
```

**Response (200):**

```json
{
  "vitals": [
    {
      "vital_type": "blood_pressure",
      "value": 142.0,
      "unit": "mmHg",
      "secondary_value": 88.0,
      "interpretation": "high",
      "confidence": 0.90,
      "raw_text": "BP 142/88"
    },
    {
      "vital_type": "heart_rate",
      "value": 76.0,
      "unit": "bpm",
      "interpretation": "normal",
      "confidence": 0.85
    }
  ],
  "vital_count": 6,
  "critical_findings": [],
  "processing_time_ms": 1.2
}
```

Interpretations: `normal`, `low`, `high`, `critical_low`, `critical_high` per AHA 2017 BP guidelines.

### POST /api/v1/vitals/batch

Batch extraction for up to 50 documents.

### GET /api/v1/vitals/types

Catalogue of 9 vital sign types: blood pressure, heart rate, temperature, respiratory rate, oxygen saturation, weight, height, BMI, pain scale.

### GET /api/v1/vitals/ranges

Adult reference ranges for all vital types including diastolic BP.

---

## Clinical Section Parsing

### POST /api/v1/sections

Parse a clinical document into constituent sections with character-offset boundaries.

**Request:**

```json
{
  "text": "CHIEF COMPLAINT:\nChest pain.\n\nHPI:\n72-year-old male...",
  "min_confidence": 0.0
}
```

**Response (200):**

```json
{
  "sections": [
    {
      "header": "CHIEF COMPLAINT",
      "category": "chief_complaint",
      "header_start": 0,
      "header_end": 16,
      "body_end": 28,
      "body": "Chest pain.",
      "confidence": 1.0
    }
  ],
  "section_count": 2,
  "categories_found": ["chief_complaint", "history_of_present_illness"]
}
```

Detects 35 section categories covering H&P notes, discharge, dental, SOAP, and meta sections.

### POST /api/v1/sections/batch

Batch parsing for up to 100 documents with aggregate statistics.

### POST /api/v1/sections/query

Position-in-section query: given a character offset, returns which section contains it.

```json
{ "text": "...", "position": 42 }
```

### GET /api/v1/sections/categories

Catalogue of all 35 section categories with descriptions.

---

## Abbreviation Expansion

### POST /api/v1/abbreviations

Detect and expand medical abbreviations with context-aware disambiguation for ambiguous terms.

**Request:**

```json
{
  "text": "Pt c/o SOB and CP. PMH: HTN, DM, COPD. Meds: ASA, metoprolol BID.",
  "min_confidence": 0.60,
  "expand_in_place": true,
  "domains": null
}
```

**Response (200):**

```json
{
  "abbreviations": [
    {
      "abbreviation": "SOB",
      "expansion": "shortness of breath",
      "domain": "pulmonology",
      "confidence": 0.85,
      "is_ambiguous": false,
      "start": 10,
      "end": 13
    }
  ],
  "expanded_text": "Pt c/o shortness of breath (SOB) and chest pain (CP)...",
  "abbreviation_count": 7,
  "processing_time_ms": 3.1
}
```

220+ unambiguous entries across 12 clinical domains. 10 ambiguous abbreviations with context-aware disambiguation (section headers, keyword proximity).

### POST /api/v1/abbreviations/batch

Batch expansion for up to 50 documents.

### GET /api/v1/abbreviations/lookup/{abbreviation}

Dictionary lookup for a specific abbreviation. Returns all possible expansions (with senses for ambiguous terms).

### GET /api/v1/abbreviations/dictionary/stats

Coverage statistics by clinical domain.

### GET /api/v1/abbreviations/domains

Catalogue of 12 clinical domains with descriptions: cardiology, pulmonology, endocrine, neurology, gastroenterology, renal, infectious, musculoskeletal, hematology, general, dental, pharmacy.

---

## Assertion Detection

### POST /api/v1/assertions

Classify an entity's assertion status using ConText/NegEx-inspired algorithm. Determines whether an entity is present, absent, possible, conditional, hypothetical, or family.

**Request:**

```json
{
  "text": "Patient denies chest pain. Family history of diabetes.",
  "entity_start": 15,
  "entity_end": 25
}
```

**Response (200):**

```json
{
  "assertion": "absent",
  "confidence": 0.92,
  "trigger": "denies",
  "trigger_type": "negation"
}
```

97 compiled regex triggers across negation (24), uncertainty (20), family (8), hypothetical (12), and conditional (6) patterns. Includes pseudo-trigger blocking and scope termination.

### POST /api/v1/assertions/batch

Batch assertion detection for up to 200 entity spans per request with summary counts.

### GET /api/v1/assertions/statuses

Catalogue of 6 assertion status types with descriptions.

### GET /api/v1/assertions/stats

Detection statistics and trigger counts.

---

## Concept Normalization

### POST /api/v1/normalize

Link extracted entity text to standardized medical ontology codes (UMLS CUI, SNOMED-CT, RxNorm, ICD-10-CM, LOINC).

**Request:**

```json
{
  "text": "HTN",
  "entity_type": "DISEASE",
  "min_similarity": 0.80,
  "enable_fuzzy": true
}
```

**Response (200):**

```json
{
  "match": {
    "cui": "C0020538",
    "preferred_term": "Hypertension",
    "match_type": "alias",
    "confidence": 0.95,
    "codes": {
      "snomed_ct": "38341003",
      "icd_10_cm": "I10"
    }
  }
}
```

Three-strategy resolution: exact match (1.0), alias match (0.95), fuzzy match (≥0.80 threshold). ~140 medical concepts across 5 type groups with type-aware filtering.

### POST /api/v1/normalize/batch

Batch normalization for up to 500 entities with aggregate match statistics.

### GET /api/v1/normalize/lookup/{cui}

Reverse CUI lookup: given a UMLS CUI, returns the concept with aliases and codes.

### GET /api/v1/normalize/dictionary/stats

Coverage statistics by type group (condition, medication, procedure, anatomy, lab) and ontology.

---

## Relation Extraction

### POST /api/v1/relations

Extract semantic relations between entity pairs. Supports 12 relation types: treats, causes, diagnoses, contraindicates, administered_for, dosage_of, location_of, result_of, worsens, prevents, monitors, side_effect_of.

**Request:**

```json
{
  "text": "Metformin treats type 2 diabetes.",
  "entities": [
    { "text": "Metformin", "entity_type": "MEDICATION", "start_char": 0, "end_char": 9 },
    { "text": "type 2 diabetes", "entity_type": "DISEASE", "start_char": 17, "end_char": 32 }
  ],
  "max_distance": 150,
  "min_confidence": 0.3,
  "relation_types": null
}
```

**Response (200):**

```json
{
  "relations": [
    {
      "subject": "Metformin",
      "object": "type 2 diabetes",
      "relation_type": "treats",
      "confidence": 0.87,
      "evidence": "treats"
    }
  ],
  "relation_count": 1,
  "processing_time_ms": 1.5
}
```

Entity-type constraints prevent nonsensical pairings. Proximity and co-sentence bonuses for confidence scoring.

### GET /api/v1/relations/types

Catalogue of all 12 relation types with descriptions and entity-type constraints.

---

## Temporal Extraction

### POST /api/v1/temporal

Extract dates, durations, frequencies, relative time references, and temporal relations from clinical text.

**Request:**

```json
{
  "text": "Admitted 03/20/2026. Started metformin 3 days ago. Follow-up in 2 weeks. Takes aspirin BID.",
  "reference_date": "2026-03-25"
}
```

**Response (200):**

```json
{
  "expressions": [
    { "type": "date", "text": "03/20/2026", "resolved_date": "2026-03-20", "confidence": 0.90 },
    { "type": "relative", "text": "3 days ago", "resolved_date": "2026-03-22", "confidence": 0.85 }
  ],
  "frequencies": [
    { "text": "BID", "normalized": "twice daily", "times_per_day": 2.0 }
  ],
  "temporal_links": [
    { "type": "before", "source": "metformin", "target": "admission" }
  ],
  "expression_count": 3,
  "frequency_count": 1,
  "link_count": 1
}
```

Supports 4 date formats, durations, relative times, ages, postoperative days, and 40+ frequency abbreviations.

### GET /api/v1/temporal/frequency-map

Catalogue of 40+ clinical frequency abbreviations (QD, BID, TID, QID, q2h–q72h, PRN, STAT, etc.) with normalized descriptions.

---

## PHI De-identification

### POST /api/v1/deidentify

Detect and redact Protected Health Information (PHI) per HIPAA Safe Harbor rules. Covers all 18 identifier categories.

**Request:**

```json
{
  "text": "Dr. John Smith saw patient Jane Doe (SSN: 123-45-6789) on 03/25/2026.",
  "strategy": "redact",
  "phi_types": null,
  "confidence_threshold": 0.5
}
```

| strategy | Description | Example |
|----------|-------------|---------|
| `redact` | Replace with `[TYPE]` tags | `[NAME] saw patient [NAME]` |
| `mask` | Replace with asterisks | `** **** ***** saw patient **** ***` |
| `surrogate` | Deterministic synthetic values | `Dr. Robert Chen saw patient Lisa Park` |

**Response (200):**

```json
{
  "deidentified_text": "Dr. [NAME] saw patient [NAME] (SSN: [SSN]) on [DATE].",
  "phi_entities": [
    { "text": "John Smith", "phi_type": "NAME", "start": 4, "end": 14, "confidence": 0.95 }
  ],
  "phi_count": 4,
  "strategy": "redact"
}
```

### POST /api/v1/deidentify/batch

Batch de-identification for up to 50 documents.

---

## Clinical Note Quality

### POST /api/v1/quality

Analyze clinical note quality across 5 dimensions before NLP processing.

**Request:**

```json
{
  "text": "CC: chest pain. HPI: 72yo M with CP...",
  "expected_sections": ["Chief Complaint", "HPI", "Assessment", "Plan"]
}
```

**Response (200):**

```json
{
  "overall_score": 72.5,
  "grade": "C",
  "dimensions": [
    { "dimension": "completeness", "score": 65.0, "weight": 0.25, "findings": [...] },
    { "dimension": "readability", "score": 80.0, "weight": 0.20, "findings": [...] },
    { "dimension": "structure", "score": 75.0, "weight": 0.20, "findings": [...] },
    { "dimension": "information_density", "score": 70.0, "weight": 0.20, "findings": [...] },
    { "dimension": "consistency", "score": 85.0, "weight": 0.15, "findings": [...] }
  ],
  "recommendations": ["Add Assessment section", "Reduce abbreviation density"],
  "text_hash": "a1b2c3..."
}
```

Grades: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60. Finding severities: critical, warning, info.

### POST /api/v1/quality/batch

Batch quality analysis for up to 100 notes with aggregate summary (min/max/avg scores, grade distribution).

### GET /api/v1/quality/dimensions

Catalogue of 5 quality dimensions with descriptions and default weights.

---

## Social Determinants of Health

### POST /api/v1/sdoh

Extract social and behavioral risk factors from clinical text across 8 domains aligned with Healthy People 2030.

**Request:**

```json
{
  "text": "Patient is homeless, unemployed, food insecure. History of tobacco use.",
  "min_confidence": 0.50
}
```

**Response (200):**

```json
{
  "extractions": [
    {
      "domain": "housing",
      "text": "homeless",
      "sentiment": "adverse",
      "confidence": 0.90,
      "z_codes": ["Z59.01"]
    },
    {
      "domain": "substance_use",
      "text": "tobacco use",
      "sentiment": "adverse",
      "confidence": 0.85,
      "z_codes": ["Z72.0"]
    }
  ],
  "extraction_count": 4,
  "adverse_count": 4,
  "protective_count": 0
}
```

8 domains: housing, employment, education, food security, transportation, social support, substance use, financial. ICD-10-CM Z-code mapping (Z55–Z65). Negation-aware sentiment flipping.

### POST /api/v1/sdoh/batch

Batch extraction for up to 50 documents with aggregate adverse/protective counts.

### GET /api/v1/sdoh/domains

Catalogue of all 8 SDoH domains with trigger counts and Z-codes.

### GET /api/v1/sdoh/domains/{name}

Domain detail with adverse/protective trigger breakdown.

### GET /api/v1/sdoh/z-codes

Flat Z-code catalogue across all domains.

---

## Comorbidity Scoring

### POST /api/v1/comorbidity

Calculate the Charlson Comorbidity Index (CCI) from ICD-10-CM codes and/or free text. Implements Charlson–Deyo adaptation with 17 disease categories.

**Request:**

```json
{
  "icd_codes": ["E11.9", "I21.0", "N18.3"],
  "text": "History of diabetes, prior MI, and stage 3 CKD.",
  "patient_age": 72,
  "age_adjust": true,
  "hierarchical_exclusion": true
}
```

**Response (200):**

```json
{
  "score": 6,
  "age_adjusted_score": 8,
  "risk_group": "severe",
  "mortality_10yr": 0.42,
  "matched_categories": [
    { "category": "diabetes_uncomplicated", "weight": 1, "source": "icd", "code": "E11.9" },
    { "category": "myocardial_infarction", "weight": 1, "source": "icd", "code": "I21.0" },
    { "category": "renal_disease", "weight": 2, "source": "icd", "code": "N18.3" }
  ],
  "age_points": 2
}
```

Risk groups: `low` (0), `mild` (1–2), `moderate` (3–4), `severe` (≥5). 10-year mortality via Charlson exponential survival formula.

### POST /api/v1/comorbidity/batch

Batch CCI calculation for up to 50 patients with aggregate statistics.

### GET /api/v1/comorbidity/categories

Catalogue of 17 CCI categories with weights and descriptions.

### GET /api/v1/comorbidity/categories/{name}

Category detail with ICD-10-CM code prefix list.

---

## Conversation Memory

Session-scoped conversation memory for context-aware clinical analysis. Records analysis turns and provides aggregated context from previous analyses within a session.

### POST /api/v1/conversation/turns

Record a completed analysis result in a session's conversation history.

```bash
curl -X POST http://localhost:8000/api/v1/conversation/turns \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123-session",
    "text": "Patient presents with chest pain and shortness of breath. History of hypertension.",
    "entities": [
      {"text": "chest pain", "entity_type": "SYMPTOM", "confidence": 0.95},
      {"text": "shortness of breath", "entity_type": "SYMPTOM", "confidence": 0.88},
      {"text": "hypertension", "entity_type": "DISEASE", "confidence": 0.92}
    ],
    "icd_codes": [
      {"code": "R07.9", "description": "Chest pain, unspecified", "confidence": 0.82}
    ],
    "risk_score": 0.65,
    "risk_level": "moderate",
    "summary": "Patient with acute chest pain and dyspnea, history of HTN.",
    "document_id": "doc-001",
    "metadata": {"source": "ER", "provider": "Dr. Smith"}
  }'
```

**Response:**

```json
{
  "session_id": "user-123-session",
  "turn_id": 1,
  "turn_count": 1
}
```

### POST /api/v1/conversation/context

Retrieve aggregated context from a session's recent conversation history. Returns deduplicated entities, ICD codes, and risk trends across all recorded turns.

```bash
curl -X POST http://localhost:8000/api/v1/conversation/context \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123-session",
    "last_n": 5
  }'
```

**Response:**

```json
{
  "session_id": "user-123-session",
  "turn_count": 2,
  "turns": [
    {
      "turn": 1,
      "timestamp": 1711425600.0,
      "text_length": 81,
      "entities_by_type": {"SYMPTOM": ["chest pain", "shortness of breath"], "DISEASE": ["hypertension"]},
      "entity_count": 3,
      "icd_codes": [{"code": "R07.9", "description": "Chest pain, unspecified"}],
      "risk": {"score": 0.65, "level": "moderate"},
      "summary": "Patient with acute chest pain and dyspnea, history of HTN."
    }
  ],
  "unique_entities": ["chest pain", "hypertension", "shortness of breath"],
  "unique_icd_codes": ["R07.9"],
  "overall_risk_trend": [0.65]
}
```

### DELETE /api/v1/conversation/{session_id}

Clear all conversation history for a session. Returns 404 if the session does not exist.

```bash
curl -X DELETE http://localhost:8000/api/v1/conversation/user-123-session
```

**Response:**

```json
{
  "session_id": "user-123-session",
  "status": "cleared"
}
```

### GET /api/v1/conversation/stats

Return conversation memory usage statistics.

```bash
curl http://localhost:8000/api/v1/conversation/stats
```

**Response:**

```json
{
  "active_sessions": 42,
  "total_turns": 187,
  "max_sessions": 5000,
  "max_turns_per_session": 50,
  "session_ttl_seconds": 7200.0
}
```

### GET /api/v1/conversation/sessions

List all active conversation sessions with turn counts and last-access timestamps.

```bash
curl http://localhost:8000/api/v1/conversation/sessions
```

**Response:**

```json
{
  "sessions": [
    {
      "session_id": "user-123-session",
      "turn_count": 3,
      "last_access": 1711425900.0,
      "oldest_turn_id": 1,
      "newest_turn_id": 3
    }
  ],
  "total": 1
}
```

---

## Document Search

### POST /api/v1/search

Hybrid BM25 + TF-IDF search over ingested clinical documents with optional medical query expansion and re-ranking.

**Request:**

```json
{
  "query": "chest pain with elevated troponin",
  "top_k": 10,
  "min_score": 0.01,
  "alpha": 0.5,
  "expand_query": true,
  "rerank": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | — | Natural-language search query |
| `top_k` | int | 10 | Max results (1–100) |
| `min_score` | float | 0.01 | Minimum hybrid score threshold |
| `alpha` | float | 0.5 | BM25 vs TF-IDF weight (0 = pure TF-IDF, 1 = pure BM25) |
| `expand_query` | bool | true | Apply medical query expansion (synonyms, abbreviations) |
| `rerank` | bool | true | Apply clinical re-ranking on initial results |

**Response (200):**

```json
{
  "results": [
    {
      "document_id": "doc-123",
      "score": 0.87,
      "snippet": "...elevated troponin levels consistent with acute MI...",
      "content_hash": "abc123"
    }
  ],
  "total": 1,
  "query_expansion": {
    "original": "chest pain with elevated troponin",
    "expanded_terms": ["angina", "thoracic pain", "troponin I", "troponin T"]
  },
  "reranked": true
}
```

### POST /api/v1/search/reindex

Rebuild the in-memory search index from the document database. Use after bulk ingestion.

---

## Batch Processing

### POST /api/v1/batch

Submit multiple documents for asynchronous processing via Celery + Redis.

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
  "created_at": "2026-03-25T12:00:00Z"
}
```

### GET /api/v1/batch/{job_id}

Poll batch job status and retrieve results when complete.

**Response (200):**

```json
{
  "job_id": "batch-a1b2c3d4",
  "status": "completed",
  "document_count": 2,
  "completed_count": 2,
  "failed_count": 0,
  "results": [...],
  "completed_at": "2026-03-25T12:00:05Z"
}
```

Status values: `queued`, `processing`, `completed`, `failed`, `partial`.

---

## Model Registry

### GET /api/v1/models

List all available ML models with their task, version, and status.

### GET /api/v1/models/{model_name}

Get detailed information for a specific model including metrics, training timestamps, and deployment status.

---

## Infrastructure

### GET /api/v1/health

Overall health check. Exempt from authentication and rate limiting.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "database": "connected",
    "redis": "connected",
    "models_loaded": 3
  },
  "uptime_seconds": 3600
}
```

### GET /api/v1/health/live

Kubernetes liveness probe. Returns 200 if the process is alive.

### GET /api/v1/health/ready

Kubernetes readiness probe. Returns 200 if all dependencies are connected and models are loaded.

### GET /api/v1/metrics

Prometheus text exposition format metrics (inference latencies, error counts, batch sizes). Falls back to JSON if `prometheus_client` is not installed.

### GET /api/v1/metrics/models

Per-model inference summary for dashboards.

### GET /api/v1/drift/status

Aggregated data-drift and prediction-drift status for all monitored models. Returns `stable`, `warning`, or `drifted` overall status with PSI scores and confidence drift indicators.

### POST /api/v1/drift/record

Record a prediction for drift monitoring. Used for back-filling and real-time ingestion.

---

## Error Codes

All errors return a JSON body:

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

**Exempt endpoints:** `/api/v1/health`, `/docs`, `/openapi.json`, `/metrics`

When rate limited, the API returns HTTP 429 with:

```json
{
  "detail": "Rate limit exceeded"
}
```
