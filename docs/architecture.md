# ClinIQ Architecture

This document describes the system architecture of the ClinIQ clinical NLP platform, including component descriptions, data flows, key design decisions, and operational strategies.

---

## System Architecture Overview

```
                                 Internet
                                    |
                            +-------+-------+
                            |   Nginx /     |
                            |   Ingress     |
                            | (TLS, CORS,   |
                            |  Rate Limit)  |
                            +-------+-------+
                                    |
                   +----------------+----------------+
                   |                                 |
            +------+------+                  +-------+------+
            | FastAPI API |                  | React SPA    |
            | (Port 8000) |                  | (Port 5173)  |
            +------+------+                  +--------------+
                   |
        +----------+----------+
        |          |          |
   +----+----+  +-+-----+ +--+-------+
   | ML      |  | Celery | | Auth    |
   | Pipeline |  | Worker | | Middleware|
   +----+----+  +--+-----+ +--+------+
        |           |          |
   +----+----+ +---+---+ +----+----+
   | Models  | | Redis | | Postgres|
   | (Files) | | Cache | | (Data)  |
   +---------+ | Queue | +---------+
               +---+---+
                   |
           +-------+-------+
           |               |
      +----+----+    +-----+----+
      |  MinIO  |    |  MLflow  |
      | (Files) |    | (Track)  |
      +---------+    +----------+

      Observability: Prometheus -> Grafana
```

---

## Component Descriptions

### API Layer

**FastAPI Application** (`backend/app/main.py`)
- Serves all REST API endpoints under `/api/v1`
- Manages application lifecycle (database init, model loading, shutdown)
- CORS middleware with configurable origin allowlist
- Request timing via `X-Process-Time` header
- Custom exception handling with structured error responses

**Route Modules** (`backend/app/api/v1/routes/`)
- `analyze.py` -- Full pipeline orchestrator (NER + ICD + Summary + Risk)
- `ner.py` -- Standalone named entity recognition
- `icd.py` -- ICD-10 code prediction and code lookup
- `summarize.py` -- Clinical text summarization
- `risk.py` -- Risk scoring
- `batch.py` -- Async batch processing (submit + poll)
- `models.py` -- Model registry and metadata
- `health.py` -- Liveness, readiness, and deep health checks
- `auth.py` -- JWT token issuance, user registration, API key management

### ML Pipeline

**NER Models** (`backend/app/ml/ner/model.py`)
- `RuleBasedNERModel` -- Regex pattern matching with compiled patterns
- `SpacyNERModel` -- scispaCy (`en_ner_bc5cdr_md`) integration
- `TransformerNERModel` -- Fine-tuned BioBERT with BIO tagging
- `CompositeNERModel` -- Ensemble with union/intersection/majority voting
- All models support negation detection and uncertainty flagging
- 14 entity types: DISEASE, SYMPTOM, MEDICATION, DOSAGE, PROCEDURE, ANATOMY, LAB_VALUE, TEST, TREATMENT, DEVICE, BODY_PART, DURATION, FREQUENCY, TEMPORAL

**ICD-10 Classifiers** (`backend/app/ml/icd/model.py`)
- `SklearnICDClassifier` -- TF-IDF features + scikit-learn multi-label classifier
- `TransformerICDClassifier` -- ClinicalBERT with sliding window for long documents
- `HierarchicalICDClassifier` -- Two-stage: chapter prediction then code prediction
- Supports batch prediction for throughput optimization

**Summarization** (`backend/app/ml/summarization/model.py`)
- `ExtractiveSummarizer` -- TextRank with clinical section bias (Assessment/Plan boosted) and personalized PageRank
- `AbstractiveSummarizer` -- HuggingFace BART/T5 with chunked processing for long documents
- Three detail levels: brief (15% retention), standard (30%), detailed (50%)

**Risk Scoring** (`backend/app/ml/risk/model.py`)
- `RuleBasedRiskScorer` -- Weighted analysis across three categories: medication risk (polypharmacy, drug interactions, high-risk agents), diagnostic complexity (ICD chapter weights, urgency keywords), follow-up urgency (compliance patterns, social barriers)
- `MLRiskScorer` -- Placeholder for trained classifier; extracts entity-based and ICD chapter features
- Scores 0-100 mapped to levels: low (<35), moderate (35-59), high (60-79), critical (80+)

**Dental NLP** (`backend/app/ml/dental/model.py`)
- `DentalNERModel` -- Tooth numbering (Universal/FDI/Palmer), dental procedures, periodontal measurements, surfaces, conditions
- `PeriodontalRiskAssessment` -- Risk scoring from pocket depths, bleeding, mobility
- `CDTCodePredictor` -- CDT dental procedure code prediction

**Monitoring & Explainability**
- `SHAPExplainer` (`backend/app/ml/explainability/`) -- SHAP values for model predictions
- `DriftDetector` (`backend/app/ml/monitoring/`) -- Statistical drift detection on feature distributions
- `MetricsCollector` (`backend/app/ml/monitoring/`) -- Prometheus-compatible metrics export

### Data Layer

**PostgreSQL 16** -- Primary relational store for users, API keys, audit logs, analysis results, and model metadata. Async via `asyncpg` + SQLAlchemy 2.0.

**Redis 7** -- Three logical databases:
- DB 0: Response caching (1-hour TTL by default)
- DB 1: Celery task broker
- DB 2: Celery result backend

**MinIO** -- S3-compatible object storage for uploaded documents, model artifacts, and exported reports.

**MLflow** -- Experiment tracking, model versioning, and artifact logging. Backed by PostgreSQL for metadata and file-based artifact storage.

### Infrastructure

**Nginx** -- TLS termination, static file serving, reverse proxy to API and frontend. Adds security headers (X-Frame-Options, X-Content-Type-Options, CSP).

**Prometheus + Grafana** -- Metrics collection from API, Redis exporter, PostgreSQL exporter, and node exporter. Pre-provisioned dashboards for request latency, throughput, model inference times, and system resources.

**Celery Workers** -- Async processing for batch jobs and long-running analysis tasks. Configurable concurrency (2 dev / 4 prod).

---

## Data Flow

### Single Document Analysis

```
Client Request
     |
     v
[1] API Gateway (Nginx) -- TLS termination, rate limit check
     |
     v
[2] FastAPI Middleware -- JWT/API key validation, request timing
     |
     v
[3] /api/v1/analyze route handler
     |
     +---> [4a] Text Preprocessing (clean, normalize, segment)
     |
     +---> [4b] NER Stage
     |         |-- Rule-based pattern matching
     |         |-- scispaCy entity extraction
     |         |-- Transformer token classification
     |         |-- Overlap resolution + negation detection
     |
     +---> [4c] ICD-10 Prediction Stage
     |         |-- Feature extraction (TF-IDF or tokenization)
     |         |-- Multi-label classification
     |         |-- Sliding window for long documents
     |         |-- Top-k filtering with confidence threshold
     |
     +---> [4d] Summarization Stage
     |         |-- Section detection and clinical bias scoring
     |         |-- TextRank with personalized PageRank
     |         |-- Key findings extraction
     |
     +---> [4e] Risk Scoring Stage
     |         |-- Medication risk (polypharmacy, interactions)
     |         |-- Diagnostic complexity (ICD chapters, keywords)
     |         |-- Follow-up urgency (compliance patterns)
     |         |-- Weighted aggregation + recommendations
     |
     v
[5] Response Assembly -- merge stage results, compute timing
     |
     v
[6] Audit Log -- write to PostgreSQL (async, non-blocking)
     |
     v
[7] JSON Response to Client
```

### Batch Processing

```
Client POST /batch
     |
     v
[1] Validate documents array + create batch job record
     |
     v
[2] Submit Celery task (returns job_id immediately)
     |
     v
[3] Client polls GET /batch/{job_id}
     |
     v
[4] Celery worker processes each document through the pipeline
     |
     v
[5] Results stored in Redis (result backend)
     |
     v
[6] Client receives completed results on next poll
```

---

## Key Design Decisions

### ADR-001: FastAPI over Flask

**Context:** Needed a Python web framework for a high-throughput ML inference API.

**Decision:** FastAPI with Pydantic v2 for automatic request validation, OpenAPI schema generation, and native async support.

**Rationale:**
- Native `async/await` for non-blocking I/O (database, Redis, external calls)
- Automatic request/response validation with Pydantic reduces boilerplate
- Auto-generated OpenAPI docs eliminate documentation drift
- Dependency injection system simplifies testing and configuration
- ~3x throughput improvement over Flask for I/O-bound workloads

### ADR-002: PostgreSQL + Redis Dual Data Store

**Context:** Need persistent storage for audit trails and user data, plus low-latency caching and task queuing.

**Decision:** PostgreSQL 16 for relational data; Redis 7 for caching, rate limiting, and Celery broker/backend.

**Rationale:**
- PostgreSQL provides ACID transactions required for HIPAA audit logging
- Redis sorted sets enable sliding-window rate limiting with O(log N) operations
- Redis as Celery broker eliminates the need for RabbitMQ (one fewer service)
- Separate Redis databases (0/1/2) isolate cache, broker, and result concerns

### ADR-003: MLflow for Model Tracking

**Context:** Need to version ML models, track experiments, and reproduce training runs.

**Decision:** MLflow as the experiment tracking and model registry system.

**Rationale:**
- First-party support for PyTorch, scikit-learn, and HuggingFace
- UI for comparing runs across hyperparameter sweeps
- Model registry with stage transitions (staging -> production)
- Artifact logging for training data snapshots and evaluation reports

### ADR-004: Microservices-Ready Monolith

**Context:** Small team starting a new platform, but expecting to scale.

**Decision:** Monolithic FastAPI application with clear module boundaries, designed for future decomposition into microservices.

**Rationale:**
- Single deployable unit reduces operational overhead during early development
- Module structure (`ml/ner/`, `ml/icd/`, `ml/summarization/`, `ml/risk/`) maps directly to future service boundaries
- Shared utilities (preprocessing, feature engineering) can become a shared library
- Celery workers already run as separate processes, easing the transition

---

## API Versioning Strategy

All endpoints are prefixed with `/api/v1`. When breaking changes are introduced:

1. New version routes are added under `/api/v2` alongside existing v1 routes
2. v1 routes remain operational for a minimum of 6 months after v2 release
3. Deprecation headers (`Sunset`, `Deprecation`) are added to v1 responses
4. Clients are notified via changelog and API key contact email

Non-breaking additions (new optional fields, new endpoints) are added to the current version without incrementing.

---

## Caching Strategy

Redis caching is applied at two levels:

**Response Cache** (Redis DB 0, default TTL: 1 hour)
- Cache key: SHA-256 hash of `(endpoint + request body + model version)`
- Applied to deterministic inference endpoints (NER, ICD, summarization)
- Cache-Control headers inform clients of cache status
- Cache invalidated on model version change

**Model Cache** (in-memory, LRU, `model_cache_size=3`)
- Keeps the 3 most recently used models in memory
- Lazy loading: models are loaded on first request
- Eviction frees GPU/CPU memory for new model loads

**Rate Limit Tracking** (Redis DB 0, sorted sets)
- Sliding window counter per API key or IP address
- Default: 100 requests per 24-hour window (free tier)
- Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

---

## Authentication Flow

ClinIQ supports two authentication methods:

### JWT Bearer Token (User Authentication)

```
[1] POST /api/v1/auth/token  { username, password }
         |
[2] Validate credentials (bcrypt hash comparison)
         |
[3] Issue JWT (HS256, 30-min expiry)
         |
[4] Client sends: Authorization: Bearer <token>
         |
[5] Middleware decodes + validates JWT on each request
```

### API Key (Programmatic Access)

```
[1] POST /api/v1/auth/api-keys  (authenticated via JWT)
         |
[2] Generate: cliniq_<32-byte-urlsafe-token>
         |
[3] Store bcrypt hash in PostgreSQL (plaintext returned once)
         |
[4] Client sends: X-API-Key: cliniq_<token>
         |
[5] Middleware verifies key against stored hash
```

Both methods are checked by the auth middleware. Health checks and documentation endpoints (`/docs`, `/openapi.json`) are exempt from authentication.
