# ClinIQ — 16-Month Production App Roadmap

## Building a Production Healthcare NLP Platform to Transform Your CV

*A complete engineering plan for building a professional-grade clinical NLP application from zero to production, designed specifically to demonstrate ML Engineering mastery and healthcare domain expertise.*

### Build Status (2026-03-24)

All phases are **COMPLETE**.

#### Post-PRD Enhancements — Session 14 (2026-03-24)
- [x] **Test coverage pushed from 97% to 99%** (4142 statements, 52 missed) — 80 new tests across 8 modules:
  - `test_coverage_final.py` (36 tests) — targeted edge cases: TransformerNER BIO tag extraction (continuation, special tokens, O-after-entity), CompositeNER intersection/majority/overlap, SklearnICD 1D proba reshape + decision_function sigmoid + batch predict, DentalNER quadrant mapping + surface validation, TextDistributionMonitor empty records + vocab diversity, PredictionMonitor confidence/label drift, ModelMetrics gauge methods, main.py exception handlers via TestClient, ExtractiveSummarizer InferenceError, AnalysisService error wrapping, AnalysisRequest validation, SHAP format_explanation
  - `test_dental_model_extended.py` (≈20 tests) — DentalAssessment.to_dict, PeriodontalRiskAssessor lifecycle, primary tooth extraction, procedure extraction
  - `test_batch_schema_validator.py` — batch schema validation edges
  - `test_document_service_extended.py` — document service error handling
  - `test_logging_middleware_extended.py` — logging middleware dispatch
  - `test_metrics_utils.py` — metric utility functions
  - `test_seed_extended.py` — seed data validation
  - `test_text_preprocessing_extended.py` — text preprocessing pipelines
- [x] **Remaining 52 missed lines** are untestable __main__ guards, real ML library import paths (SHAP/torch), and Redis async connection paths
- [x] **Total test suite: 1206 passing** (backend: 1206, frontend: 113), 0 failures

#### Post-PRD Enhancements — Session 13 (2026-03-24)
- [x] **Test coverage pushed from 96% to 97%** (4142 statements, 114 missed) — 47 new tests across 4 modules:
  - `test_deps_get_current_user.py` (12 tests) — JWT auth (valid/invalid/missing-sub/user-not-found/inactive), API key auth (valid/not-found/hash-mismatch/orphaned-user), no-credentials, get_current_active_user (active/inactive)
  - `test_pipeline_dental.py` (20 tests) — _run_dental (assessment populated, entities, perio risk score, CDT codes, recommendations, model metadata, without perio assessor, failure capture, dental disabled), load error handling (single/multiple failures, ensure_loaded idempotency), _components/_collect_model_versions helpers, batch edge cases
  - `test_rate_limit_redis.py` (10 tests) — Redis sliding-window _check_redis (within/over/exact limit, pipeline operations), in-memory fallback edges, _get_client_key (API key precedence, X-Forwarded-For, client host, unknown)
  - `test_configure_logging.py` (5 tests) — JSON/console format, default params, JSONRenderer/ConsoleRenderer processor selection
- [x] **Bug fix: CDT_CODES import error** — `pipeline.py` `_run_dental` attempted module-level `from app.ml.dental.model import CDT_CODES` but CDT_CODES is a class attribute; replaced with runtime attribute access and reverse-lookup
- [x] **Coverage highlights**: deps.py 53%→100%, pipeline.py 82%→100%, rate_limit.py 81%→99%
- [x] **Total test suite: 1126 passing** (backend: 1126, frontend: 113), 0 failures

#### Post-PRD Enhancements — Session 12 (2026-03-24)
- [x] **Test coverage pushed from 90% to 96%** (4139 statements, 185 missed) — 64 new tests across 5 modules:
  - `test_transformer_icd.py` (19 tests) — TransformerICDClassifier (load success/failure/no-id2label, predict with sigmoid, top_k limiting, low-confidence filtering, chapter population, error propagation, unknown label indices), sliding-window long-document handling (trigger condition, max-pooling aggregation), batch predict (count, averaged timing, error), HierarchicalICDClassifier (load delegation, two-stage chapter→code dispatch, missing chapter graceful skip, confidence sorting, batch delegation)
  - `test_shap_explainer_advanced.py` (16 tests) — TokenSHAPExplainer real SHAP path (binary shap_values, multi-class shap_values with best-class selection, vectorizer caching, no-background vectorizer, KernelExplainer caching, zero-background baseline, decision_function sigmoid fallback), AttentionExplainer (explain returns SHAPExplanation, special token exclusion, mean-layer aggregation, max-head aggregation, no-attentions graceful fallback, predicted_value from logits, sub-word accumulation, torch ImportError, top_positive_features populated)
  - `test_metrics_collector_advanced.py` (14 tests) — ModelMetrics Prometheus init (success/failure/disabled), Prometheus record paths (inference, batch, error), fallback collect (empty, after recording, batch size), Prometheus collect (success, exception→empty, namespace filtering), _InferenceTimer context manager (records inference, records error on exception)
  - `test_db_session.py` (6 tests) — get_db_session (yields session + commits, rollback on exception), get_db_context (commits on success, rollback on exception), init_db (creates tables), close_db (disposes engine)
  - `test_main_lifespan.py` (9 tests) — lifespan (app starts with lifespan, handles DB init failure), exception handlers (NOT_FOUND→404, VALIDATION_ERROR→400, AUTH→401, unknown→500, all 12 codes mapped), process-time middleware (X-Process-Time header), root endpoint (returns app info)
- [x] **Updated README** — coverage badge 89% → 96%
- [x] **Total test suite: 1079 passing** (backend: 1079, frontend: 113), 0 failures

#### Post-PRD Enhancements — Session 11 (2026-03-24)
- [x] **Test coverage pushed from 78% to 89%** (4148 statements, 439 missed) — 94 new tests across 5 modules:
  - `test_sklearn_icd_classifier.py` (26 tests) — SklearnICDClassifier (load empty/from file/failure, predict with predict_proba/decision_function, batch predict, ensure_loaded), TransformerICDClassifier (load mocks), HierarchicalICDClassifier (dispatch, no-match, batch, load), ICDCodePrediction/ICDPredictionResult dataclasses, get_chapter_for_code helper (13 ICD-10 chapter lookups + edge cases)
  - `test_spacy_ner.py` (18 tests) — SpacyNERModel (load/load-from-path/load-failure, entity extraction, type mapping, negation detection, uncertainty detection, inference error), TransformerNERModel (load/load-from-path/load-failure, BIO tag extraction, inference error), Entity dataclass to_dict
  - `test_auth_get_current_user.py` (11 tests) — JWT auth (valid token, invalid token, missing sub, user not found, inactive user), API key auth (valid key, invalid key), no credentials, get_optional_user (no auth, auth error, valid auth)
  - `test_seed.py` (10 tests) — ICD-10 seed data validation (count, structure, dental codes, common codes, no duplicates), seed_icd_codes (new + skip existing), seed_admin_user (create + skip existing), seed_all orchestrator
  - `test_abstractive_summarizer.py` (8 tests) — AbstractiveSummarizer load (success/from-path/failure), single/multi-chunk summarization, error propagation, detail level, SummarizationResult dataclass
  - `test_ml_risk_scorer.py` (14 tests) — MLRiskScorer (load with/without path, load failure, null assessment, classifier prediction, feature extraction with/without entities, ICD chapter features, inference error), RuleBasedRiskScorer recommendation generation (critical/high/moderate/low tiers, max cap)
- [x] **Removed dead code** — `app/api/v1/schemas.py` (312 lines, 0% coverage, never imported — actual schemas in `app/api/schemas/`)
- [x] **Updated README** — coverage badge 84% → 89%, Python badge 3.11 → 3.12
- [x] **Total test suite: 1015 passing** (backend: 1015, frontend: 113), 0 failures

#### Post-PRD Enhancements — Session 10 (2026-03-24)
- [x] **Frontend test infrastructure** — Set up Vitest 4.1 + React Testing Library + jsdom environment with ResizeObserver polyfill for Recharts compatibility
- [x] **113 new frontend tests** across 7 test modules (~1,200 lines):
  - `EntityTag.test.tsx` (22 tests) — entity text rendering, confidence percentage display/rounding, click handlers, cursor-pointer conditional class, sm/md size variants, colour mapping per EntityType (6 types), EntityTypeBadge label correctness
  - `ConfidenceBar.test.tsx` (19 tests) — percentage display/rounding/visibility toggle, label rendering, bar width computation, colour threshold boundaries (green≥0.9, blue≥0.7, amber≥0.5, red<0.5), sm/md/lg size variants, 0%/100% edge cases, custom className
  - `RiskGauge.test.tsx` (19 tests) — numeric score display, "/ 100" denominator, risk level labels (low/moderate/high/critical), SVG element/circle count, size prop, progress colour from riskColors, strokeDashoffset math (0/50/100), custom className, badge colour mapping
  - `LoadingSkeleton.test.tsx` (14 tests) — Skeleton (animate-pulse, className, style), CardSkeleton structure, TableSkeleton row counts (default/custom), ChartSkeleton 12 bars with random heights, TextBlockSkeleton line counts and last-line 60% width
  - `ErrorBoundary.test.tsx` (8 tests) — healthy child rendering, default/custom fallback on throw, "Try again" button, error message in technical details, onError callback with Error + ErrorInfo, reset/recovery after retry, collapsible details element
  - `api.test.ts` (17 tests) — all 13 API endpoint functions (analyzeText, extractEntities, predictICD, summarizeText, assessRisk, createBatchJob, getBatchJob, getDocuments, getDocument, getModels, getDashboardStats, login, getCurrentUser, logout), axios instance config (baseURL, timeout), interceptor registration
  - `Dashboard.test.tsx` (14 tests) — page heading/description, 4 stat cards with formatted values, trend percentage indicators, Processing Volume/Recent Activity/Entity Distribution sections, entity type counts and labels, recent activity items and actions
- [x] **Total frontend tests: 113 passing**, 0 failures (Vitest 4.1, jsdom, ~1.2s runtime)

#### Post-PRD Enhancements — Session 9 (2026-03-24)
- [x] **92 new unit tests** across 6 new test modules (~1,750 lines) covering all previously untested API route handlers:
  - `test_analyze_route.py` (25 tests) — full pipeline stage helpers (_run_ner, _run_icd, _run_summary, _run_risk) with confidence filtering, compression maths, category mapping, factor capping; run_analysis orchestration (all-enabled, selective disable, document_id echo, text_length, 500/422 errors, audit trail, audit failure resilience)
  - `test_ner_route.py` (13 tests) — entity type filtering, negation/uncertainty exclusion, confidence threshold, combined filter composition, field mapping; endpoint happy path, sorting, model name passthrough, error propagation
  - `test_icd_route.py` (11 tests) — confidence filter, top_k forwarding, chapter toggle, empty/all-filtered cases; prediction endpoint + ICD code lookup (found/404/uppercase)
  - `test_summarize_route.py` (11 tests) — compression ratio edge cases, key_points toggle, detail_level forwarding, model metadata; endpoint timing and error handling
  - `test_risk_route.py` (17 tests) — _score_to_category boundaries, score/category normalisation, domain filtering, zero-score exclusion, protective factors, recommendations; endpoint happy path and errors
  - `test_models_route.py` (13 tests) — ORM serialisation (dates, UUIDs), list_models grouping, empty registry; get_model found/404/no-production/no-active
- [x] **Total test suite: 921 passing** (829 existing + 92 new), 0 failures, 0 errors (+ 40 SDK tests passing separately)

#### Post-PRD Enhancements — Session 8 (2026-03-24)
- [x] **Resolved all 35 test failures** — full suite now passes: **829 tests passing, 0 failures, 0 errors** (+ 40 SDK tests passing separately)
- [x] **Production bug fixes** (4 fixes in application code):
  - Replaced PostgreSQL-only `JSONB` columns with portable `JSON().with_variant(JSONB, "postgresql")` — enables SQLite test compatibility without compromising PostgreSQL production performance
  - Removed duplicate index definitions (`ix_audit_log_timestamp`, `ix_documents_content_hash`) that caused schema creation failures on SQLite
  - Fixed rate-limit middleware: return `JSONResponse` for 429 instead of `raise HTTPException` (Starlette's `BaseHTTPMiddleware` cannot propagate raised `HTTPException` as proper responses)
  - Added missing `risk_domains` and `patient_context` fields to `RiskScoreRequest` schema (route handler referenced them but the Pydantic model lacked them)
- [x] **Test infrastructure fixes** (14 test files, 11 categories of fixes):
  - Fixed bcrypt/passlib compatibility (downgraded bcrypt to 4.0.1)
  - Corrected 6 wrong mock patch targets (patching source modules, not import aliases)
  - Fixed Celery bound-task invocation in worker tests (`task.run()` not `__wrapped__`)
  - Stabilised PSI drift tests with seeded RNG and larger sample sizes
  - Rewrote integration test suite with correct constructor signatures and endpoint paths
  - Fixed `__builtins__` import mock for SHAP explainer fallback tests

#### Post-PRD Enhancements — Session 7 (2026-03-24)
- [x] **114 new unit tests** across 3 new test modules (~920 lines):
  - `test_rule_based_icd.py` (55 tests) — RuleBasedICDClassifier keyword-matching predictions for 15+ ICD-10 codes (I10, E11.9, J44.1, N18.9, etc.), synonym confidence boosting, deduplication, batch predict, top_k limiting, edge cases (empty text, no matches, very long docs, case insensitivity)
  - `test_extractive_summarizer.py` (30 tests) — TextRank pipeline full inference, detail-level behaviour (brief < standard < detailed), key findings extraction and clinical term filtering, Assessment/Plan section weighting, cosine similarity matrix correctness, PageRank convergence properties, edge cases (empty, single sentence, whitespace)
  - `test_main_app.py` (29 tests) — route registration verification for all 9 API endpoint groups, error-code-to-HTTP-status mapping (12 known codes + unknown fallback), FastAPI app metadata, root endpoint response format
- [x] **Bug fix: missing DentalAssessment and PeriodontalRiskAssessor** — pipeline.py imported `DentalAssessment` and `PeriodontalRiskAssessor` from `dental/model.py` but they didn't exist; added `DentalAssessment` dataclass (composite dental pipeline result) and `PeriodontalRiskAssessor` adapter class wrapping `PeriodontalRiskAssessment` with load/ensure_loaded lifecycle
- [x] **Bug fix: SummaryResult → SummarizationResult** — fixed stale import alias in `test_api.py` and corrected `conftest.py` mock_summarizer to use the actual `SummarizationResult` fields (key_findings, detail_level, sentence counts)

#### Post-PRD Enhancements — Session 6 (2026-03-24)
- [x] **100% docstring coverage** — Added module docstrings to all 17 `__init__.py` files, function docstrings to middleware dispatch methods, RBAC `check_role`, Pydantic validators, Alembic migration functions, retry decorator internals, metrics collector primitives, and test conftest overrides. Verified via full AST scan: zero public functions or modules without docstrings
- [x] **Python 3.12+ compatibility** — Replaced deprecated `datetime.utcnow()` with `datetime.now(UTC)` in Pydantic schema defaults (`common.py`, `batch.py`)
- [x] **Complete type annotations** — Added return type hints to `process_batch_task`, `health_check`, `require_role`, and `add_process_time_header`; zero untyped public function signatures remaining
- [x] **SDK test suite** — 40 new tests (~680 lines) across 2 modules:
  - `test_models.py` — Entity, ICDPrediction, Summary, RiskAssessment, BatchJob dataclasses; `AnalysisResult.from_dict` factory (empty, null, partial, full responses)
  - `test_client.py` — ClinIQClient init, all endpoint methods (analyze, NER, ICD, summarize, risk, batch submit/status/wait, health, models, login), pipeline flag forwarding, HTTP error propagation (404, 500), context manager lifecycle

#### Post-PRD Enhancements — Session 5 (2026-03-24)
- [x] **Wired real Celery dispatch to batch route** — `_process_batch_stub` replaced with `process_batch_task.delay()` from `worker.py`; documents are serialised to JSON-safe dicts for the task payload
- [x] **Wired real JWT auth to auth routes** — replaced `_get_current_user_placeholder` (which always raised 401) with the real `get_current_user` dependency from `deps.py` that validates JWT tokens and API keys
- [x] **Cleaned up stale imports** — removed unused `datetime`, `timezone`, `func`, `Request`, `NotFoundError`, `AuthenticationError`, `AuthorizationError` across batch and auth route modules
- [x] **New test modules** — 3 new test suites (~790 lines):
  - `test_health_route.py` — database/Redis/model probes, liveness/readiness, overall status aggregation (healthy/degraded/unhealthy)
  - `test_auth_route.py` — login (valid/invalid/disabled), registration (success/duplicate), profile retrieval, API key creation
  - `test_batch_route.py` — job creation + Celery dispatch, ETA scaling, serialised doc fields, progress computation, zero-total guard, 404 handling

#### Post-PRD Enhancements — Session 4 (2026-03-24)
- [x] **Bug fix: broken `deps.py` imports** — `MLPipeline` and `get_pipeline` never existed in the pipeline module; replaced with `ClinicalPipeline` and wired to model registry singletons (`get_ner_model`, `get_icd_model`, `get_summarizer`, `get_risk_scorer`)
- [x] **Expanded test coverage** — 4 new test modules (1,200+ lines):
  - `test_risk_scorer.py` — RiskFactor/RiskScore dataclasses, text/entity/ICD factor extraction, category scoring, overall score calculation, risk level thresholds, recommendation generation, end-to-end scoring (high/low-risk, empty text, custom weights)
  - `test_analysis_service.py` — document storage with SHA-256 hashing, entity persistence with field mapping, prediction storage, audit logging, mark_document_processed
  - `test_deps.py` — ClinicalPipeline construction from model registry, superuser gate authorization
  - `test_worker.py` — Celery process_batch_task (success, partial failure, progress updates), health_check task
- [x] **React ErrorBoundary component** — class-based error boundary with retry button, dark mode support, expandable error details, and optional `onError` callback for external reporting; wired into App.tsx root

#### Post-PRD Enhancements — Session 3 (2026-03-24)
- [x] **Expanded test coverage** — 4 new test modules (1200 lines) for previously untested ML infrastructure:
  - `test_feature_engineering.py` — ClinicalFeatureExtractor (TF-IDF + custom clinical features), BagOfWordsExtractor, keyword detection, empty input handling
  - `test_drift_detector.py` — PSI computation (numeric + categorical), TextDistributionMonitor (reference freezing, distribution shift), PredictionMonitor (confidence + label drift, multi-model independence)
  - `test_metrics_collector.py` — Fallback histogram/counter/gauge primitives, ModelMetrics (inference recording, error tracking, batch sizes, ms→s conversion), inference timer context manager
  - `test_shap_explainer.py` — SHAPExplanation dataclass, format_explanation (highlighted segments, direction labeling, attribution filtering), TokenSHAPExplainer fallback path

#### Post-PRD Enhancements — Session 2 (2026-03-24)
- [x] **Expanded test coverage** — 6 new test modules (738 lines) for previously untested code:
  - `test_model_registry.py` — lazy loading, singleton caching, thread safety (barrier test), reset, functional smoke tests
  - `test_rate_limit.py` — in-memory rate limiting, health bypass, 429 enforcement, per-client-key tracking
  - `test_auth_middleware.py` — active-user gate, superuser gate, RBAC factory, superuser bypass
  - `test_logging_middleware.py` — X-Request-ID UUID validity, X-Process-Time presence, configure_logging
  - `test_document_service.py` — single/batch analysis, InferenceError propagation, text hashing, pipeline caching
  - `test_composite_ner.py` — union/intersection/majority voting, deduplication, edge cases
- [x] **Educational inline comments** — Added design-decision documentation to:
  - `model_registry.py` — explains lazy loading, double-checked locking, thread safety, separation from routes
  - `rate_limit.py` — sliding window algorithm, in-memory fallback trade-offs, client identification
  - `auth.py` — dual auth schemes, RBAC factory, API key timing safety, optional auth
- [x] **Model registry health_check()** — new function reports per-model load status without triggering lazy loading
- [x] **Health endpoint integration** — `/health` now shows per-model readiness (ner, icd, summarizer, risk_scorer)

#### Post-PRD Hardening (2026-03-24)
- [x] Fixed broken `document_service.py` (syntax errors, bad imports, mismatched variables)
- [x] Fixed test `conftest.py` (wrong class names: MLPipeline→ClinicalPipeline, SummaryResult→SummarizationResult)
- [x] Removed bare TODO comments from route stubs
- [x] Added input validation to `preprocess_clinical_text()` (type checks, empty-string handling)
- [x] Added `retry.py` — configurable retry decorator with exponential back-off and jitter for inference calls
- [x] Added `validation.py` — pre-inference text validation (length, encoding, null bytes, noise ratio)
- [x] Full test suites for both new utility modules
- [x] **Wired real ML models to all API routes** — replaced all mock/placeholder inference stubs
  - Added `model_registry.py` — thread-safe singleton for lazy-loading and caching ML models
  - Added `RuleBasedICDClassifier` — 25 keyword/pattern rules for ICD-10 prediction
  - NER route → RuleBasedNERModel (pattern matching + negation/uncertainty detection)
  - ICD route → RuleBasedICDClassifier (keyword matching + synonym boosting)
  - Summarize route → ExtractiveSummarizer (TF-IDF TextRank + biased sentence scoring)
  - Risk route → RuleBasedRiskScorer (medication, diagnostic complexity, follow-up urgency)
  - Full-pipeline `/analyze` route → all four real model stages

#### Phase Completion Summary:

- [x] **Phase 0** — Foundation: Project structure, Docker Compose, FastAPI backend, database schema, Alembic migrations, API routes, ML model skeletons, middleware, Dockerfile, Makefile, CI/CD
- [x] **Phase 1** — Core ML: Text preprocessing, feature engineering, TF-IDF baselines, scispaCy NER, rule-based NER, ICD-10 classifiers (sklearn + transformer), evaluation metrics
- [x] **Phase 2** — Application Layer: React 18 + TypeScript dashboard with all pages (upload, entities, ICD, summary, risk, timeline, models), TanStack Query, Tailwind CSS, dark mode
- [x] **Phase 3** — Production Infrastructure: Docker production config, Nginx reverse proxy, Prometheus + Grafana monitoring, Kubernetes manifests with HPA, MLflow tracking
- [x] **Phase 4** — Advanced ML: Dental NLP module (tooth numbering, CDT codes, periodontal risk), SHAP explainability, attention visualization, data drift detection, metrics collection
- [x] **Phase 5** — Testing & Hardening: Unit tests (NER, ICD, dental, risk, summarization, security, config, pipeline), ML smoke tests, integration tests, Locust load testing
- [x] **Phase 6** — Documentation & Launch: README, architecture docs, API reference, model cards (NER, ICD-10, summarization), evaluation report, deployment guides, HIPAA compliance docs, Python SDK

---

## Table of Contents

1. [The App: What We're Building and Why](#1-the-app-what-were-building-and-why)
2. [Strategic CV Impact Analysis](#2-strategic-cv-impact-analysis)
3. [Technical Architecture Overview](#3-technical-architecture-overview)
4. [Technology Stack — Final Choices](#4-technology-stack--final-choices)
5. [Phase 0 — Foundation (Months 1–2)](#5-phase-0--foundation-months-12)
6. [Phase 1 — Core ML Pipeline (Months 3–5)](#6-phase-1--core-ml-pipeline-months-35)
7. [Phase 2 — Application Layer (Months 5–7)](#7-phase-2--application-layer-months-57)
8. [Phase 3 — Production Infrastructure (Months 7–9)](#8-phase-3--production-infrastructure-months-79)
9. [Phase 4 — Advanced ML Features (Months 9–11)](#9-phase-4--advanced-ml-features-months-911)
10. [Phase 5 — Scale, Observability & Hardening (Months 11–13)](#10-phase-5--scale-observability--hardening-months-1113)
11. [Phase 6 — Polish, Documentation & Launch (Months 13–16)](#11-phase-6--polish-documentation--launch-months-1316)
12. [Data Strategy & Compliance](#12-data-strategy--compliance)
13. [MLOps Pipeline Design](#13-mlops-pipeline-design)
14. [Testing Strategy](#14-testing-strategy)
15. [Repository Structure](#15-repository-structure)
16. [Month-by-Month Milestone Calendar](#16-month-by-month-milestone-calendar)
17. [Skills Demonstrated Per Phase](#17-skills-demonstrated-per-phase)
18. [Risk Register & Mitigation](#18-risk-register--mitigation)
19. [How to Talk About This Project in Interviews](#19-how-to-talk-about-this-project-in-interviews)
20. [Definition of Done — Production Checklist](#20-definition-of-done--production-checklist)

---

## 1. The App: What We're Building and Why

### Product Name: **ClinIQ** — Clinical Intelligence & Query Platform

### The One-Liner

> An AI-powered clinical NLP platform that ingests unstructured medical text (clinical notes, discharge summaries, medical transcriptions) and transforms them into structured, searchable, actionable clinical intelligence — with automated diagnosis coding, entity extraction, risk scoring, and clinical summarization.

### Why This Specific App

This isn't a random project. Every design decision is engineered to maximize your CV impact:

- **It solves a real problem** — 80% of clinical data is unstructured text; hospitals and health tech companies desperately need tools to make it usable
- **It proves end-to-end ML engineering** — from raw text ingestion to deployed, monitored prediction services
- **It leverages your clinical background** — your dentistry expertise informs the domain modeling, evaluation criteria, and clinical validation in ways no pure engineer could replicate
- **It demonstrates production engineering** — CI/CD, monitoring, testing, containerization, API design, authentication — not just a Jupyter notebook
- **It's a platform, not a script** — shows systems thinking, not just model training
- **It's in a high-demand vertical** — healthcare AI is one of the fastest-growing ML job markets

### Core Features

- **Clinical Text Ingestion** — Upload or paste clinical notes, discharge summaries, medical transcriptions; batch processing via CSV/JSON; REST API for programmatic access
- **Medical Entity Recognition (NER)** — Extract diseases, symptoms, medications, procedures, anatomical terms, lab values from unstructured text using fine-tuned biomedical NER models
- **Diagnosis Code Prediction (ICD-10)** — Predict ICD-10 diagnosis codes from clinical text; multi-label classification with confidence scores; explainability layer showing which text segments drive predictions
- **Clinical Text Summarization** — Generate concise clinical summaries from verbose notes; extractive and abstractive approaches; specialty-aware summarization (dental, general, surgical, etc.)
- **Risk Scoring** — Patient risk assessment based on extracted entities and clinical context; configurable risk models per specialty
- **Clinical Dashboard** — Real-time visualization of processed documents; entity frequency analysis; diagnosis distribution; temporal trends
- **Audit Trail & Compliance** — Full logging of all predictions; model version tracking; HIPAA-aware data handling patterns (even with synthetic data, demonstrate the architecture)

### What This Is NOT

- This is NOT a consumer health app — it's a professional clinical tool
- This is NOT a wrapper around GPT — it uses fine-tuned, domain-specific models you trained yourself
- This is NOT a demo — it has authentication, rate limiting, monitoring, tests, CI/CD, and documentation

---

## 2. Strategic CV Impact Analysis

### What Hiring Managers Will See

| CV Line Item | What It Proves | Phase |
|---|---|---|
| "Built end-to-end clinical NLP pipeline processing unstructured medical text" | ML Engineering, domain expertise | Phase 1 |
| "Fine-tuned BioBERT/ClinicalBERT for multi-label ICD-10 code prediction" | Deep learning, transfer learning, healthcare NLP | Phase 1–2 |
| "Designed and implemented medical NER system using transformer architectures" | NLP engineering, biomedical AI | Phase 1 |
| "Built FastAPI microservices serving ML models with sub-200ms latency" | Backend engineering, ML serving | Phase 2 |
| "Implemented MLOps pipeline: experiment tracking, model registry, automated retraining" | MLOps, production ML | Phase 3 |
| "Containerized deployment with Docker/Kubernetes, CI/CD via GitHub Actions" | DevOps, infrastructure | Phase 3 |
| "Built React dashboard with real-time clinical analytics and entity visualization" | Full-stack, data visualization | Phase 2–4 |
| "Designed HIPAA-aware data architecture with audit logging and encryption" | Compliance, security, healthcare domain | Phase 3 |
| "Implemented model monitoring: data drift detection, performance degradation alerts" | Production ML, observability | Phase 5 |
| "Achieved 87% F1 on ICD-10 prediction, outperforming baseline by 23%" | Measurable impact, evaluation rigor | Phase 4 |
| "Leveraged 5 years of clinical dental experience for domain-informed model evaluation" | Unique differentiator | Throughout |

### Roles This Project Qualifies You For

- ML Engineer (Healthcare / NLP)
- Applied Scientist — Clinical NLP
- NLP Engineer
- MLOps Engineer
- AI/ML Platform Engineer
- Healthcare AI Product Engineer
- Data Scientist — Healthcare (senior enough to build, not just analyze)

---

## 3. Technical Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  React SPA   │  │  REST API    │  │  Python SDK       │  │
│  │  Dashboard   │  │  Consumers   │  │  (pip installable)│  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────────┘  │
└─────────┼─────────────────┼──────────────────┼──────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      API GATEWAY                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI — Auth, Rate Limiting, Request Validation    │   │
│  │  OpenAPI/Swagger Docs — Versioned Endpoints           │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌──────────────────┐
│  NER Service   │ │ ICD-10     │ │  Summarization   │
│  (BioBERT +    │ │ Classifier │ │  Service         │
│   scispaCy)    │ │ (Clinical  │ │  (Fine-tuned     │
│                │ │  BERT)     │ │   BART/T5)       │
└───────┬────────┘ └─────┬──────┘ └────────┬─────────┘
        │                │                 │
        ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML SERVING LAYER                           │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Model        │  │  Feature     │  │  Prediction      │  │
│  │  Registry     │  │  Store       │  │  Cache (Redis)   │  │
│  │  (MLflow)     │  │  (Redis)     │  │                  │  │
│  └───────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  PostgreSQL   │  │  Redis       │  │  MinIO/S3        │  │
│  │  (Structured  │  │  (Cache,     │  │  (Model          │  │
│  │   data, audit)│  │   sessions)  │  │   artifacts,     │  │
│  │              │  │              │  │   raw data)      │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 OBSERVABILITY LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Prometheus   │  │  Grafana     │  │  Structured       │  │
│  │  (Metrics)    │  │  (Dashboards)│  │  Logging (JSON)   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

- **Microservices, not monolith** — Each ML model runs as an independent service; this proves you understand production ML architecture, not just model training
- **FastAPI over Flask** — Async support, automatic OpenAPI docs, type validation, modern Python
- **PostgreSQL + Redis** — You already know these from Pythoughts; reuse that expertise
- **MLflow for experiment tracking** — Industry standard; shows you know MLOps workflows
- **Docker Compose for local dev, Kubernetes-ready for prod** — Progressive complexity
- **MinIO for object storage** — S3-compatible; you already know it from Pythoughts

---

## 4. Technology Stack — Final Choices

### Backend & API

- **Python 3.11+** — Primary language
- **FastAPI** — API framework with async support, auto-docs
- **Pydantic v2** — Request/response validation (you already know this from DDD work)
- **SQLAlchemy 2.0 + Alembic** — ORM and migrations for PostgreSQL
- **Celery + Redis** — Background task queue for batch processing
- **JWT + OAuth2** — Authentication (leverage your Better Auth knowledge conceptually)

### Machine Learning

- **PyTorch 2.x** — Deep learning framework
- **HuggingFace Transformers** — Fine-tuning and serving transformer models
- **scikit-learn** — Classical ML baselines, preprocessing, evaluation
- **scispaCy** — Biomedical NLP preprocessing and entity recognition
- **spaCy 3.x** — NLP pipeline foundation
- **MLflow** — Experiment tracking, model registry, model serving
- **Weights & Biases** — Experiment visualization (free tier)
- **ONNX Runtime** — Model optimization for inference speed

### Data & Storage

- **PostgreSQL 16** — Primary database (structured data, audit logs, user data)
- **Redis 7** — Caching, session management, feature store, task queue broker
- **MinIO** — Object storage for model artifacts, raw data files, exports

### Frontend

- **React 18 + TypeScript** — Dashboard SPA
- **Tailwind CSS** — Styling (you're comfortable with it)
- **Recharts or D3.js** — Data visualization
- **React Query / TanStack Query** — Server state management

### Infrastructure & DevOps

- **Docker + Docker Compose** — Containerization
- **GitHub Actions** — CI/CD pipeline
- **Prometheus + Grafana** — Monitoring and dashboards
- **Nginx** — Reverse proxy, SSL termination
- **Terraform** (optional, Phase 5) — Infrastructure as code

### Testing

- **pytest** — Backend tests
- **pytest-cov** — Coverage reporting
- **locust** — Load testing
- **Great Expectations** — Data quality validation
- **Jest + React Testing Library** — Frontend tests

---

## 5. Phase 0 — Foundation (Months 1–2)

### Month 1: Research, Data, and Project Scaffolding

**Goal:** Understand the clinical NLP landscape deeply, acquire and explore datasets, and set up the project infrastructure.

#### Week 1–2: Domain Research & Dataset Acquisition

- **Research clinical NLP landscape:**
  - Read 5–10 key papers on clinical NLP (focus on ICD coding, NER, clinical summarization)
  - Study existing tools: Amazon Comprehend Medical, Google Healthcare NLP API, John Snow Labs Spark NLP — understand what exists so you can articulate how ClinIQ is different
  - Document findings in a `research/` directory in your repo

- **Acquire datasets:**
  - **MTSamples** — Download immediately (free, no credentialing) — medical transcription samples across 40+ specialties
  - **MIMIC-III** — Apply for access via PhysioNet (requires CITI training certificate, ~2 weeks to get approved) — de-identified ICU clinical notes
  - **i2b2/n2c2 2010 dataset** — Clinical NER shared task data (requires DUA)
  - **ICD-10 code mappings** — Download from CMS.gov (free)
  - **UMLS Metathesaurus** — Apply for license (free for research/education)
  - **Synthetic data generation plan** — Design a pipeline to generate synthetic clinical notes for testing (important for HIPAA demonstration)

- **Exploratory Data Analysis:**
  - MTSamples: document counts per specialty, text length distribution, vocabulary analysis, common medical terms frequency
  - Identify class imbalance issues and document mitigation strategies
  - Create a data quality report as a Jupyter notebook
  - Apply your clinical lens: which specialties have distinctive language? Where do you expect classification to be hardest? Document these hypotheses

#### Week 3–4: Project Infrastructure Setup

- **Repository initialization:**
  - Create monorepo: `cliniq` on GitHub
  - Set up branch protection rules (main, develop, feature branches)
  - Configure PR templates, issue templates, CODEOWNERS
  - Write comprehensive `.gitignore` for Python ML projects

- **Development environment:**
  - Create `docker-compose.yml` with PostgreSQL, Redis, MinIO
  - Set up Python virtual environment with `pyproject.toml` (use Poetry or uv)
  - Configure pre-commit hooks: black, ruff, mypy, isort
  - Create `Makefile` with common commands: `make lint`, `make test`, `make dev`, `make build`

- **Database schema v1:**
  - Design initial PostgreSQL schema:
    - `documents` — raw clinical text, metadata, upload source
    - `predictions` — model outputs linked to documents
    - `entities` — extracted NER entities linked to documents
    - `icd_codes` — ICD-10 code reference table
    - `audit_log` — every prediction request logged
    - `users` — authentication and API key management
  - Write Alembic migrations
  - Seed ICD-10 reference data

- **CI/CD foundation:**
  - GitHub Actions workflow: lint → type-check → test → build Docker image
  - Configure branch-based deployment (develop → staging, main → production)
  - Set up test coverage reporting with minimum threshold (start at 60%, increase over time)

### Month 2: Classical ML Baseline & Data Pipeline

**Goal:** Build the data ingestion pipeline and train classical ML baselines for all core tasks.

#### Week 5–6: Data Pipeline

- **Text preprocessing pipeline:**
  - Build a modular preprocessing pipeline:
    - Text cleaning: remove formatting artifacts, normalize whitespace, handle encoding issues
    - Sentence segmentation (clinical text has unusual sentence boundaries)
    - Tokenization with medical vocabulary awareness
    - Section detection (clinical notes have sections: HPI, Assessment, Plan, etc.)
  - Implement as a configurable pipeline class (not a script) so components can be swapped
  - Add unit tests for each preprocessing step

- **Feature engineering module:**
  - TF-IDF with medical domain customizations (custom stopwords, n-gram ranges)
  - Bag-of-words baseline features
  - Document-level features: length, section count, medical abbreviation density
  - Entity-based features (using scispaCy off-the-shelf for now)
  - Save feature extraction pipeline for reproducibility

#### Week 7–8: Classical ML Baselines

- **ICD-10 Code Prediction — Classical Baseline:**
  - Multi-label classification (documents can have multiple ICD codes)
  - Train and evaluate: Logistic Regression, Linear SVC, Random Forest, XGBoost
  - Proper evaluation: micro/macro F1, precision@k, hamming loss
  - Document baseline results thoroughly — these become your "before" numbers

- **Medical NER — Rule-Based + scispaCy Baseline:**
  - Run scispaCy's `en_ner_bc5cdr_md` on your dataset
  - Evaluate entity extraction quality: precision, recall, F1 per entity type
  - Build a rule-based layer on top for entities scispaCy misses (medication dosages, dental-specific terms)
  - Document baseline NER performance

- **Clinical Text Classification — Specialty Detection:**
  - Classify documents by medical specialty (40+ classes in MTSamples)
  - This is simpler than ICD-10 coding — good warm-up task
  - Achieve strong baseline here, then document clearly

- **Experiment tracking setup:**
  - Configure MLflow for local experiment tracking
  - Log all baseline experiments: parameters, metrics, artifacts
  - Create comparison dashboards in MLflow UI

---

## 6. Phase 1 — Core ML Pipeline (Months 3–5)

### Month 3: Fine-Tuning Transformers for Clinical NLP

**Goal:** Fine-tune domain-specific transformers that significantly outperform classical baselines.

#### Week 9–10: ICD-10 Prediction with ClinicalBERT

- **Model selection and setup:**
  - Evaluate pre-trained models: BioBERT, ClinicalBERT, PubMedBERT, GatorTron (if compute allows)
  - Set up training infrastructure: GPU-enabled environment (use Google Colab Pro, Lambda Labs, or your own GPU)
  - Implement proper data splits: train/val/test with stratification for multi-label

- **Fine-tuning pipeline:**
  - Tokenization with model-specific tokenizer, handling long documents (clinical notes often exceed 512 tokens)
  - Implement sliding window or hierarchical approach for long documents
  - Multi-label classification head: sigmoid activation, binary cross-entropy loss
  - Class imbalance handling: weighted loss, focal loss, or oversampling
  - Learning rate scheduling: linear warmup + cosine decay
  - Mixed precision training (fp16) for speed

- **Training discipline:**
  - Log everything to MLflow: hyperparameters, training curves, evaluation metrics per epoch
  - Save model checkpoints (best validation loss, best F1)
  - Document training time, compute cost, and convergence behavior
  - Compare against classical baseline — quantify improvement

#### Week 11–12: Medical NER with Fine-Tuned Transformers

- **NER model training:**
  - Fine-tune BioBERT for token classification on clinical NER datasets
  - Entity types: diseases, symptoms, medications, procedures, anatomy, lab values
  - Use IOB2 tagging scheme
  - Handle nested entities (common in clinical text)

- **Post-processing pipeline:**
  - Entity linking: map extracted entities to UMLS concepts (using scispaCy's UMLS linker)
  - Confidence thresholding: only surface entities above configurable confidence
  - Entity normalization: "HTN" → "Hypertension", "DM2" → "Type 2 Diabetes Mellitus"
  - Dental-specific entity rules (leverage your clinical knowledge here — this is your moat)

- **Evaluation:**
  - Per-entity-type precision, recall, F1
  - Compare: scispaCy baseline → your fine-tuned model
  - Error analysis with clinical interpretation — explain why certain entities are hard to extract

### Month 4: Summarization & Risk Scoring

#### Week 13–14: Clinical Summarization

- **Extractive summarization:**
  - Implement TextRank-based extractive summarization as baseline
  - Sentence importance scoring with clinical relevance weighting
  - Section-aware extraction (prioritize Assessment and Plan sections)

- **Abstractive summarization:**
  - Fine-tune BART-base or T5-small on clinical summarization
  - If compute is limited, use a smaller model and document the tradeoff
  - Generate summaries at configurable detail levels: brief (2–3 sentences), standard (paragraph), detailed (section-by-section)

- **Evaluation:**
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - Clinical relevance scoring: does the summary capture clinically important information?
  - Design a small manual evaluation protocol using your clinical expertise
  - Document: "As a clinician, here's what I look for in a clinical summary, and here's how the model performs against those criteria"

#### Week 15–16: Risk Scoring System

- **Risk model design:**
  - Combine NER outputs + ICD predictions + clinical features into risk scores
  - Implement configurable risk models per specialty
  - Start with a simple weighted scoring system, then add ML-based risk prediction
  - Risk categories: medication interaction risk, diagnostic complexity, follow-up urgency

- **Feature engineering for risk:**
  - Number and severity of extracted diagnoses
  - Medication count and known interaction patterns
  - Presence of critical keywords (e.g., "urgent," "emergent," "stat")
  - Patient age/demographics if available

- **Clinical validation:**
  - Use your dental/clinical expertise to validate risk scores make clinical sense
  - Document the clinical reasoning behind risk factor weighting
  - Create test cases from clinical scenarios you've encountered

### Month 5: Model Optimization & Integration

#### Week 17–18: Model Optimization

- **Inference speed optimization:**
  - Export models to ONNX format
  - Benchmark: PyTorch inference vs. ONNX Runtime
  - Quantization experiments: INT8 quantization, measure accuracy vs. speed tradeoff
  - Target: sub-200ms inference per document for NER and classification
  - Document all optimization results with proper benchmarks

- **Model ensemble (optional but impressive):**
  - Ensemble NER results from multiple models
  - Weighted voting for ICD-10 prediction
  - Document ensemble improvement over single model

#### Week 19–20: Pipeline Integration

- **Unified inference pipeline:**
  - Build a pipeline manager that orchestrates: preprocessing → NER → ICD prediction → summarization → risk scoring
  - Configurable pipeline: users can enable/disable components
  - Batch processing support for multiple documents
  - Async processing with Celery for large batches

- **Pipeline testing:**
  - Integration tests for the full pipeline
  - Test with adversarial inputs: empty text, non-English text, extremely long documents, non-medical text
  - Performance benchmarks: throughput (documents per second), latency (p50, p95, p99)
  - Memory profiling under load

---

## 7. Phase 2 — Application Layer (Months 5–7)

### Month 5–6: FastAPI Backend

**Goal:** Build a production-quality REST API that serves the ML pipeline.

#### API Design

- **Endpoint structure (versioned):**
  - `POST /api/v1/analyze` — Full pipeline analysis of clinical text
  - `POST /api/v1/ner` — NER-only extraction
  - `POST /api/v1/icd-predict` — ICD-10 code prediction only
  - `POST /api/v1/summarize` — Clinical summarization only
  - `POST /api/v1/risk-score` — Risk scoring only
  - `POST /api/v1/batch` — Batch processing (async, returns job ID)
  - `GET /api/v1/batch/{job_id}` — Check batch job status
  - `GET /api/v1/models` — List available models and versions
  - `GET /api/v1/health` — Health check with model status

- **Authentication & authorization:**
  - JWT-based authentication for dashboard users
  - API key authentication for programmatic access
  - Rate limiting per API key tier (free: 100 req/day, pro: 10K req/day)
  - Request logging to audit table

- **Request/response design:**
  - Pydantic v2 models for all request/response schemas
  - Consistent error response format with error codes
  - Support for configurable output: verbosity level, confidence thresholds, included models
  - Streaming responses for large batch results

#### Implementation Details

- **Model loading strategy:**
  - Lazy loading: models load on first request, not on app startup
  - Model warm-up endpoint for production deployments
  - Graceful degradation: if one model fails, return partial results
  - Model version pinning per request (optional)

- **Caching layer:**
  - Redis caching for repeated analysis of identical text
  - Cache key: hash of input text + model version + configuration
  - Configurable TTL per endpoint
  - Cache hit/miss metrics

- **Background processing:**
  - Celery workers for batch jobs
  - Progress tracking via Redis
  - Webhook support: notify caller when batch job completes
  - Dead letter queue for failed jobs

### Month 6–7: React Dashboard

**Goal:** Build a professional clinical analytics dashboard.

#### Dashboard Pages & Components

- **Document Upload & Analysis:**
  - Drag-and-drop file upload (TXT, CSV, PDF)
  - Paste clinical text directly
  - Real-time analysis progress indicator
  - Results display: highlighted entities in original text, ICD code predictions with confidence bars, clinical summary, risk score visualization

- **Entity Visualization:**
  - Interactive text annotation: click entities to see UMLS mapping, confidence, context
  - Color-coded entity types (diseases = red, medications = blue, procedures = green, anatomy = purple)
  - Entity frequency charts across documents
  - Entity co-occurrence network graph

- **Analytics Dashboard:**
  - Document processing volume over time
  - ICD code distribution charts
  - Model performance metrics (if ground truth available)
  - Specialty distribution of processed documents
  - Average risk scores by category

- **Model Management Page:**
  - Current model versions and deployment dates
  - Performance metrics per model version
  - Comparison between model versions
  - One-click rollback to previous version (frontend only — backend handles actual rollback)

- **Audit & Compliance Page:**
  - Full audit trail of all predictions
  - Search and filter by date, user, document, model version
  - Export audit logs as CSV
  - Data retention policy display

#### Frontend Engineering Standards

- TypeScript strict mode throughout
- Component library: build reusable components for entity tags, confidence bars, risk gauges
- Responsive design: works on desktop and tablet (clinical settings often use tablets)
- Dark mode support (because you love dark themes, and clinicians working night shifts will too)
- Accessibility: WCAG 2.1 AA compliance (screen reader support, keyboard navigation)
- State management with TanStack Query for server state
- Error boundaries and graceful degradation
- Loading skeletons for async data

---

## 8. Phase 3 — Production Infrastructure (Months 7–9)

### Month 7–8: Containerization & Deployment

#### Docker Architecture

- **Multi-stage Dockerfiles:**
  - `backend/Dockerfile` — Python API + ML models (multi-stage: build → production)
  - `frontend/Dockerfile` — React build → nginx serve
  - `worker/Dockerfile` — Celery workers with ML model access
  - Keep images lean: use `python:3.11-slim`, install only production dependencies

- **Docker Compose (local development):**
  ```
  Services:
  - api (FastAPI backend)
  - worker (Celery workers)
  - frontend (React dev server / nginx)
  - postgres (PostgreSQL 16)
  - redis (Redis 7)
  - minio (MinIO object storage)
  - mlflow (MLflow tracking server)
  - prometheus (metrics collection)
  - grafana (dashboards)
  ```

- **Health checks for all services:**
  - API: `/health` endpoint
  - Workers: Celery inspect ping
  - Database: connection test
  - Redis: PING
  - Models: warm-up verification

#### CI/CD Pipeline (GitHub Actions)

- **Pipeline stages:**
  1. **Lint & Type Check** — ruff, mypy, eslint, tsc
  2. **Unit Tests** — pytest with coverage gate (≥80%)
  3. **Integration Tests** — API tests against test database
  4. **Build Docker Images** — Multi-platform (amd64, arm64)
  5. **Security Scan** — Trivy for container vulnerability scanning
  6. **Push to Registry** — GitHub Container Registry or Docker Hub
  7. **Deploy to Staging** — Automatic on `develop` branch
  8. **Deploy to Production** — Manual approval on `main` branch

- **ML-specific CI:**
  - Model smoke tests: load model, run inference on 10 test cases, verify output schema
  - Data validation: run Great Expectations suite on test data
  - Model performance regression tests: ensure new code doesn't degrade model metrics

### Month 8–9: Security & Compliance Architecture

#### HIPAA-Aware Design (Critical for Healthcare CV)

Even though you're using synthetic/public data, implementing HIPAA-aware patterns proves you understand healthcare production requirements:

- **Data encryption:**
  - At rest: PostgreSQL encryption, MinIO server-side encryption
  - In transit: TLS everywhere (even between internal services in production)
  - API keys stored hashed (bcrypt)

- **Access control:**
  - Role-based access control (RBAC): admin, analyst, API-only
  - Principle of least privilege for service accounts
  - Session management with configurable timeout

- **Audit logging:**
  - Every prediction request logged: who, when, what input (hashed), what output, which model version
  - Immutable audit log (append-only table with no DELETE permissions)
  - Audit log export for compliance review

- **Data handling:**
  - No PHI in logs (log document IDs, not content)
  - Configurable data retention policies
  - Secure deletion with confirmation
  - Document the data flow architecture showing where PHI would exist and how it's protected

- **Document your compliance architecture:**
  - Create a `docs/security/` directory with:
    - Data flow diagram showing encryption boundaries
    - Access control matrix
    - Audit logging specification
    - Incident response plan template
  - This documentation alone is a significant CV differentiator

---

## 9. Phase 4 — Advanced ML Features (Months 9–11)

### Month 9–10: Advanced NLP Capabilities

#### Relation Extraction

- **Clinical relation extraction:**
  - Extract relationships between entities: "Patient takes [Metformin] for [Type 2 Diabetes]"
  - Relation types: treats, causes, diagnosed_with, contraindicated_with, dosage_of
  - Fine-tune a relation extraction model or use dependency parsing with rules
  - Visualize relations as a knowledge graph on the dashboard

#### Temporal Information Extraction

- **Timeline construction:**
  - Extract temporal expressions from clinical text ("3 days ago," "since 2019," "post-op day 2")
  - Build patient timelines from longitudinal notes
  - Visualize clinical events on a timeline component in the dashboard

#### Negation and Uncertainty Detection

- **Clinical negation handling:**
  - "Patient denies chest pain" → chest pain is NEGATED
  - "No evidence of malignancy" → malignancy is NEGATED
  - "Possible pneumonia" → pneumonia is UNCERTAIN
  - Implement NegEx algorithm or fine-tune a negation classifier
  - This is critical for clinical NLP and shows deep domain understanding

#### Dental-Specific NLP Module

- **Your unique differentiator:**
  - Build a specialty module for dental clinical notes
  - Dental-specific entity types: tooth numbers, surfaces, dental procedures, periodontal measurements
  - Dental diagnosis coding (CDT codes in addition to ICD-10)
  - Periodontal risk assessment model
  - This module is something NO other ML Engineer building a portfolio project will have
  - Write a detailed blog post about building dental NLP — this will get attention in both the dental tech and ML communities

### Month 10–11: Explainability & Model Interpretability

#### Explainability Features

- **Token-level attribution:**
  - Implement attention visualization for transformer predictions
  - Integrated Gradients or SHAP for feature attribution
  - Show users: "The model predicted ICD-10 code J18.9 (Pneumonia) because of these text segments..."
  - Highlight contributing tokens in the original document

- **Confidence calibration:**
  - Calibrate model confidence scores (Platt scaling or isotonic regression)
  - Show calibrated confidence: "When the model says 85% confidence, it's correct ~85% of the time"
  - Confidence vs. accuracy plots in the dashboard

- **Clinical validation interface:**
  - Build a simple annotation interface where a clinician can confirm/reject model predictions
  - Store feedback for potential model retraining
  - Calculate inter-annotator agreement if multiple reviewers
  - This demonstrates human-in-the-loop ML — a critical production concept

---

## 10. Phase 5 — Scale, Observability & Hardening (Months 11–13)

### Month 11–12: Monitoring & Observability

#### ML-Specific Monitoring

- **Data drift detection:**
  - Monitor input text distribution: vocabulary shifts, document length changes, specialty distribution changes
  - Statistical tests: KL divergence, Population Stability Index (PSI)
  - Alert when input distribution diverges significantly from training data
  - Dashboard showing drift metrics over time

- **Model performance monitoring:**
  - Track prediction confidence distribution over time
  - Monitor prediction latency (p50, p95, p99) per model
  - Detect performance degradation: if average confidence drops, alert
  - A/B testing framework for comparing model versions

- **Infrastructure monitoring:**
  - Prometheus metrics: request count, latency, error rate, GPU utilization, memory usage
  - Grafana dashboards: API health, model serving metrics, infrastructure utilization
  - Alerting rules: error rate > 1%, latency p95 > 500ms, memory > 80%

#### Operational Dashboards

- **Four dashboards:**
  1. **API Health** — Request volume, latency, error rates, status codes
  2. **ML Models** — Inference latency, prediction distribution, confidence trends, drift metrics
  3. **Infrastructure** — CPU, memory, disk, network per service
  4. **Business Metrics** — Documents processed, unique users, feature usage

### Month 12–13: Performance & Scalability

#### Load Testing

- **Locust load testing suite:**
  - Simulate realistic traffic patterns: burst analysis requests, steady batch processing, concurrent users
  - Test targets: 100 concurrent users, 50 requests/second, sub-500ms p95 latency
  - Identify bottlenecks: is it model inference, database, or network?
  - Document results with charts and analysis

#### Performance Optimization

- **Inference optimization:**
  - Model batching: batch concurrent requests for GPU efficiency
  - Async inference with proper request queuing
  - Model caching: keep hot models in memory, lazy-load cold models
  - Connection pooling for database and Redis

- **Database optimization:**
  - Query optimization: EXPLAIN ANALYZE on slow queries
  - Proper indexing: audit log (timestamp, user_id), predictions (document_id, model_version)
  - Partitioning: audit log table by month (it will grow fast)
  - Read replicas if needed (design for it, implement if you have resources)

#### Kubernetes Readiness (Optional but Impressive)

- **Kubernetes manifests:**
  - Deployment, Service, Ingress for each component
  - HorizontalPodAutoscaler based on CPU and custom metrics (request queue depth)
  - ConfigMaps and Secrets management
  - Resource requests and limits tuned from load testing
  - This can be documented as "Kubernetes-ready" even if you deploy on Docker Compose in practice

---

## 11. Phase 6 — Polish, Documentation & Launch (Months 13–16)

### Month 13–14: Documentation & Developer Experience

#### Documentation Pyramid

- **README.md (Top Level):**
  - One-paragraph description
  - Architecture diagram
  - Quick start (3 commands to run locally)
  - Feature overview with screenshots
  - Tech stack summary
  - Links to detailed docs

- **docs/architecture.md:**
  - System architecture with diagrams
  - Design decisions and tradeoffs (ADRs — Architecture Decision Records)
  - Data flow diagrams
  - API contract documentation

- **docs/ml/:**
  - `model-card-icd10.md` — Model card for ICD-10 predictor (following Google's Model Card format)
  - `model-card-ner.md` — Model card for NER model
  - `model-card-summarization.md` — Model card for summarization
  - `training-guide.md` — How to retrain models
  - `evaluation-report.md` — Comprehensive evaluation with clinical analysis
  - Each model card includes: intended use, training data, performance metrics, limitations, ethical considerations

- **docs/api/:**
  - Auto-generated OpenAPI docs (from FastAPI)
  - Usage examples in Python, curl, JavaScript
  - Authentication guide
  - Rate limiting documentation
  - Error code reference

- **docs/deployment/:**
  - Local development setup guide
  - Docker Compose deployment guide
  - Production deployment guide
  - Environment variables reference
  - Troubleshooting guide

- **docs/security/:**
  - HIPAA compliance architecture
  - Data flow diagram with encryption boundaries
  - Access control documentation
  - Audit logging specification

#### Python SDK (pip installable)

- **Build a thin Python client:**
  - `pip install cliniq-client`
  - Provides typed interface to the ClinIQ API
  - Handles authentication, retries, pagination
  - This is a cherry on top — shows you think about developer experience, not just model accuracy
  - Publish to PyPI (or at minimum, make it installable from GitHub)

### Month 14–15: Blog Posts & Content

#### Write These Blog Posts

1. **"From Dentist to ML Engineer: Building a Clinical NLP Platform"** — Your origin story; this will get engagement because it's a unique career path
2. **"Fine-Tuning ClinicalBERT for ICD-10 Code Prediction: A Practical Guide"** — Technical deep dive; useful for others, establishes expertise
3. **"Why Clinical NLP Is Harder Than You Think: Lessons from Building ClinIQ"** — Domain insights that only someone with clinical experience would know (negation, abbreviations, section structure, clinical reasoning patterns)
4. **"Building a Dental NLP Module: An Underserved Niche in Healthcare AI"** — Your unique angle; nobody else is writing about this
5. **"Production ML for Healthcare: HIPAA-Aware Architecture Patterns"** — Shows production thinking; relevant to hiring managers at health tech companies

#### Where to Publish

- Your personal blog (pythoughts.com) — Establish your platform
- Medium (Towards Data Science or Towards AI publications) — Reach
- Dev.to — Developer community
- LinkedIn articles — Professional network
- Cross-post strategically: original on your blog, syndicated elsewhere

### Month 15–16: Final Polish & Launch

#### GitHub Repository Polish

- **Pinned repos (3 repos, this order):**
  1. `cliniq` — The main platform (star of the show)
  2. `clinical-nlp-classification` — Phase 1 classical ML project (shows fundamentals)
  3. `pythinker-flow` — Agent/LLM engineering (shows breadth)

- **GitHub profile README update:**
  - Lead with: "ML Engineer | Healthcare NLP | 5 Years Clinical Experience"
  - Feature ClinIQ with a screenshot or architecture diagram
  - Link to blog posts and live demo

- **Repository badges:**
  - CI/CD status badge
  - Test coverage badge
  - Python version badge
  - License badge
  - Documentation badge (link to docs)

#### Live Demo Deployment

- **Deploy a live demo instance:**
  - Pre-loaded with synthetic clinical notes
  - Rate-limited free tier for public access
  - Deploy on a VPS (you already know Dokploy/VPS deployment from Pythoughts)
  - SSL certificate, custom domain (e.g., cliniq.techmatrix.com or cliniq-demo.xyz)

- **Demo walkthrough video:**
  - 3–5 minute screencast showing the platform in action
  - Upload to YouTube, embed in README
  - Shows: document upload → entity extraction → ICD prediction → summary → risk score → dashboard

#### Resume & LinkedIn Finalization

- **Resume bullet points (craft these carefully):**
  - "Designed and built ClinIQ, a production clinical NLP platform processing unstructured medical text into structured clinical intelligence with automated ICD-10 coding, entity extraction, and risk scoring"
  - "Fine-tuned BioBERT and ClinicalBERT models achieving XX% F1 on ICD-10 prediction and XX% F1 on medical NER, with clinical validation leveraging 5 years of dental practice experience"
  - "Engineered ML serving infrastructure with FastAPI, Docker, and CI/CD pipeline; sub-200ms inference latency, comprehensive monitoring with Prometheus/Grafana, and HIPAA-aware audit architecture"
  - "Published X blog posts on clinical NLP and healthcare AI, demonstrating domain expertise at the intersection of clinical practice and machine learning engineering"

- **LinkedIn strategy:**
  - Update headline: "ML Engineer | Healthcare NLP | Building AI Systems for Clinical Intelligence"
  - Feature the ClinIQ project
  - Share blog posts weekly during Month 15–16
  - Engage with healthcare AI community posts

---

## 12. Data Strategy & Compliance

### Dataset Usage Plan

| Dataset | Purpose | Access | Timeline |
|---|---|---|---|
| MTSamples | Specialty classification, initial NER training | Free, immediate | Month 1 |
| MIMIC-III | ICD-10 prediction, clinical NER, summarization | PhysioNet DUA (~2 weeks) | Month 1–2 |
| i2b2/n2c2 2010 | Clinical NER benchmarking | DUA required | Month 2 |
| ICD-10 codes | Reference data for prediction targets | CMS.gov, free | Month 1 |
| UMLS | Entity linking and normalization | Free license for research | Month 2 |
| Synthetic data | Testing, demo, HIPAA compliance demonstration | Self-generated | Month 3+ |

### Synthetic Data Generation

- Build a synthetic clinical note generator for testing and demos
- Use templates + controlled randomization to create realistic but non-real clinical notes
- Include dental-specific synthetic notes (leverage your expertise)
- Document the generation process and use it exclusively for the public demo
- This demonstrates HIPAA awareness: "I never expose real clinical data, even de-identified data, in public demos"

### Data Versioning

- Use DVC (Data Version Control) for dataset versioning
- Track data transformations and preprocessing steps
- Ensure reproducibility: anyone checking out a specific commit can reproduce results with the exact same data

---

## 13. MLOps Pipeline Design

### Experiment Tracking

```
MLflow Tracking Server
├── Experiments
│   ├── icd10-classification
│   │   ├── baseline-logreg
│   │   ├── baseline-svm
│   │   ├── clinicalbert-v1
│   │   ├── clinicalbert-v2-focal-loss
│   │   └── ensemble-v1
│   ├── medical-ner
│   │   ├── scispacy-baseline
│   │   ├── biobert-ner-v1
│   │   └── biobert-ner-v2-dental
│   ├── summarization
│   │   ├── textrank-baseline
│   │   ├── bart-base-v1
│   │   └── t5-small-v1
│   └── risk-scoring
│       ├── weighted-rules-v1
│       └── xgboost-v1
└── Model Registry
    ├── icd10-predictor (Production: v3, Staging: v4)
    ├── medical-ner (Production: v2)
    ├── summarizer (Production: v1)
    └── risk-scorer (Production: v1)
```

### Model Lifecycle

1. **Training** — Researcher trains model, logs to MLflow
2. **Evaluation** — Automated evaluation suite runs, metrics logged
3. **Registration** — Model registered in MLflow Model Registry as "Staging"
4. **Validation** — Automated tests run against staging model (smoke tests, regression tests, latency benchmarks)
5. **Promotion** — Manual approval promotes model to "Production"
6. **Serving** — API loads Production model
7. **Monitoring** — Continuous monitoring for drift and performance degradation
8. **Retirement** — Previous version kept for rollback, eventually archived

### Automated Retraining Pipeline (Design, implement if time allows)

- Triggered by: data drift detection, scheduled (monthly), manual trigger
- Steps: pull latest data → preprocess → train → evaluate → register → notify
- Gate: only auto-promote if metrics exceed current production model
- Notification: Slack/email alert on training completion with metrics comparison

---

## 14. Testing Strategy

### Testing Pyramid

```
           ╱  E2E Tests  ╲           ← Few: full workflow tests
          ╱  (Cypress/     ╲
         ╱   Playwright)    ╲
        ╱                    ╲
       ╱  Integration Tests   ╲      ← Some: API + DB + Model
      ╱  (pytest + TestClient) ╲
     ╱                          ╲
    ╱     Unit Tests              ╲   ← Many: functions, classes
   ╱     (pytest, Jest)            ╲
  ╱─────────────────────────────────╲
```

### Test Categories

- **Unit tests (target: 80%+ coverage):**
  - Preprocessing functions: tokenization, cleaning, section detection
  - Feature engineering: TF-IDF, entity features
  - API route handlers (with mocked models)
  - Pydantic model validation
  - Utility functions

- **Integration tests:**
  - API endpoints with real database (test container)
  - Model loading and inference pipeline
  - Celery task execution
  - Authentication and authorization flows
  - Cache hit/miss scenarios

- **ML-specific tests:**
  - Model smoke tests: load model, run inference, verify output schema and types
  - Performance regression tests: ensure metrics don't degrade below thresholds
  - Data validation: input schema, output schema, value ranges
  - Adversarial inputs: empty text, non-English text, extremely long text, injection attempts

- **Load tests (Locust):**
  - Sustained load: 50 req/s for 10 minutes
  - Burst: 200 req/s for 30 seconds
  - Soak: 10 req/s for 1 hour (memory leak detection)

- **E2E tests (optional, time permitting):**
  - Upload document → verify entities extracted → verify ICD codes predicted → verify summary generated
  - Batch upload → verify job completion → verify results

---

## 15. Repository Structure

```
cliniq/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Lint, test, build on PR
│   │   ├── cd-staging.yml            # Deploy to staging on develop merge
│   │   ├── cd-production.yml         # Deploy to prod on main merge (manual gate)
│   │   └── ml-tests.yml              # Model smoke tests
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── routes/
│   │   │   │   │   ├── analyze.py    # Full pipeline analysis
│   │   │   │   │   ├── ner.py        # NER endpoint
│   │   │   │   │   ├── icd.py        # ICD-10 prediction
│   │   │   │   │   ├── summarize.py  # Summarization
│   │   │   │   │   ├── risk.py       # Risk scoring
│   │   │   │   │   ├── batch.py      # Batch processing
│   │   │   │   │   ├── models.py     # Model management
│   │   │   │   │   └── health.py     # Health checks
│   │   │   │   └── deps.py           # Dependency injection
│   │   │   └── middleware/
│   │   │       ├── auth.py
│   │   │       ├── rate_limit.py
│   │   │       └── logging.py
│   │   ├── core/
│   │   │   ├── config.py             # Settings (Pydantic BaseSettings)
│   │   │   ├── security.py           # JWT, API keys
│   │   │   └── exceptions.py         # Custom exceptions
│   │   ├── db/
│   │   │   ├── models.py             # SQLAlchemy models
│   │   │   ├── session.py            # Database session
│   │   │   └── migrations/           # Alembic migrations
│   │   ├── ml/
│   │   │   ├── pipeline.py           # Unified inference pipeline
│   │   │   ├── ner/
│   │   │   │   ├── model.py          # NER model wrapper
│   │   │   │   ├── preprocessing.py  # NER-specific preprocessing
│   │   │   │   └── postprocessing.py # Entity linking, normalization
│   │   │   ├── icd/
│   │   │   │   ├── model.py          # ICD-10 classifier wrapper
│   │   │   │   ├── preprocessing.py
│   │   │   │   └── explainability.py # SHAP/attention viz
│   │   │   ├── summarization/
│   │   │   │   ├── model.py
│   │   │   │   └── preprocessing.py
│   │   │   ├── risk/
│   │   │   │   ├── scorer.py
│   │   │   │   └── rules.py
│   │   │   ├── dental/               # YOUR DIFFERENTIATOR
│   │   │   │   ├── entities.py       # Dental-specific NER
│   │   │   │   ├── cdt_codes.py      # CDT code prediction
│   │   │   │   └── perio_risk.py     # Periodontal risk model
│   │   │   └── utils/
│   │   │       ├── text_preprocessing.py
│   │   │       ├── feature_engineering.py
│   │   │       └── metrics.py
│   │   ├── services/
│   │   │   ├── document_service.py
│   │   │   ├── prediction_service.py
│   │   │   └── audit_service.py
│   │   └── main.py                   # FastAPI app initialization
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── ml/
│   │   └── conftest.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── Makefile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DocumentUpload/
│   │   │   ├── EntityViewer/
│   │   │   ├── ICDResults/
│   │   │   ├── ClinicalSummary/
│   │   │   ├── RiskGauge/
│   │   │   ├── Timeline/
│   │   │   ├── Dashboard/
│   │   │   └── common/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── Dockerfile
│   └── package.json
├── ml/
│   ├── notebooks/                    # Exploration, EDA, experiments
│   │   ├── 01_eda_mtsamples.ipynb
│   │   ├── 02_baseline_models.ipynb
│   │   ├── 03_clinicalbert_finetuning.ipynb
│   │   ├── 04_ner_experiments.ipynb
│   │   └── 05_dental_nlp.ipynb
│   ├── training/
│   │   ├── train_icd.py
│   │   ├── train_ner.py
│   │   ├── train_summarizer.py
│   │   └── configs/                  # Training configs (YAML)
│   ├── evaluation/
│   │   ├── evaluate_models.py
│   │   └── clinical_validation.py
│   └── data/
│       ├── raw/
│       ├── processed/
│       └── synthetic/
├── sdk/                              # Python SDK
│   ├── cliniq_client/
│   │   ├── client.py
│   │   ├── models.py
│   │   └── exceptions.py
│   ├── pyproject.toml
│   └── README.md
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   ├── nginx/
│   ├── prometheus/
│   ├── grafana/
│   └── k8s/                          # Kubernetes manifests (optional)
├── docs/
│   ├── architecture.md
│   ├── api/
│   ├── ml/
│   │   ├── model-card-icd10.md
│   │   ├── model-card-ner.md
│   │   └── training-guide.md
│   ├── security/
│   │   ├── hipaa-architecture.md
│   │   └── data-flow-diagram.md
│   └── deployment/
├── research/                         # Papers, notes, references
├── scripts/                          # Utility scripts
├── .pre-commit-config.yaml
├── Makefile
├── README.md
└── LICENSE
```

---

## 16. Month-by-Month Milestone Calendar

### Month 1 — Research & Scaffolding
- [ ] 5–10 clinical NLP papers read and summarized
- [ ] MTSamples downloaded and EDA complete
- [ ] MIMIC-III access application submitted
- [ ] Git repository initialized with full project structure
- [ ] Docker Compose development environment running
- [ ] PostgreSQL schema v1 with Alembic migrations
- [ ] CI pipeline: lint → type-check → test (skeleton)
- [ ] Pre-commit hooks configured

### Month 2 — Data Pipeline & Baselines
- [ ] Text preprocessing pipeline built and tested
- [ ] Feature engineering module complete
- [ ] Classical ML baselines trained for specialty classification
- [ ] Classical ML baselines trained for ICD-10 prediction (multi-label)
- [ ] scispaCy NER baseline evaluated
- [ ] MLflow experiment tracking configured
- [ ] All baselines logged with full metrics

### Month 3 — Transformer Fine-Tuning (Classification)
- [ ] ClinicalBERT fine-tuned for ICD-10 prediction
- [ ] Long document handling strategy implemented (sliding window or hierarchical)
- [ ] Multi-label classification with proper loss function
- [ ] Results compared against classical baseline (improvement quantified)
- [ ] Training pipeline reproducible and documented

### Month 4 — NER + Summarization Models
- [ ] BioBERT fine-tuned for medical NER
- [ ] Entity linking pipeline (UMLS) implemented
- [ ] NER evaluation complete with per-entity-type metrics
- [ ] Extractive summarization baseline built
- [ ] BART/T5 fine-tuned for clinical summarization
- [ ] Risk scoring system designed and implemented

### Month 5 — Model Optimization & Integration
- [ ] ONNX export and optimization complete
- [ ] Inference benchmarks documented (latency, throughput)
- [ ] Unified inference pipeline orchestrating all models
- [ ] Batch processing pipeline with Celery
- [ ] Integration tests for full pipeline

### Month 6 — FastAPI Backend
- [ ] All API endpoints implemented and documented
- [ ] Authentication (JWT + API keys) working
- [ ] Rate limiting implemented
- [ ] Request validation with Pydantic v2
- [ ] Caching layer with Redis
- [ ] Audit logging for all predictions
- [ ] OpenAPI documentation auto-generated
- [ ] Backend test coverage ≥ 80%

### Month 7 — React Dashboard
- [ ] Document upload and analysis page complete
- [ ] Entity visualization with interactive annotation
- [ ] ICD-10 prediction results display with confidence
- [ ] Clinical summary display
- [ ] Risk score visualization
- [ ] Analytics dashboard with charts
- [ ] Responsive design, dark mode support
- [ ] Frontend tests written

### Month 8 — Containerization & CI/CD
- [ ] Multi-stage Dockerfiles for all services
- [ ] Docker Compose production configuration
- [ ] GitHub Actions CI/CD pipeline complete
- [ ] Automated testing in CI (unit, integration, ML smoke tests)
- [ ] Security scanning (Trivy) in pipeline
- [ ] Staging deployment automated

### Month 9 — Security & Compliance
- [ ] HIPAA-aware data architecture documented
- [ ] Encryption at rest and in transit configured
- [ ] RBAC implemented
- [ ] Audit log export functionality
- [ ] Security documentation complete
- [ ] Data retention policy implemented

### Month 10 — Advanced NLP Features
- [ ] Relation extraction implemented
- [ ] Negation/uncertainty detection working
- [ ] Temporal information extraction functional
- [ ] Dental NLP module built (entities, CDT codes, perio risk)
- [ ] Knowledge graph visualization on dashboard

### Month 11 — Explainability
- [ ] Attention visualization for transformer predictions
- [ ] SHAP/Integrated Gradients for ICD-10 predictions
- [ ] Confidence calibration implemented
- [ ] Clinical validation interface built
- [ ] Explainability integrated into API responses and dashboard

### Month 12 — Monitoring & Observability
- [ ] Prometheus metrics for all services
- [ ] Grafana dashboards (API health, ML models, infrastructure, business)
- [ ] Data drift detection implemented
- [ ] Performance degradation alerting configured
- [ ] Structured logging throughout

### Month 13 — Load Testing & Optimization
- [ ] Locust load testing suite built
- [ ] Performance benchmarks documented
- [ ] Bottlenecks identified and optimized
- [ ] Database query optimization complete
- [ ] Sub-200ms inference latency achieved (or documented why not and what would fix it)

### Month 14 — Documentation
- [ ] README polished with architecture diagram and screenshots
- [ ] Model cards written for all models
- [ ] API documentation complete with examples
- [ ] Deployment guide written
- [ ] Security documentation finalized
- [ ] Python SDK published (GitHub or PyPI)

### Month 15 — Content & Visibility
- [ ] Blog post 1: "From Dentist to ML Engineer" published
- [ ] Blog post 2: "Fine-Tuning ClinicalBERT" published
- [ ] Blog post 3: "Why Clinical NLP Is Hard" published
- [ ] Blog post 4: "Building Dental NLP" published
- [ ] Blog post 5: "Production ML for Healthcare" published
- [ ] All posts shared on LinkedIn with engagement strategy

### Month 16 — Launch & Job Prep
- [ ] Live demo deployed (VPS, custom domain, SSL)
- [ ] Demo walkthrough video recorded and uploaded
- [ ] GitHub pinned repos updated
- [ ] GitHub profile README rewritten
- [ ] Resume updated with ClinIQ bullet points
- [ ] LinkedIn fully updated
- [ ] 20+ targeted job applications submitted
- [ ] Interview prep complete (project explanations rehearsed)

---

## 17. Skills Demonstrated Per Phase

| Phase | ML Skills | Engineering Skills | Domain Skills |
|---|---|---|---|
| 0 — Foundation | EDA, feature engineering, classical ML | Git, Docker, PostgreSQL, CI/CD | Clinical data understanding |
| 1 — Core ML | Transfer learning, fine-tuning, NER, multi-label classification | Modular pipeline design, experiment tracking | Medical terminology, ICD-10 coding |
| 2 — App Layer | Model serving, inference optimization | FastAPI, React, API design, caching | Clinical workflow design |
| 3 — Production | MLOps, model registry | Docker, CI/CD, security, HIPAA architecture | Healthcare compliance |
| 4 — Advanced ML | Relation extraction, negation detection, explainability | Knowledge graph visualization | Dental NLP, clinical reasoning |
| 5 — Scale | Drift detection, monitoring, A/B testing | Prometheus, Grafana, load testing | Clinical validation |
| 6 — Polish | Model cards, evaluation reports | SDK development, documentation | Blog content, domain thought leadership |

---

## 18. Risk Register & Mitigation

### Risk 1: Scope Creep
- **Probability:** Very High
- **Impact:** Project never finishes
- **Mitigation:** Follow the month-by-month calendar strictly. If a feature takes longer than planned, cut scope on the current feature before moving to the next phase. Ship a working version of each phase before adding complexity
- **Rule:** If you're more than 2 weeks behind on any phase, skip the "nice-to-have" features in that phase and move on

### Risk 2: Compute Limitations
- **Probability:** Medium
- **Impact:** Can't train large models
- **Mitigation:** Use Google Colab Pro ($10/month) for training. Use smaller models (BioBERT-base, not large). Use ONNX optimization aggressively. Document compute constraints honestly — hiring managers respect honesty about tradeoffs
- **Backup:** If GPU access is impossible, use classical ML + smaller transformers (DistilBERT) and document why

### Risk 3: Dataset Access Delays
- **Probability:** Medium (MIMIC-III approval can take weeks)
- **Impact:** Phase 1 delayed
- **Mitigation:** Start with MTSamples immediately (no approval needed). Apply for MIMIC-III in Week 1. Design pipeline to be dataset-agnostic so you can swap datasets later
- **Backup:** MTSamples alone is sufficient for a strong project; MIMIC-III is bonus

### Risk 4: Burnout
- **Probability:** High (16 months is a long project alongside M.S. and work)
- **Impact:** Project abandoned
- **Mitigation:** Build in rest weeks. Celebrate milestones. Keep Sunday check-in ritual. Remember: a finished 80% of this plan is still an incredible project. Give yourself permission to cut Phase 4 advanced features if needed
- **Rule:** Take one full week off every 4 months. No guilt. Sustainability beats intensity

### Risk 5: Perfectionism
- **Probability:** High (you have strong aesthetic sensibility)
- **Impact:** Months spent on UI when ML pipeline isn't done
- **Mitigation:** ML pipeline is ALWAYS the priority. Dashboard is secondary. A beautiful dashboard with weak ML is worse than an ugly dashboard with strong ML. Build the dashboard in Month 7, not Month 1
- **Rule:** No frontend work until Month 6

### Risk 6: M.S. Coursework Conflicts
- **Probability:** Medium
- **Impact:** ClinIQ progress slows during heavy coursework periods
- **Mitigation:** Look at your M.S. course schedule. Identify heavy weeks in advance. Plan ClinIQ light weeks around exam periods. Try to align course projects with ClinIQ when possible (use clinical datasets in coursework)
- **Opportunity:** DTSC-670 and subsequent AI electives may directly overlap with ClinIQ skills

---

## 19. How to Talk About This Project in Interviews

### The 30-Second Pitch

> "I built ClinIQ, a production clinical NLP platform that transforms unstructured medical text into structured clinical intelligence. It uses fine-tuned biomedical transformers for diagnosis coding, entity extraction, and clinical summarization, served through a FastAPI backend with a React dashboard. My 5 years as a practicing dentist informed everything from the model evaluation to a specialty dental NLP module that no one else in the ML community has built."

### The 2-Minute Technical Deep Dive

> "ClinIQ processes clinical notes through a multi-model pipeline. For ICD-10 code prediction, I fine-tuned ClinicalBERT with a multi-label classification head using focal loss to handle class imbalance, achieving XX% micro-F1. For entity extraction, I fine-tuned BioBERT for token classification with UMLS entity linking in the post-processing layer. The summarization component uses fine-tuned BART-base.
>
> The models are served through FastAPI with ONNX Runtime for optimized inference — I got latency down to under 200ms per document. The pipeline supports batch processing via Celery workers with Redis as the broker. Everything runs in Docker containers with a full CI/CD pipeline through GitHub Actions, including model smoke tests and data validation.
>
> What I'm most proud of is the dental NLP module — I built dental-specific entity recognition for tooth numbers, surfaces, and periodontal measurements. As a former dentist, I could validate the model's clinical outputs in ways that pure engineers can't. That clinical lens also informed my evaluation: I didn't just measure F1 scores, I analyzed errors through a clinical significance lens."

### Common Interview Questions & Answers

**"What was the hardest technical challenge?"**

> "Handling clinical text that exceeds BERT's 512-token limit. Discharge summaries can be thousands of tokens. I implemented a hierarchical approach — segment the document into sections, process each section independently, then aggregate predictions with section-weighted voting. The weights were informed by clinical importance: the Assessment and Plan sections should influence ICD coding more than social history."

**"How did you handle class imbalance in ICD-10 prediction?"**

> "The ICD-10 code distribution follows a severe power law — a few codes appear in 30% of documents while most codes appear in less than 1%. I used three approaches: focal loss to down-weight easy examples, stratified multi-label splitting to ensure rare codes appear in all splits, and a hierarchical prediction approach that first predicts the ICD chapter, then the specific code within that chapter."

**"How would you scale this to a hospital system processing 10,000 notes per day?"**

> "The current architecture already supports horizontal scaling because the ML models run as stateless services behind a load balancer. I'd add GPU auto-scaling based on queue depth, implement model batching for GPU efficiency, and add a message queue (Kafka) between ingestion and processing. I've documented Kubernetes manifests that would support this. The monitoring is already in place — Prometheus tracks inference latency and throughput, and Grafana dashboards would show when we need to scale."

**"How did your dentistry background actually help?"**

> "Three specific ways. First, in evaluation: when the NER model extracted 'crown' from a dental note, I knew whether it was a dental crown (prosthetic) or a tooth crown (anatomy) — that kind of disambiguation requires clinical knowledge. Second, in the dental NLP module: I built entity recognition for tooth numbering systems (universal, Palmer, FDI) and periodontal charting terminology that doesn't exist in any public NLP model. Third, in risk scoring: I designed the periodontal risk model based on clinical parameters I actually used in practice — probing depths, bleeding on probing, attachment loss."

---

## 20. Definition of Done — Production Checklist

Use this checklist to verify ClinIQ is truly "production-grade" before putting it on your resume:

### Code Quality
- [ ] Type hints throughout Python codebase (mypy passes with strict mode)
- [ ] TypeScript strict mode in frontend
- [ ] Pre-commit hooks running (black, ruff, isort, mypy, eslint)
- [ ] No TODO comments in main branch (moved to GitHub issues)
- [ ] Consistent code style enforced by linters

### Testing
- [ ] Backend test coverage ≥ 80%
- [ ] Frontend test coverage ≥ 70%
- [ ] ML smoke tests pass in CI
- [ ] Integration tests cover all API endpoints
- [ ] Load tests documented with results

### Documentation
- [ ] README with quickstart, architecture diagram, and screenshots
- [ ] Model cards for all ML models
- [ ] API documentation with examples
- [ ] Deployment guide
- [ ] Security architecture documented
- [ ] CONTRIBUTING.md with setup instructions

### ML
- [ ] All models have documented evaluation metrics
- [ ] Comparison against baselines documented
- [ ] Clinical error analysis written
- [ ] Experiment tracking shows full history
- [ ] Models versioned in registry

### Infrastructure
- [ ] Docker Compose runs full stack with one command
- [ ] CI/CD pipeline green on main branch
- [ ] Health checks on all services
- [ ] Monitoring dashboards configured
- [ ] Structured logging implemented
- [ ] Secrets management (no hardcoded credentials)

### Security
- [ ] Authentication on all endpoints
- [ ] Rate limiting configured
- [ ] Input validation on all user-facing endpoints
- [ ] SQL injection prevention (parameterized queries via ORM)
- [ ] XSS prevention in frontend
- [ ] CORS configured properly
- [ ] Audit logging active

### Demo
- [ ] Live instance deployed and accessible
- [ ] Demo data loaded (synthetic)
- [ ] Walkthrough video recorded
- [ ] Custom domain with SSL

---

## Final Words

This is an ambitious plan. It's designed to be ambitious because half-measures don't change careers. But remember:

- **A finished 80% of this plan is still an extraordinary project.** If you complete through Phase 3 and skip the advanced features in Phase 4–5, you still have a production-grade clinical NLP platform that will get you interviews.
- **The dental NLP module is your secret weapon.** Prioritize it even if you cut other advanced features. It's the thing nobody else has.
- **Document everything as you go.** Don't save documentation for Month 14. Write READMEs as you build. Record decisions as you make them. Future-you will thank present-you.
- **Your clinical expertise makes every part of this project better.** Don't just build the ML — interpret it, validate it, explain it through your clinical lens. That's what transforms this from "another NLP project" into "a clinical intelligence platform built by someone who understands healthcare."

Now go build it.

---

*Roadmap created: March 2026*
*Target completion: July 2027*
*Review cadence: Monthly (first Sunday of each month)*
