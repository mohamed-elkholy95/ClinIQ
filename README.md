# ClinIQ

Clinical NLP platform that extracts structured data from unstructured medical text.

[![CI/CD](https://github.com/mohamed-elkholy95/ClinIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamed-elkholy95/ClinIQ/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-2806%20backend%20%7C%20238%20frontend-brightgreen)](backend/tests/)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](backend/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What it does

ClinIQ takes clinical notes and returns structured output: medical entities, ICD-10 codes, summaries, and risk scores. It supports rule-based, scispaCy, and transformer (BioBERT/ClinicalBERT) backends.

**29 API endpoint groups** covering the full clinical NLP pipeline:

- **Named Entity Recognition** — diseases, symptoms, medications, dosages, procedures, lab values, temporal expressions. Negation and uncertainty detection.
- **ICD-10 Code Prediction** — scikit-learn baselines, transformer classifiers, and hierarchical chapter→code models.
- **Clinical Summarization** — extractive (TextRank) and abstractive (BART/T5) at three detail levels.
- **Risk Scoring** — medication risk, diagnostic complexity, follow-up urgency with actionable recommendations.
- **Document Classification** — 13 clinical document types (discharge summary, progress note, operative note, etc.).
- **Medication Extraction** — drug names, dosages, routes, frequencies, durations, indications, PRN status. 220+ drug dictionary.
- **Allergy Extraction** — drug/food/environmental allergens, reaction severity, NKDA detection. ~150 allergen entries.
- **Vital Signs Extraction** — BP, HR, temp, RR, SpO2, weight, height, BMI, pain. Physiological range validation.
- **Section Parsing** — 35 clinical section categories with character-offset boundaries.
- **Abbreviation Expansion** — 220+ unambiguous + 10 ambiguous abbreviations across 12 clinical domains.
- **Assertion Detection** — ConText/NegEx-inspired present/absent/possible/conditional/hypothetical/family classification.
- **Concept Normalization** — entity linking to UMLS CUI, SNOMED-CT, RxNorm, ICD-10-CM, LOINC. ~140 curated concepts.
- **Relation Extraction** — 12 semantic relation types (treats, causes, contraindicates, etc.) between entity pairs.
- **Temporal Extraction** — dates, durations, frequencies (40+ abbreviations), relative time resolution, temporal relations.
- **PHI De-identification** — HIPAA Safe Harbor compliant detection and redaction of all 18 identifier categories.
- **Clinical Note Quality** — 5-dimension quality scoring (completeness, readability, structure, density, consistency).
- **SDoH Extraction** — 8 social determinant domains with ICD-10-CM Z-code mapping (Z55–Z65).
- **Comorbidity Scoring** — Charlson Comorbidity Index from ICD codes and free text with 10-year mortality estimation.
- **Dental NLP** — tooth numbering (Universal/FDI/Palmer), surface ID, periodontal measurements, CDT codes.
- **Hybrid Search** — BM25 + TF-IDF with medical query expansion, synonym matching, and clinical re-ranking.
- **Streaming Analysis** — Server-Sent Events for real-time stage-by-stage pipeline progress.

## Architecture

```
  Clients              API Gateway              ML Services
  ───────              ───────────              ───────────
  React SPA    ──►   Nginx / Ingress   ──►   NER (scispaCy/BioBERT)
  Python SDK         Rate Limiting            ICD-10 Prediction
  REST / cURL        JWT + API Key            Summarization
                                              Risk Scoring
                          │                   Dental NLP
                          ▼
                   FastAPI + Celery
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       PostgreSQL       Redis      MinIO / MLflow
       (Audit, PHI)   (Cache,     (Objects,
                       Queue)      Tracking)
                          │
                          ▼
                    Prometheus + Grafana
```

## Quick start

```bash
git clone https://github.com/cliniq/cliniq.git && cd cliniq
docker compose up -d
open http://localhost:8000/docs    # Swagger UI
open http://localhost:5173         # React dashboard
```

## API

**Core analysis:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Full pipeline: NER + ICD + Summary + Risk |
| `POST` | `/api/v1/analyze/stream` | SSE streaming pipeline progress |
| `POST` | `/api/v1/analyze/enhanced` | All 14 modules in one call |
| `POST` | `/api/v1/ner` | Named entity recognition |
| `POST` | `/api/v1/icd-predict` | ICD-10 code prediction |
| `POST` | `/api/v1/summarize` | Clinical text summarization |
| `POST` | `/api/v1/risk-score` | Risk scoring |

**Specialized extraction:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/medications` | Structured medication extraction |
| `POST` | `/api/v1/allergies` | Allergy extraction with severity |
| `POST` | `/api/v1/vitals` | Vital signs extraction |
| `POST` | `/api/v1/sections` | Clinical section parsing |
| `POST` | `/api/v1/abbreviations` | Abbreviation expansion |
| `POST` | `/api/v1/assertions` | Assertion status detection |
| `POST` | `/api/v1/normalize` | Concept normalization / entity linking |
| `POST` | `/api/v1/relations` | Semantic relation extraction |
| `POST` | `/api/v1/temporal` | Temporal information extraction |
| `POST` | `/api/v1/classify` | Document type classification |
| `POST` | `/api/v1/quality` | Clinical note quality analysis |
| `POST` | `/api/v1/sdoh` | Social determinants of health |
| `POST` | `/api/v1/comorbidity` | Charlson Comorbidity Index |
| `POST` | `/api/v1/deidentify` | PHI de-identification |

**Infrastructure:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Hybrid document search |
| `POST` | `/api/v1/batch` | Async batch processing |
| `GET`  | `/api/v1/health` | Health check + probes |
| `GET`  | `/api/v1/metrics` | Prometheus metrics |
| `GET`  | `/api/v1/drift/status` | Data/model drift monitoring |
| `POST` | `/api/v1/auth/token` | JWT authentication |

Most endpoints also support `/batch` variants and `GET` catalogues. Full reference: [docs/api/api-reference.md](docs/api/api-reference.md)

## Tech stack

| Layer | Technologies |
|-------|-------------|
| Backend | FastAPI, Python 3.11+, Pydantic v2, Celery, SQLAlchemy 2.0 |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| ML | PyTorch, HuggingFace Transformers, scikit-learn, scispaCy |
| Data | PostgreSQL 16, Redis 7, MinIO, MLflow |
| Infra | Docker Compose, Kubernetes, Nginx, Prometheus, Grafana |
| CI | GitHub Actions, Ruff, mypy, Trivy, Bandit |

## Development

### Prerequisites

- Docker + Docker Compose v2+
- Python 3.11+
- Node.js 20+ (frontend)

### Setup

```bash
cp backend/.env.example backend/.env

# Start infrastructure
docker compose up -d postgres redis minio

# Backend
cd backend
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install && npm run dev
```

### Testing

```bash
# Backend tests
cd backend
pytest tests/ -v                          # all tests
pytest tests/unit/ -v                     # unit only
pytest tests/ -v --cov=app --cov-report=term-missing  # with coverage

# SDK tests
cd sdk
pytest tests/ -v

# Linting
ruff check backend/
```

## Project structure

```
ClinIQ/
├── backend/
│   ├── app/
│   │   ├── api/v1/routes/    # 29 route modules (analyze, ner, icd, medications, etc.)
│   │   ├── api/schemas/      # Pydantic request/response models
│   │   ├── core/             # Config, security, exceptions
│   │   ├── db/               # SQLAlchemy models, Alembic migrations
│   │   ├── middleware/        # Auth, rate limiting, logging
│   │   ├── ml/               # 14+ NLP modules (NER, ICD, summarization, risk, dental,
│   │   │                     #   de-id, medications, allergies, vitals, sections,
│   │   │                     #   abbreviations, assertions, normalization, relations,
│   │   │                     #   temporal, classifier, quality, sdoh, comorbidity, search)
│   │   └── services/         # Business logic, model registry
│   └── tests/                # 2,806 tests (unit, integration, load)
├── frontend/                 # React 18 + TypeScript + Tailwind (238 tests)
├── sdk/                      # Python client SDK (40 tests)
├── infra/                    # K8s manifests, Nginx, Prometheus, Grafana
├── docs/                     # Architecture, API ref, 21 model cards, deployment
└── docker-compose.yml        # Dev + production compose files
```

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api/api-reference.md)
- [Model Cards](docs/ml/)
- [Local Setup](docs/deployment/local-setup.md)
- [Production Guide](docs/deployment/production-guide.md)
- [HIPAA Compliance](docs/security/hipaa-compliance.md)

## License

MIT — see [LICENSE](LICENSE).
