# ClinIQ

Clinical NLP platform that extracts structured data from unstructured medical text.

[![CI/CD](https://github.com/cliniq/cliniq/actions/workflows/ci.yml/badge.svg)](https://github.com/cliniq/cliniq/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What it does

ClinIQ takes clinical notes and returns structured output: medical entities, ICD-10 codes, summaries, and risk scores. It supports rule-based, scispaCy, and transformer (BioBERT/ClinicalBERT) backends.

**Core capabilities:**

- **Named Entity Recognition** — diseases, symptoms, medications, dosages, procedures, lab values, temporal expressions. Supports negation and uncertainty detection.
- **ICD-10 Code Prediction** — scikit-learn baselines, transformer classifiers, and hierarchical chapter→code models.
- **Clinical Text Summarization** — extractive (TextRank) and abstractive (BART/T5) at three detail levels.
- **Risk Scoring** — medication risk, diagnostic complexity, follow-up urgency with actionable recommendations.
- **Dental NLP** — tooth numbering (Universal/FDI/Palmer), surface ID, periodontal measurements, CDT codes.
- **PHI De-identification** — HIPAA Safe Harbor compliant detection and redaction of all 18 identifier categories.

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

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Full pipeline: NER + ICD + Summary + Risk |
| `POST` | `/api/v1/ner` | Named entity recognition |
| `POST` | `/api/v1/icd-predict` | ICD-10 code prediction |
| `POST` | `/api/v1/summarize` | Clinical text summarization |
| `POST` | `/api/v1/risk-score` | Risk scoring |
| `POST` | `/api/v1/deidentify` | PHI de-identification |
| `POST` | `/api/v1/batch` | Async batch processing |
| `GET`  | `/api/v1/batch/{job_id}` | Batch job status |
| `GET`  | `/api/v1/health` | Health check |
| `POST` | `/api/v1/auth/token` | Obtain JWT token |

Full reference: [docs/api/api-reference.md](docs/api/api-reference.md)

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
│   │   ├── api/              # Routes, schemas, dependencies
│   │   ├── core/             # Config, security, exceptions
│   │   ├── db/               # SQLAlchemy models, migrations
│   │   ├── middleware/        # Auth, rate limiting, logging
│   │   ├── ml/               # NER, ICD, summarization, risk, dental, de-id
│   │   └── services/         # Business logic
│   └── tests/
├── frontend/                 # React SPA
├── sdk/                      # Python client SDK
├── infra/                    # K8s, Nginx, Prometheus configs
├── docs/                     # Architecture, API ref, model cards
└── docker-compose.yml
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
