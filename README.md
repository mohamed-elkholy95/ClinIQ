# ClinIQ

**Clinical Intelligence & Query Platform** -- AI-powered clinical NLP that transforms unstructured medical text into structured, actionable intelligence.

[![CI/CD](https://github.com/cliniq/cliniq/actions/workflows/ci.yml/badge.svg)](https://github.com/cliniq/cliniq/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/cliniq/cliniq)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
                          ClinIQ Platform Architecture

  +-------------+       +-----------------+       +---------------------+
  |   Clients   |       |   API Gateway   |       |    ML Services      |
  |             |       |                 |       |                     |
  | React SPA   +------>+ Nginx / Ingress +------>+ NER (scispaCy /     |
  | Python SDK  |       | Rate Limiting   |       |      BioBERT)       |
  | REST / cURL |       | JWT + API Key   |       | ICD-10 Prediction   |
  +-------------+       +--------+--------+       | Summarization       |
                                 |                | Risk Scoring        |
                                 v                | Dental NLP          |
                        +--------+--------+       +----------+----------+
                        |  FastAPI (v1)   |                  |
                        |  Celery Workers |                  |
                        +--------+--------+                  |
                                 |                           |
              +------------------+---------------------------+---+
              |                  |                               |
              v                  v                               v
     +--------+------+  +-------+--------+  +---------+---------+--+
     |  PostgreSQL   |  |     Redis      |  |  MinIO    | MLflow   |
     |  (Audit, PHI) |  | (Cache, Queue) |  | (Objects) | (Track)  |
     +---------------+  +----------------+  +----------+----------+
              |                  |                       |
              +------------------+-----------------------+
                                 |
                    +------------+------------+
                    |     Observability       |
                    |  Prometheus + Grafana   |
                    +-------------------------+
```

---

## Quick Start

```bash
git clone https://github.com/cliniq/cliniq.git && cd cliniq
docker compose up -d
open http://localhost:8000/docs
```

The API documentation is available at `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc). The React dashboard runs at `http://localhost:5173`.

---

## Features

| Feature | Description |
|---------|-------------|
| **Medical Entity Recognition (NER)** | Extract diseases, symptoms, medications, dosages, procedures, lab values, and temporal expressions from clinical text. Supports rule-based, scispaCy, and transformer (BioBERT) backends with negation and uncertainty detection. |
| **ICD-10 Code Prediction** | Predict ICD-10 diagnosis codes from clinical narratives using scikit-learn baselines, transformer classifiers (ClinicalBERT), and hierarchical chapter-then-code models. |
| **Clinical Text Summarization** | Generate concise summaries of clinical notes via extractive (TextRank with clinical relevance weighting) or abstractive (BART/T5) methods. Three detail levels: brief, standard, detailed. |
| **Risk Scoring** | Assess medication risk, diagnostic complexity, and follow-up urgency. Rule-based scorer with weighted polypharmacy, drug interaction, and ICD chapter analysis. Generates actionable clinical recommendations. |
| **Dental NLP Module** | Specialized NER for dental records: tooth numbering (Universal, FDI, Palmer), surface identification, periodontal measurements, CDT code prediction, and periodontal risk assessment. |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, Python 3.11, Pydantic v2, Celery, SQLAlchemy 2.0, Alembic |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS |
| **ML** | PyTorch, HuggingFace Transformers, scikit-learn, scispaCy, SHAP |
| **Data** | PostgreSQL 16, Redis 7, MinIO (S3-compatible), MLflow |
| **Infrastructure** | Docker Compose, Kubernetes, Nginx, Prometheus, Grafana |
| **CI/CD** | GitHub Actions, Trivy, Bandit, Ruff, mypy |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Full pipeline: NER + ICD + Summary + Risk |
| `POST` | `/api/v1/ner` | Named entity recognition only |
| `POST` | `/api/v1/icd-predict` | ICD-10 code prediction only |
| `POST` | `/api/v1/summarize` | Clinical text summarization only |
| `POST` | `/api/v1/risk-score` | Risk scoring only |
| `POST` | `/api/v1/batch` | Async batch processing |
| `GET`  | `/api/v1/batch/{job_id}` | Poll batch job status |
| `GET`  | `/api/v1/models` | List loaded models |
| `GET`  | `/api/v1/models/{name}` | Model details |
| `GET`  | `/api/v1/health` | Health check (liveness + readiness) |
| `POST` | `/api/v1/auth/token` | Obtain JWT access token |
| `POST` | `/api/v1/auth/register` | Register new user |
| `POST` | `/api/v1/auth/api-keys` | Generate API key |
| `GET`  | `/api/v1/auth/me` | Current user info |

See the full [API Reference](docs/api/api-reference.md) for request/response schemas and examples.

---

## Project Structure

```
ClinIQ/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── schemas/          # Pydantic request/response models
│   │   │   └── v1/routes/        # FastAPI route handlers
│   │   ├── core/                 # Config, security, exceptions
│   │   ├── db/                   # SQLAlchemy models, sessions, migrations
│   │   ├── middleware/           # Auth, rate limiting, logging
│   │   ├── ml/
│   │   │   ├── dental/           # Dental NLP module
│   │   │   ├── explainability/   # SHAP explainers
│   │   │   ├── icd/              # ICD-10 classifiers
│   │   │   ├── monitoring/       # Drift detection, metrics
│   │   │   ├── ner/              # Named entity recognition
│   │   │   ├── risk/             # Risk scoring
│   │   │   ├── summarization/    # Text summarization
│   │   │   └── utils/            # Preprocessing, features, metrics
│   │   ├── services/             # Business logic services
│   │   ├── main.py               # FastAPI application entry point
│   │   └── worker.py             # Celery worker
│   ├── tests/
│   │   ├── unit/                 # Unit tests
│   │   ├── integration/          # API integration tests
│   │   ├── ml/                   # ML model smoke tests
│   │   └── load/                 # Locust load tests
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/           # Reusable UI components
│   │   ├── pages/                # Dashboard, EntityViewer, ICDResults, etc.
│   │   ├── hooks/                # React hooks (useAnalysis, useDocuments)
│   │   ├── services/             # API client
│   │   └── types/                # TypeScript type definitions
│   └── vite.config.ts
├── infra/
│   ├── grafana/                  # Dashboard provisioning
│   ├── k8s/                      # Kubernetes manifests
│   ├── nginx/                    # Reverse proxy config
│   └── prometheus/               # Metrics collection config
├── docs/                         # Documentation (you are here)
├── docker-compose.yml            # Development environment
├── docker-compose.prod.yml       # Production environment
├── Makefile                      # Development commands
└── .github/workflows/ci.yml     # CI/CD pipeline
```

---

## Development Setup

### Prerequisites

- Docker and Docker Compose v2+
- Python 3.11+
- Node.js 20+ (for frontend)
- Make (optional, for convenience commands)

### Setup

```bash
# Clone the repository
git clone https://github.com/cliniq/cliniq.git
cd cliniq

# Copy environment configuration
cp backend/.env.example backend/.env

# Start infrastructure services
docker compose up -d postgres redis minio

# Install backend dependencies
cd backend
pip install -e ".[dev]"

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# (In a separate terminal) Start the frontend
cd frontend
npm install
npm run dev
```

Or use Make shortcuts:

```bash
make dev          # Start infrastructure + API server
make frontend     # Start frontend dev server (separate terminal)
make dev-all      # Start everything via Docker Compose
```

---

## Testing

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests (requires running services)
make test-ml           # ML model smoke tests
make test-cov          # Tests with HTML coverage report
make lint              # Ruff linter
make typecheck         # mypy type checking
make quality           # Lint + typecheck
make loadtest          # Locust load tests (interactive browser UI)
```

---

## Documentation

- [Architecture](docs/architecture.md) -- System design, data flows, and ADRs
- [API Reference](docs/api/api-reference.md) -- Endpoints, schemas, and examples
- [Model Cards](docs/ml/) -- NER, ICD-10, and summarization model documentation
- [Evaluation Report](docs/ml/evaluation-report.md) -- Model performance metrics
- [Local Setup](docs/deployment/local-setup.md) -- Developer environment guide
- [Production Guide](docs/deployment/production-guide.md) -- Deployment and operations
- [HIPAA Compliance](docs/security/hipaa-compliance.md) -- Security architecture

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
