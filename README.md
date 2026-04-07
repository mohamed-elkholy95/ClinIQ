# ClinIQ

> AI-powered clinical NLP platform for extracting structured intelligence from unstructured medical text.

[![CI/CD](https://github.com/mohamed-elkholy95/ClinIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamed-elkholy95/ClinIQ/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-3062%20backend%20%7C%20596%20frontend-brightgreen)](backend/tests/)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](backend/tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ClinIQ is a full-stack healthcare AI portfolio project built to demonstrate production-minded engineering around clinical natural language processing. It combines a FastAPI backend, a React frontend, a Python SDK, ML-oriented pipeline modules, observability, deployment assets, and security-aware documentation in a single public repository.

## Why this repo stands out

- End-to-end product shape: API, UI, SDK, docs, infra, and tests live in one coherent codebase.
- Broad clinical NLP coverage: entity extraction, ICD coding, summarization, risk scoring, de-identification, search, temporal reasoning, and more.
- Production-minded design: auth, rate limiting, monitoring, async workers, Docker, Kubernetes, and deployment guides.
- Public-repo discipline: explicit rules for secrets, PHI safety, and documentation quality.
- Portfolio-ready depth: architecture notes, model cards, tests, and structured contributor guidance.

## Feature Highlights

ClinIQ exposes 31 API endpoint groups across the clinical NLP workflow.

| Area | Capabilities |
|------|--------------|
| Core analysis | full pipeline orchestration, streaming analysis, enhanced multi-module analysis |
| Extraction | NER, medications, allergies, vitals, sections, abbreviations, assertions, temporal extraction |
| Clinical intelligence | ICD-10 prediction, concept normalization, relation extraction, quality scoring, SDoH, comorbidity |
| Safety and privacy | PHI de-identification, audit-aware architecture, public-repo secret restrictions |
| Search and workflow | hybrid search, batch processing, conversation memory, model management |
| Platform | JWT auth, API keys, Redis caching, Celery workers, Prometheus, Grafana |

## Architecture Snapshot

```text
Clients              Gateway                Application                Data + Ops
-------              -------                -----------                ----------
React SPA   --->     Nginx / Ingress  --->  FastAPI + Celery    --->   PostgreSQL
Python SDK           TLS + rate limit       ML pipeline modules        Redis
REST / cURL          auth boundary          API schemas + services     MinIO / MLflow
                                                                      Prometheus + Grafana
```

More detail: [Architecture](docs/architecture.md)

## Technical Scope

- Backend: FastAPI, Python 3.11+, Pydantic v2, SQLAlchemy 2, Celery
- Frontend: React 18, TypeScript, Vite, TanStack Query
- ML: PyTorch, HuggingFace Transformers, scikit-learn, scispaCy
- Data and infra: PostgreSQL, Redis, MinIO, MLflow, Docker Compose, Kubernetes
- Quality: pytest, Vitest, Ruff, mypy, Bandit, Trivy, GitHub Actions

## Quick Start

### Prerequisites

- Conda
- Python 3.11+
- Node.js 20+
- Docker + Docker Compose v2+

### Local setup

Use the Conda environment named `dev` for all Python commands in this repository.

```bash
conda activate dev
cp backend/.env.example backend/.env

docker compose up -d postgres redis minio

cd backend
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Useful local URLs:

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

## Example API Surface

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Full pipeline analysis |
| `POST` | `/api/v1/analyze/stream` | Server-sent event streaming pipeline |
| `POST` | `/api/v1/ner` | Named entity recognition |
| `POST` | `/api/v1/icd-predict` | ICD-10 code prediction |
| `POST` | `/api/v1/summarize` | Clinical summarization |
| `POST` | `/api/v1/risk-score` | Risk scoring |
| `POST` | `/api/v1/deidentify` | PHI de-identification |
| `POST` | `/api/v1/search` | Hybrid clinical search |
| `GET` | `/api/v1/health` | Health and readiness checks |

Full reference: [API Reference](docs/api/api-reference.md)

## Project Structure

```text
ClinIQ/
|-- backend/      FastAPI app, ML modules, middleware, DB, tests
|-- frontend/     React application and UI tests
|-- sdk/          Python client SDK
|-- docs/         Architecture, deployment, security, API, model cards
|-- infra/        Kubernetes, Nginx, Prometheus, Grafana
|-- scripts/      Utility scripts
|-- AGENTS.md     Repo operating rules for contributors and AI agents
```

## Documentation Map

- [Documentation Home](docs/README.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api/api-reference.md)
- [Local Setup](docs/deployment/local-setup.md)
- [Production Guide](docs/deployment/production-guide.md)
- [HIPAA Compliance Architecture](docs/security/hipaa-compliance.md)
- [Model Cards](docs/ml/)
- [Contributing Guide](CONTRIBUTING.md)
- [Agent and Repo Rules](AGENTS.md)

## Security and Public Repo Notes

This is a public GitHub repository.

- Never commit secrets, `.env` files, real credentials, or production-only configuration values.
- Never commit PHI or realistic patient data.
- Use synthetic or clearly de-identified examples only.
- Follow the rules in [AGENTS.md](AGENTS.md) and the security guidance in [docs/security/hipaa-compliance.md](docs/security/hipaa-compliance.md).

## Development Commands

```bash
conda activate dev
make test
make test-unit
make test-integration
make test-ml
make lint
make typecheck
cd frontend && npm run test
cd sdk && pytest tests/ -v
```

## Portfolio Framing

ClinIQ is designed to demonstrate:

- backend API design for ML-heavy systems
- frontend product thinking for technical workflows
- structured documentation and architecture communication
- deployment awareness across local, Docker, and Kubernetes setups
- security-minded engineering in a public healthcare-adjacent repository

## License

MIT - see [LICENSE](LICENSE).

