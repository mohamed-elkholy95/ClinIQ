# Contributing to ClinIQ

ClinIQ is a public AI portfolio repository built to look and behave like a serious software project. Contributions should improve the codebase without lowering its quality bar for security, clarity, testing, or presentation.

## What good contributions look like

- focused scope with a clear reason for the change
- consistent naming, structure, and documentation
- tests for new or changed behavior
- no secrets, PHI, or production-only values
- updates to docs when behavior, setup, or interfaces change

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Conda | Current | Python environment management |
| Python | 3.11+ | Backend runtime |
| Node.js | 20+ | Frontend tooling |
| Docker | Current | Local infrastructure |
| Git | 2.30+ | Version control |

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/mohamed-elkholy95/ClinIQ.git
cd ClinIQ
```

### 2. Use the required Conda environment

All Python work in this repository should run from the Conda environment named `dev`.

```bash
conda activate dev
```

Do not create a repo-local `.venv` for normal development unless a maintainer explicitly asks for it.

### 3. Install dependencies

```bash
cd backend
pip install -e ".[dev]"

cd ../frontend
npm install

cd ..
pre-commit install
```

### 4. Start local infrastructure

```bash
cp backend/.env.example backend/.env
docker compose up -d postgres redis minio
```

### 5. Prepare the database

```bash
conda activate dev
cd backend
alembic upgrade head
python -m app.db.seed
```

### 6. Run the application

Backend:

```bash
conda activate dev
make dev
```

Frontend:

```bash
make frontend
```

Local endpoints:

- API docs: `http://localhost:8000/docs`
- Frontend app: `http://localhost:5173`

## Project Layout

```text
backend/    API, ML modules, middleware, database, tests
frontend/   React UI, API clients, components, page tests
sdk/        Python SDK
infra/      Kubernetes manifests, Nginx, monitoring
docs/       Architecture, security, deployment, model cards
scripts/    Utility scripts
```

## Engineering Standards

### Backend

- add type hints to production Python code
- keep route handlers thin when service extraction is practical
- use Pydantic models for API boundaries
- use structured logging instead of `print`
- keep security-sensitive changes aligned with auth, logging, rate limit, and sanitization behavior

### Frontend

- keep TypeScript types aligned with backend contracts
- provide loading, error, and empty states for async experiences
- preserve accessibility basics for interactive UI
- keep route-level code in `frontend/src/pages/` and shared UI in `frontend/src/components/`

### Documentation

Update docs when a change affects:

- setup steps
- environment variables
- API routes or payloads
- security posture
- model behavior or limitations
- frontend workflows or navigation

## Testing Expectations

Run the narrowest useful tests first, then broaden when the change affects shared behavior.

```bash
conda activate dev
make test
make test-unit
make test-integration
make test-ml
make test-cov
make lint
make typecheck
cd frontend && npm run test
cd sdk && pytest tests/ -v
```

Minimum expectation:

- new logic has tests
- changed contracts have updated tests
- documentation-only changes are reviewed for accuracy
- final handoff states what was and was not validated

## Pull Request Guidance

Before opening a PR, verify:

- the change is focused
- tests relevant to the change pass locally
- docs are updated where needed
- no secrets or PHI were introduced
- examples and screenshots use synthetic data only

Preferred commit style:

```text
type(scope): short description
```

Examples:

- `feat(ner): add temporal entity post-processing`
- `fix(frontend): handle empty result state in risk dashboard`
- `docs(readme): improve portfolio-facing project overview`

## Public Repository Rules

ClinIQ is public. Treat that as a hard constraint.

- never commit `.env` files, tokens, passwords, or real credentials
- never commit PHI or realistic patient records
- never add internal-only infrastructure details that should remain private
- keep placeholders as placeholders in docs and manifests

See [AGENTS.md](AGENTS.md) for the full repo operating rules.

## Where to Look Next

- [README.md](README.md)
- [AGENTS.md](AGENTS.md)
- [docs/README.md](docs/README.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/security/hipaa-compliance.md](docs/security/hipaa-compliance.md)

