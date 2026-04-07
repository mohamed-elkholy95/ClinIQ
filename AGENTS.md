# AGENTS.md

This document defines the operating rules for human contributors and AI agents working in the ClinIQ repository.

ClinIQ is a public GitHub repository. Treat everything in this repository as public unless a file is explicitly ignored and intended for local-only use. Do not place secrets, credentials, protected health information (PHI), proprietary datasets, or internal-only operational details in tracked files, pull requests, issues, discussions, screenshots, or generated artifacts.

## Purpose

Use this file as the repo-level contract for:

- how to navigate the codebase
- how to make safe changes
- what tests to run
- what documentation to update
- what must never be committed to Git
- what restrictions apply because this project processes clinical text
- what local environment to use for backend work

This file complements, but does not replace:

- `README.md`
- `CONTRIBUTING.md`
- `docs/security/hipaa-compliance.md`
- `docs/deployment/production-guide.md`

## Required Local Environment

For backend development in this repository, always use the Conda environment named `dev`.

- Activate it before running Python tooling: `conda activate dev`
- Install backend dependencies inside that environment
- Run tests, linters, migrations, scripts, and local backend servers from that environment
- Do not create or rely on repo-local virtualenvs such as `.venv/` for normal project work unless a maintainer explicitly asks for it

Frontend and SDK work can use their standard package tooling, but any Python command in this repository should assume `conda activate dev` first.

## Repository Overview

Top-level areas:

- `backend/`: FastAPI application, ML modules, middleware, database models, migrations, Celery worker, tests
- `frontend/`: React + TypeScript + Vite application with page-level clinical tools and Vitest coverage
- `sdk/`: Python client SDK for API consumers
- `docs/`: architecture, API reference, deployment guides, security docs, model cards
- `infra/`: Kubernetes manifests, Nginx, Prometheus, Grafana provisioning
- `scripts/`: utility scripts such as synthetic data generation

Primary entry points:

- `backend/app/main.py`: FastAPI application bootstrap
- `backend/app/api/v1/routes/`: API route modules
- `backend/app/api/schemas/`: request and response contracts
- `backend/app/ml/`: NLP and ML modules
- `backend/app/services/`: service-layer orchestration
- `frontend/src/App.tsx`: frontend route composition
- `frontend/src/pages/`: user-facing feature pages
- `frontend/src/services/`: API client logic

## Core Principles

- Prefer small, focused changes over broad rewrites.
- Preserve public API behavior unless the task explicitly requires a breaking change.
- Keep backend schemas, route behavior, frontend types, tests, and docs in sync.
- Use synthetic, de-identified, or clearly fake clinical content only.
- Never weaken security controls for convenience.
- Never commit secrets, PHI, tokens, private certificates, or environment files.

## Public Repo Restrictions

Because this repository is public:

- Never commit `.env` files, secret overrides, kube secrets with real values, cloud credentials, SSH keys, API keys, session tokens, or database dumps.
- Never paste real patient notes, names, dates of birth, MRNs, phone numbers, addresses, insurance IDs, or any other PHI into source, tests, fixtures, examples, screenshots, or documentation.
- Never commit production URLs that expose private infrastructure, internal IPs, VPN-only hostnames, or internal dashboards unless they are already intentionally public.
- Never include real incident details, customer identifiers, or internal access procedures in docs.
- Never store plaintext secrets in code comments, sample curl commands, CI definitions, Docker Compose files, Kubernetes manifests, or markdown.

Allowed patterns:

- placeholder values such as `changeme_in_production`
- example domains such as `cliniq.example.com`
- synthetic patient examples with obviously fake identifiers
- references to local setup via `.env.example`

## Secrets Handling Rules

Secrets must stay outside version control.

- Use `.env.example` files for placeholders only.
- Keep real local values in ignored files such as `.env`.
- For Kubernetes, keep committed manifests value-free and inject real values at deploy time.
- If a secret must be referenced in docs, describe where it comes from, not the value itself.
- If a secret is discovered in the repo, rotate it immediately and remove it from history using the project maintainers' preferred incident process.

Do not rely on obscurity:

- base64-encoded values are still secrets
- partial tokens are still sensitive if reusable
- screenshots of terminals, dashboards, or cloud consoles can leak secrets

## PHI and Clinical Data Rules

ClinIQ handles clinical NLP workflows, so data discipline is mandatory.

- Use only synthetic or de-identified text in tests, examples, benchmarks, and screenshots.
- Do not log raw clinical note text unless the task explicitly requires it and the destination is confirmed local-only and ignored.
- Prefer hashes, counts, IDs, and derived metrics over raw text retention.
- Do not add fixtures that resemble real records closely enough to be mistaken for source data.
- Review `docs/security/hipaa-compliance.md` before changing data flow, logging, persistence, or auth behavior.

## Change Workflow

When making changes:

1. Read the relevant module and adjacent tests before editing.
2. Check whether the change affects backend, frontend, SDK, docs, or infra contracts.
3. Implement the smallest coherent change that solves the problem.
4. Run targeted tests first, then broader validation if the change spans subsystems.
5. Update documentation when behavior, setup, or public interfaces change.

Avoid:

- unrelated refactors in the same change
- drive-by formatting churn
- renaming files or symbols without a concrete reason
- speculative architecture changes not required by the task

## Backend Guidance

Backend stack:

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2
- Celery
- Redis
- PostgreSQL

Backend conventions:

- Always start with `conda activate dev` before any Python command.
- Add type hints to production Python code.
- Use Pydantic schemas for request and response boundaries.
- Keep route logic thin when service-layer extraction is practical.
- Use structured logging, not ad hoc prints.
- Respect existing validation and exception patterns in `backend/app/core/`.
- Keep security-sensitive changes aligned with auth, rate-limit, sanitize, and logging middleware.

Backend commands:

```bash
conda activate dev
cd backend
pip install -e ".[dev]"
python -m pytest tests/ -v
ruff check app/ tests/
ruff format app/ tests/
mypy app/
alembic upgrade head
```

Common backend entry points:

- `backend/app/api/v1/routes/`
- `backend/app/api/schemas/`
- `backend/app/services/`
- `backend/app/ml/`
- `backend/app/core/config.py`
- `backend/app/core/security.py`
- `backend/app/middleware/`

## Frontend Guidance

Frontend stack:

- React 18
- TypeScript
- Vite
- Vitest
- TanStack Query

Frontend conventions:

- Keep types explicit and aligned with backend response shapes.
- Add loading, error, and empty states for async flows.
- Preserve accessibility basics such as labels, focus behavior, and readable states.
- Keep page components in `frontend/src/pages/` and reusable UI in `frontend/src/components/`.
- Update service clients in `frontend/src/services/` when backend contracts change.

Frontend commands:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
npm run test
```

## SDK Guidance

The Python SDK lives in `sdk/`.

- Keep client behavior aligned with the published API.
- Update SDK models when request or response contracts change.
- Add or update SDK tests for changed client behavior.
- Update `sdk/README.md` when SDK usage changes.
- Use `conda activate dev` before running SDK Python commands.

SDK commands:

```bash
conda activate dev
cd sdk
pytest tests/ -v
```

## Infrastructure Guidance

Infrastructure files exist for local development and deployment examples, but this public repo must remain secret-free.

- Keep committed Kubernetes and Docker manifests free of real credentials.
- Use placeholders in `infra/k8s/secrets.yml` and deployment docs only.
- Never commit generated certs, kubeconfigs, or cloud-specific access files.
- If deployment behavior changes, update the matching docs in `docs/deployment/`.

Review these before infra changes:

- `docker-compose.yml`
- `docker-compose.prod.yml`
- `infra/k8s/`
- `infra/nginx/nginx.conf`
- `docs/deployment/production-guide.md`

## Documentation Update Rules

Update docs whenever a change affects:

- setup steps
- environment variables
- API routes or payloads
- security behavior
- deployment flow
- model capabilities or limitations
- user-visible frontend workflows

Typical documentation targets:

- `README.md`
- `CONTRIBUTING.md`
- `docs/api/api-reference.md`
- `docs/architecture.md`
- `docs/deployment/local-setup.md`
- `docs/deployment/production-guide.md`
- `docs/security/hipaa-compliance.md`
- `docs/ml/model-card-*.md`

## Testing Expectations

Minimum expectation:

- run the tests directly related to the changed code
- run lint or type checks when changing Python or TypeScript logic
- note any unrun validation in the final handoff if full verification is not feasible
- use `conda activate dev` before Python-based test commands

Useful repo commands:

```bash
conda activate dev
make test
make test-unit
make test-integration
make test-ml
make test-cov
make lint
make format
make typecheck
cd frontend && npm run test
cd sdk && pytest tests/ -v
```

Prefer targeted test execution for fast feedback, then broaden coverage when the change touches shared contracts or core flows.

## Files That Need Extra Care

Treat these areas as high-risk:

- `backend/app/core/security.py`
- `backend/app/core/config.py`
- `backend/app/middleware/`
- `backend/app/db/`
- `backend/app/api/v1/routes/auth.py`
- `infra/k8s/secrets.yml`
- `docker-compose.prod.yml`
- `docs/security/hipaa-compliance.md`

Changes here can impact authentication, secret handling, persistence, deployment safety, compliance posture, or operational security.

## Review Checklist

Before opening or merging a change, verify:

- no secrets were added
- no PHI or realistic patient data was added
- changed code has relevant test coverage
- docs were updated if behavior changed
- placeholder values remain placeholders
- logs and error messages do not expose sensitive data
- public examples use synthetic content only
- Python commands were run from `conda activate dev`

## Incident Rule

If you encounter committed secrets, PHI, or other sensitive material:

1. stop normal editing
2. avoid copying the sensitive value into new files or comments
3. notify maintainers through the appropriate private channel
4. rotate or revoke credentials if applicable
5. clean the repository using the maintainers' approved response process

## Agent-Specific Expectations

If you are an AI agent operating in this repository:

- read this file before making broad changes
- prefer precise edits over repo-wide rewrites
- explain assumptions when context is ambiguous
- do not fabricate test results
- do not create or expose secrets while trying to make a feature work
- do not commit generated credentials or example values that look production-ready
- use `conda activate dev` before any Python tooling
- keep outputs suitable for a public repository and public pull request history

## Final Note

Security and privacy constraints are part of the definition of done in ClinIQ. A change that works technically but exposes secrets, PHI, or unsafe operational detail is not complete and must not be merged.
