# Contributing to ClinIQ

Thank you for your interest in contributing to ClinIQ! This guide covers
everything you need to get started: local setup, coding standards, testing,
and the pull request process.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Project Structure](#project-structure)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Commit Conventions](#commit-conventions)
7. [Pull Request Process](#pull-request-process)
8. [ML Module Guidelines](#ml-module-guidelines)
9. [Frontend Guidelines](#frontend-guidelines)

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.11 | Backend runtime |
| Node.js | ≥ 18 | Frontend tooling |
| Docker & Docker Compose | Latest stable | Local services |
| Git | ≥ 2.30 | Version control |

Optional but recommended:

- **pre-commit** (`pip install pre-commit`) — automated code quality checks
- **ruff** (`pip install ruff`) — Python linting and formatting
- **mypy** (`pip install mypy`) — static type checking

---

## Local Development Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-org>/ClinIQ.git
cd ClinIQ

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd ../frontend
npm install

# Pre-commit hooks
cd ..
pre-commit install
```

### 2. Start infrastructure services

```bash
# Copy environment template and fill in secrets
cp backend/.env.example backend/.env

# Start PostgreSQL, Redis, and MinIO
docker compose up -d postgres redis minio
```

### 3. Run database migrations

```bash
cd backend
alembic upgrade head
python -m app.db.seed  # Load ICD-10 codes and demo data
```

### 4. Start development servers

```bash
# Terminal 1 — Backend
make dev

# Terminal 2 — Frontend
make frontend
```

The API is available at `http://localhost:8000/docs` and the frontend at
`http://localhost:5173`.

### 5. Verify your setup

```bash
make test          # All backend tests
cd frontend && npm test  # Frontend tests
```

---

## Project Structure

```
ClinIQ/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes, schemas, dependencies
│   │   ├── core/          # Config, security, exceptions
│   │   ├── db/            # SQLAlchemy models, migrations, session
│   │   ├── middleware/     # Auth, rate limiting, sanitisation
│   │   ├── ml/            # ML modules (NER, ICD, risk, dental, etc.)
│   │   └── services/      # Business logic layer
│   └── tests/             # Unit, integration, ML, and load tests
├── frontend/
│   └── src/
│       ├── components/    # Reusable UI components
│       ├── hooks/         # Custom React hooks
│       ├── pages/         # Route-level page components
│       ├── services/      # API client layer
│       └── types/         # TypeScript type definitions
├── sdk/                   # Python client SDK
├── docs/                  # Architecture, API reference, model cards
├── k8s/                   # Kubernetes manifests
├── ml/                    # Training data, experiments
└── docker-compose.yml     # Local development services
```

---

## Coding Standards

### Python Backend

- **Type hints** on every function signature — no exceptions.
- **Pydantic models** for all API request/response schemas.
- **Async functions** for I/O-bound operations (database, HTTP, file).
- **Dependency injection** via FastAPI's `Depends()`.
- **Structured logging** with the `logging` module — never `print()`.
- **Docstrings** on every public function and class with `Parameters`,
  `Returns`, and `Raises` sections.
- **Educational comments** explaining *why*, not just *what*.

```python
# Good
def extract_entities(text: str, model: str = "composite") -> list[Entity]:
    """Extract clinical entities from unstructured text.

    Uses a composite approach: rule-based patterns for high-precision
    extraction (medications, vitals) combined with transformer-based
    NER for general biomedical entities.

    Parameters
    ----------
    text : str
        Clinical note text (max 500,000 characters).
    model : str
        Model variant: "composite", "rule_based", or "transformer".

    Returns
    -------
    list[Entity]
        Extracted entities with types, spans, and confidence scores.

    Raises
    ------
    ValidationError
        If text is empty or exceeds length limits.
    InferenceError
        If the ML model fails during extraction.
    """
```

### TypeScript Frontend

- **Strict mode** enabled in `tsconfig.json`.
- **Functional components** with hooks — no class components.
- **Proper error boundaries** wrapping route-level components.
- **Loading states and skeleton UI** for all async data.
- **ARIA labels** and keyboard navigation for accessibility.
- **Mobile-first responsive design** with Tailwind CSS.

### Linting & Formatting

```bash
# Python
ruff check backend/app/ backend/tests/    # Lint
ruff format backend/app/ backend/tests/    # Format
mypy backend/app/                          # Type check

# Frontend
cd frontend && npm run lint
cd frontend && npm run format
```

Pre-commit hooks run these automatically on `git commit`.

---

## Testing

### Running Tests

```bash
# All backend tests
make test

# With coverage report
make test-cov

# Specific test categories
make test-unit
make test-integration
make test-ml

# Frontend tests
cd frontend && npm test

# SDK tests
cd sdk && python -m pytest tests/ -v
```

### Test Coverage Targets

| Component | Minimum | Current |
|-----------|---------|---------|
| Backend | 80% | 97% |
| Frontend | 70% | — |
| SDK | 80% | — |

### Writing Tests

- Every new module needs a corresponding test file.
- Every new API endpoint needs at least one happy-path and one error test.
- ML modules need smoke tests (does it run without crashing?) and
  correctness tests (are outputs reasonable?).
- Use fixtures from `conftest.py` — don't duplicate setup logic.

```python
# Good test structure
class TestMedicationExtractor:
    """Tests for medication extraction module."""

    def test_extracts_common_medications(self, sample_discharge_note):
        """Verify extraction of standard medication names."""
        result = extract_medications(sample_discharge_note)
        assert len(result) > 0
        assert any(m.name == "metformin" for m in result)

    def test_handles_empty_input(self):
        """Empty text should return empty list, not crash."""
        result = extract_medications("")
        assert result == []

    def test_rejects_oversized_input(self):
        """Text exceeding max length should raise ValidationError."""
        with pytest.raises(ValidationError):
            extract_medications("x" * 600_000)
```

---

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body explaining what/why/trade-offs]
```

### Types

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Build, CI, or tooling changes |
| `ci` | CI/CD pipeline changes |
| `style` | Formatting, whitespace (no logic change) |

### Rules

- One logical change per commit.
- Commit messages explain *what changed* and *why*.
- No AI attribution in commit messages (`Co-Authored-By`, etc.).
- No megacommits (500+ lines of unrelated changes).

```
# Good
feat(allergies): add severity classification for drug allergies

Classify extracted drug allergies into 4 severity levels
(life-threatening, severe, moderate, mild) using reaction keyword
matching and clinical heuristics. Chose keyword-based approach over
ML classification because severity labels are rarely present in
training data, making supervised learning impractical.

# Bad
update code
```

---

## Pull Request Process

1. **Branch from `main`**: `git checkout -b feat/your-feature`
2. **Keep PRs focused**: One feature or fix per PR.
3. **All tests must pass**: Backend, frontend, and SDK.
4. **No coverage regression**: New code should be tested.
5. **Documentation updated**: API docs, model cards, README as needed.
6. **Self-review first**: Check the diff before requesting review.

### PR Template

```markdown
## What

Brief description of what this PR does.

## Why

Context on why this change is needed.

## How

Key implementation details and design decisions.

## Testing

How was this tested? What test cases were added?

## Checklist

- [ ] Tests pass locally
- [ ] No linting errors
- [ ] Docstrings added for new public functions
- [ ] API docs updated (if new/changed endpoints)
- [ ] No hardcoded secrets
```

---

## ML Module Guidelines

### Adding a New ML Module

1. Create a directory under `backend/app/ml/<module_name>/`.
2. Implement the module with at minimum:
   - `__init__.py` with public API exports
   - A primary class or function with full docstrings
   - Type hints on all function signatures
3. Add a Pydantic schema in `backend/app/api/schemas/`.
4. Add a route in `backend/app/api/v1/routes/`.
5. Register the route in `backend/app/api/v1/routes/__init__.py`.
6. Write unit tests in `backend/tests/`.
7. Write a model card in `docs/ml/model-card-<module>.md`.
8. Update the API reference in `docs/api/api-reference.md`.
9. Add a frontend page (if user-facing) in `frontend/src/pages/`.

### Model Card Template

Every ML module must have a model card documenting:

- **Purpose**: What clinical task does it solve?
- **Architecture**: How does it work (rule-based, classical ML, transformer)?
- **Training Data**: What data was used (or would be used)?
- **Evaluation Metrics**: How is quality measured?
- **Limitations**: Known failure modes and edge cases.
- **Ethical Considerations**: Bias risks, fairness, safety.

See existing cards in `docs/ml/` for examples.

---

## Frontend Guidelines

### Adding a New Page

1. Create a page component in `frontend/src/pages/<PageName>.tsx`.
2. Add TypeScript interfaces in `frontend/src/types/clinical.ts`.
3. Add API functions in `frontend/src/services/clinical.ts`.
4. Register the route in `frontend/src/App.tsx`.
5. Add navigation in `frontend/src/components/Sidebar.tsx`.
6. Write tests in `frontend/src/__tests__/<PageName>.test.tsx`.

### Design Principles

- Every async operation shows a loading state (skeleton or spinner).
- Every API call has error handling with user-friendly messages.
- Include 2–3 preloaded sample inputs for demo purposes.
- Use colour coding consistently (green = good, red = critical, etc.).
- All interactive elements must be keyboard-accessible.

---

## Questions?

If something isn't covered here, check:

- `docs/architecture.md` — System design and data flow
- `docs/api/api-reference.md` — Complete API documentation
- `docs/deployment/` — Deployment and infrastructure guides
- `docs/security/` — HIPAA compliance and security architecture

Or open an issue with the `question` label.
