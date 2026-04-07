# ClinIQ Local Development Setup

This guide covers a clean local setup for the ClinIQ platform and keeps the workflow aligned with the rest of the repository documentation.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Conda | Current | Python environment management |
| Python | 3.11+ | Backend runtime |
| Node.js | 20+ | Frontend build toolchain |
| Docker | 24+ | Container runtime |
| Docker Compose | v2+ | Multi-container orchestration |
| Git | Current | Source control |
| Make | Optional | Convenience commands |

Verify your environment:

```bash
docker --version
docker compose version
python --version
node --version
```

## Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/mohamed-elkholy95/ClinIQ.git
cd ClinIQ
```

### 2. Activate the required Python environment

All Python commands in this repository should run from the Conda environment named `dev`.

```bash
conda activate dev
```

### 3. Configure environment variables

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` only for local development needs. Keep real credentials out of version control.

### 4. Start infrastructure services

```bash
docker compose up -d postgres redis minio
```

Verify service health:

```bash
docker compose ps
```

### 5. Install backend dependencies

```bash
conda activate dev
cd backend
pip install -e ".[dev]"
```

### 6. Run database migrations

```bash
conda activate dev
cd backend
alembic upgrade head
```

Optional seed data:

```bash
conda activate dev
cd backend
python -m app.db.seed
```

### 7. Start the API server

```bash
conda activate dev
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use Make:

```bash
conda activate dev
make dev
```

API endpoints:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Health check: `http://localhost:8000/api/v1/health`

### 8. Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

- App: `http://localhost:5173`

### 9. Start the Celery worker

Required for async and batch workflows:

```bash
conda activate dev
cd backend
celery -A app.worker worker --loglevel=info --concurrency=2
```

### 10. Start monitoring services

Optional local monitoring stack:

```bash
docker compose up -d prometheus grafana
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

## Running Tests

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

## Troubleshooting

### Backend dependency issues

```bash
conda activate dev
cd backend
pip install -e ".[dev]"
```

### Database connection errors

```bash
docker compose ps postgres
docker compose logs postgres
```

### Redis connection errors

```bash
docker compose ps redis
docker compose logs redis
```

### Migration issues

```bash
conda activate dev
cd backend
alembic current
alembic upgrade head
```

### Port conflicts

Default local ports:

| Service | Port |
|---------|------|
| API | 8000 |
| Frontend | 5173 |
| PostgreSQL | 5432 |
| Redis | 6379 |
| MinIO | 9000 / 9001 |
| MLflow | 5000 |
| Prometheus | 9090 |
| Grafana | 3001 |

## Public Repo Reminder

This repository is public.

- do not commit `.env` files or real credentials
- do not use PHI or realistic patient data in examples
- use synthetic content and placeholders only
