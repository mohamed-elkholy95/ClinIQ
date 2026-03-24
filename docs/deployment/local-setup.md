# ClinIQ Local Development Setup

This guide walks through setting up a local development environment for the ClinIQ platform.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker | 24+ | Container runtime |
| Docker Compose | v2+ | Multi-container orchestration |
| Python | 3.11+ | Backend runtime |
| Node.js | 20+ | Frontend build toolchain |
| Make | Any | Convenience commands (optional) |
| Git | Any | Source control |

Verify your installations:

```bash
docker --version          # Docker version 24.x+
docker compose version    # Docker Compose version v2.x+
python3 --version         # Python 3.11+
node --version            # v20+
make --version            # GNU Make 4+
```

---

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/cliniq/cliniq.git
cd cliniq
```

### 2. Configure Environment Variables

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` as needed. The defaults are sufficient for local development. For production, see the [Production Guide](production-guide.md).

### 3. Start Infrastructure Services

Start PostgreSQL, Redis, and MinIO via Docker Compose:

```bash
docker compose up -d postgres redis minio
```

Verify services are healthy:

```bash
docker compose ps
```

Expected output shows all three services as `healthy`.

### 4. Install Backend Dependencies

```bash
cd backend
pip install -e ".[dev]"
```

This installs the backend package in editable mode with development dependencies (pytest, ruff, mypy, locust).

### 5. Run Database Migrations

```bash
cd backend
alembic upgrade head
```

This creates the required tables: users, api_keys, audit_log, analysis_results, and model_registry.

### 6. Start the API Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use Make:

```bash
make dev
```

The API is now available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
- Health check: http://localhost:8000/api/v1/health

### 7. Start the Frontend (Optional)

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The React dashboard is available at http://localhost:5173.

### 8. Start the Celery Worker (Optional)

Required for batch processing:

```bash
cd backend
celery -A app.worker worker --loglevel=info --concurrency=2
```

### 9. Start Monitoring (Optional)

For Prometheus + Grafana:

```bash
docker compose up -d prometheus grafana
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (default: admin/admin)

---

## Environment Variables Reference

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `ClinIQ` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `DEBUG` | `true` | Enable debug mode |
| `ENVIRONMENT` | `development` | Runtime environment: development, staging, production |

### API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_V1_PREFIX` | `/api/v1` | API version prefix |
| `DOCS_URL` | `/docs` | Swagger UI path (null to disable) |
| `OPENAPI_URL` | `/openapi.json` | OpenAPI schema path (null to disable) |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://cliniq:cliniq@localhost:5432/cliniq` | Async PostgreSQL URL |
| `DATABASE_SYNC_URL` | `postgresql://cliniq:cliniq@localhost:5432/cliniq` | Sync URL for Alembic |
| `DB_POOL_SIZE` | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | `20` | Max overflow connections |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_CACHE_TTL` | `3600` | Cache TTL in seconds |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-me-in-production-...` | JWT signing key (change in prod) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiry |
| `API_KEY_HEADER` | `X-API-Key` | API key header name |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_PERIOD` | `86400` | Window size in seconds (24h) |

### ML Models

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models` | Model file directory |
| `MODEL_CACHE_SIZE` | `3` | Max models in memory |
| `INFERENCE_TIMEOUT` | `30` | Inference timeout in seconds |
| `MAX_DOCUMENT_LENGTH` | `100000` | Max input characters |
| `MAX_BATCH_SIZE` | `100` | Max documents per batch |

### MLflow

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `cliniq` | Default experiment name |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `cliniq` | Default bucket name |
| `MINIO_SECURE` | `false` | Use HTTPS for MinIO |

### Celery

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Task broker URL |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Result backend URL |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `LOG_FORMAT` | `json` | Log format: json or text |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `["http://localhost:3000","http://localhost:5173"]` | Allowed origins (JSON array) |

---

## Running the Application

### Full Stack (Docker Compose)

Start everything:

```bash
docker compose up -d            # All services
# or
make dev-all                    # Equivalent
```

Start only what you need:

```bash
docker compose up -d postgres redis     # Database + cache only
make dev                                # Infrastructure + local API
```

### Individual Components

```bash
# API server (with hot reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Celery worker
celery -A app.worker worker --loglevel=info --concurrency=2

# Frontend dev server
cd frontend && npm run dev

# MLflow tracking server (if not using Docker)
mlflow server --backend-store-uri postgresql://cliniq:cliniq@localhost:5432/mlflow \
              --default-artifact-root file:///tmp/mlflow/artifacts \
              --host 0.0.0.0 --port 5000
```

---

## Running Tests

```bash
# All tests
make test
# or: cd backend && python -m pytest tests/ -v

# Unit tests only
make test-unit

# Integration tests (requires running infrastructure)
make test-integration

# ML model smoke tests
make test-ml

# Tests with coverage
make test-cov
# Coverage report: backend/htmlcov/index.html

# Code quality
make lint           # Ruff linter
make typecheck      # mypy type checking
make format         # Auto-format with Ruff
make quality        # Lint + typecheck

# Load testing (starts Locust web UI at http://localhost:8089)
make loadtest
```

---

## Troubleshooting

### Docker services fail to start

```bash
# Check logs
docker compose logs postgres
docker compose logs redis

# Reset volumes (destroys data)
docker compose down -v
docker compose up -d
```

### Database connection errors

```
sqlalchemy.exc.OperationalError: could not connect to server
```

Ensure PostgreSQL is running and accepting connections:

```bash
docker compose ps postgres    # Should show "healthy"
docker compose logs postgres  # Check for errors
```

If the port is in use, check for conflicting PostgreSQL instances:

```bash
sudo lsof -i :5432
```

### Redis connection errors

```
redis.exceptions.ConnectionError: Error connecting to localhost:6379
```

```bash
docker compose ps redis
docker compose logs redis
```

### Import errors / missing dependencies

```bash
cd backend
pip install -e ".[dev]"    # Reinstall with dev deps
```

### Alembic migration errors

```bash
# Check current migration state
cd backend && alembic current

# Reset migrations (destroys data)
cd backend && alembic downgrade base
cd backend && alembic upgrade head
```

### Port conflicts

Default ports and alternatives:

| Service | Default Port | Alternative |
|---------|-------------|-------------|
| API | 8000 | Set `--port` flag |
| Frontend | 5173 | Set `--port` in vite.config.ts |
| PostgreSQL | 5432 | Change in docker-compose.yml |
| Redis | 6379 | Change in docker-compose.yml |
| MinIO | 9000/9001 | Change in docker-compose.yml |
| MLflow | 5000 | Change `--port` flag |
| Prometheus | 9090 | Change in docker-compose.yml |
| Grafana | 3001 | Change in docker-compose.yml |

### Model loading errors

If ML models fail to load:

1. Check that model files exist in the `MODEL_DIR` directory
2. Verify sufficient memory (some transformer models require 2-4 GB)
3. For scispaCy models, install separately: `pip install scispacy` and download the model: `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz`
4. For GPU-enabled inference, ensure PyTorch CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
