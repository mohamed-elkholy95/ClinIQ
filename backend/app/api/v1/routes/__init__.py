"""API v1 router registry.

Imports every sub-router and mounts it onto a single ``api_router`` that is
registered with the FastAPI application in ``app/main.py``:

    app.include_router(api_router, prefix=settings.api_v1_prefix)

All sub-routers are included without an additional prefix here; each module
declares its own path strings (e.g. ``/health``, ``/analyze``, ``/ner``).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes.analyze import router as analyze_router
from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.deidentify import router as deidentify_router
from app.api.v1.routes.batch import router as batch_router
from app.api.v1.routes.drift import router as drift_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.icd import router as icd_router
from app.api.v1.routes.metrics import router as metrics_router
from app.api.v1.routes.models import router as models_router
from app.api.v1.routes.ner import router as ner_router
from app.api.v1.routes.risk import router as risk_router
from app.api.v1.routes.search import router as search_router
from app.api.v1.routes.stream import router as stream_router
from app.api.v1.routes.summarize import router as summarize_router

# ---------------------------------------------------------------------------
# Build the combined v1 router
# ---------------------------------------------------------------------------

api_router = APIRouter()

# Infrastructure / meta endpoints
api_router.include_router(health_router)   # GET  /health, /health/live, /health/ready
api_router.include_router(metrics_router)  # GET  /metrics, /metrics/models
api_router.include_router(drift_router)    # GET  /drift/status, POST /drift/record

# Core NLP inference endpoints
api_router.include_router(analyze_router)  # POST /analyze
api_router.include_router(stream_router)   # POST /analyze/stream  (SSE)
api_router.include_router(ner_router)      # POST /ner
api_router.include_router(icd_router)      # POST /icd-predict, GET /icd-codes/{code}
api_router.include_router(summarize_router)  # POST /summarize
api_router.include_router(risk_router)     # POST /risk-score

# PHI de-identification
api_router.include_router(deidentify_router)  # POST /deidentify, POST /deidentify/batch

# Async batch processing
api_router.include_router(batch_router)    # POST /batch, GET /batch/{job_id}

# Document search
api_router.include_router(search_router)   # POST /search, POST /search/reindex

# Model registry
api_router.include_router(models_router)   # GET  /models, GET /models/{model_name}

# Authentication & user management
api_router.include_router(auth_router)     # POST /auth/token, /auth/register, /auth/api-keys
                                            # GET  /auth/me

__all__ = ["api_router"]
