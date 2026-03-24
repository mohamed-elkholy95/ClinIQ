"""Health check and model management routes."""

import logging
import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.v1.deps import PipelineDep, SuperUser
from app.api.v1.schemas import HealthResponse, ModelsListResponse, ModelInfo
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Track startup time
_startup_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and model loading state.",
)
async def health_check(
    pipeline: PipelineDep,
) -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()

    models_loaded = {
        "ner": pipeline._ner_model is not None and pipeline._ner_model.is_loaded,
        "icd": pipeline._icd_model is not None and pipeline._icd_model.is_loaded,
        "summarizer": pipeline._summarizer is not None and pipeline._summarizer.is_loaded,
        "risk": pipeline._risk_scorer is not None,
    }

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models_loaded=models_loaded,
        uptime_seconds=time.time() - _startup_time,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint.",
)
async def liveness() -> dict:
    """Liveness probe - is the service running?"""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint.",
)
async def readiness(
    pipeline: PipelineDep,
) -> dict:
    """Readiness probe - is the service ready to accept requests?"""
    # Check if essential models are loaded
    pipeline_loaded = pipeline.is_loaded()

    if not pipeline_loaded:
        # Try to load
        try:
            pipeline.load()
            pipeline_loaded = True
        except Exception as e:
            logger.error(f"Pipeline not ready: {e}")
            return {"status": "not_ready", "reason": "Models not loaded"}, 503

    return {
        "status": "ready",
        "models": {
            "ner": pipeline._ner_model is not None,
            "icd": pipeline._icd_model is not None,
            "summarizer": pipeline._summarizer is not None,
            "risk": pipeline._risk_scorer is not None,
        },
    }


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available models",
    description="Get information about available ML models.",
)
async def list_models(
    pipeline: PipelineDep,
    superuser: SuperUser,
) -> ModelsListResponse:
    """List available models and their versions."""
    models = []

    # NER model
    if pipeline._ner_model:
        models.append(
            ModelInfo(
                name="ner",
                version=pipeline._ner_model.version,
                stage="production",
                is_loaded=pipeline._ner_model.is_loaded,
                metrics=None,
                deployed_at=None,
            )
        )

    # ICD model
    if pipeline._icd_model:
        models.append(
            ModelInfo(
                name="icd_classifier",
                version=pipeline._icd_model.version,
                stage="production",
                is_loaded=pipeline._icd_model.is_loaded,
                metrics=None,
                deployed_at=None,
            )
        )

    # Summarizer
    if pipeline._summarizer:
        models.append(
            ModelInfo(
                name="summarizer",
                version=pipeline._summarizer.version,
                stage="production",
                is_loaded=pipeline._summarizer.is_loaded,
                metrics=None,
                deployed_at=None,
            )
        )

    # Risk scorer
    if pipeline._risk_scorer:
        models.append(
            ModelInfo(
                name="risk_scorer",
                version=pipeline._risk_scorer.version,
                stage="production",
                is_loaded=True,
                metrics=None,
                deployed_at=None,
            )
        )

    # Default versions
    default_versions = {m.name: m.version for m in models}

    return ModelsListResponse(
        models=models,
        default_versions=default_versions,
    )


@router.post(
    "/models/reload",
    summary="Reload models",
    description="Force reload of all ML models.",
)
async def reload_models(
    superuser: SuperUser,
    pipeline: PipelineDep,
) -> dict:
    """Reload all models (superuser only)."""
    try:
        pipeline.load()
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
