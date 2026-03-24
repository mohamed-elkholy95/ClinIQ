"""Model management endpoints.

Provides read-only introspection into the deployed ML models tracked in the
model_versions database table and their current runtime status.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelVersion
from app.db.session import get_db_session

router = APIRouter(tags=["models"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_MODEL_NAMES: list[str] = [
    "rule-based-ner",
    "spacy-ner",
    "transformer-ner",
    "sklearn-baseline-icd",
    "transformer-icd",
    "hierarchical-icd",
    "textrank",
    "abstractive-summarizer",
    "rule-based-risk",
    "logistic-risk",
]


def _model_version_to_dict(mv: ModelVersion) -> dict[str, Any]:
    """Serialize a ModelVersion ORM row to a plain dict."""
    return {
        "id": str(mv.id),
        "model_name": mv.model_name,
        "version": mv.version,
        "stage": mv.stage,
        "is_active": mv.is_active,
        "mlflow_run_id": mv.mlflow_run_id,
        "metrics": mv.metrics,
        "config": mv.config,
        "deployed_at": mv.deployed_at.isoformat() if mv.deployed_at else None,
        "deployed_by": str(mv.deployed_by) if mv.deployed_by else None,
        "created_at": mv.created_at.isoformat(),
        "updated_at": mv.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    summary="List available models",
    description=(
        "Return all ML model versions tracked in the model registry, ordered by "
        "model name and descending version. Each entry includes the stage "
        "('staging', 'production', 'archived'), active flag, MLflow run ID, "
        "evaluation metrics, and deployment metadata."
    ),
    responses={
        200: {"description": "Model list returned"},
    },
)
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    stage: str | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    """Return all tracked model versions, optionally filtered by stage and active flag.

    Query parameters:
    - **stage**: Filter by deployment stage ('staging', 'production', 'archived').
    - **active_only**: When True, return only active model versions.
    """
    stmt = select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.created_at.desc())

    if stage is not None:
        stmt = stmt.where(ModelVersion.stage == stage)
    if active_only:
        stmt = stmt.where(ModelVersion.is_active.is_(True))

    result = await db.execute(stmt)
    versions = result.scalars().all()

    # Build a nested structure grouped by model name.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for mv in versions:
        grouped.setdefault(mv.model_name, []).append(_model_version_to_dict(mv))

    return {
        "models": grouped,
        "total_versions": len(versions),
        "known_model_names": _KNOWN_MODEL_NAMES,
    }


@router.get(
    "/models/{model_name}",
    summary="Get model details",
    description=(
        "Return all tracked versions of a specific model, ordered by descending creation "
        "date. Includes evaluation metrics, deployment stage, and MLflow run linkage. "
        "Returns HTTP 404 if no versions of the requested model are found."
    ),
    responses={
        200: {"description": "Model details returned"},
        404: {"description": "Model not found in the registry"},
    },
)
async def get_model(
    model_name: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, Any]:
    """Return all tracked versions of the named model.

    Path parameters:
    - **model_name**: Model identifier (e.g. 'rule-based-ner', 'sklearn-baseline-icd').
    """
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.model_name == model_name)
        .order_by(ModelVersion.created_at.desc())
    )
    versions = result.scalars().all()

    if not versions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No model versions found for '{model_name}'. "
                f"Known model names: {_KNOWN_MODEL_NAMES}"
            ),
        )

    active_versions = [mv for mv in versions if mv.is_active]
    production_versions = [mv for mv in versions if mv.stage == "production"]

    return {
        "model_name": model_name,
        "versions": [_model_version_to_dict(mv) for mv in versions],
        "version_count": len(versions),
        "active_count": len(active_versions),
        "latest_production": _model_version_to_dict(production_versions[0]) if production_versions else None,
        "latest_active": _model_version_to_dict(active_versions[0]) if active_versions else None,
    }
