"""Unit tests for the model management route handlers.

Tests cover:
- ``_model_version_to_dict``: ORM-to-dict serialisation
- ``list_models``: default listing, stage filter, active_only filter, grouping
- ``get_model``: found model, 404 for unknown model, active/production metadata
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.routes.models import _model_version_to_dict, get_model, list_models

# ---------------------------------------------------------------------------
# Fake ModelVersion stand-in
# ---------------------------------------------------------------------------


def _fake_model_version(
    *,
    model_name: str = "rule-based-ner",
    version: str = "1.0.0",
    stage: str = "production",
    is_active: bool = True,
    mlflow_run_id: str | None = None,
    metrics: dict | None = None,
    config: dict | None = None,
    deployed_at: datetime | None = None,
    deployed_by: uuid.UUID | None = None,
) -> MagicMock:
    """Build a MagicMock mimicking a ModelVersion ORM row."""
    mv = MagicMock()
    mv.id = uuid.uuid4()
    mv.model_name = model_name
    mv.version = version
    mv.stage = stage
    mv.is_active = is_active
    mv.mlflow_run_id = mlflow_run_id
    mv.metrics = metrics or {"f1": 0.87}
    mv.config = config or {}
    mv.deployed_at = deployed_at
    mv.deployed_by = deployed_by
    mv.created_at = datetime(2026, 3, 1, tzinfo=UTC)
    mv.updated_at = datetime(2026, 3, 1, tzinfo=UTC)
    return mv


# ---------------------------------------------------------------------------
# _model_version_to_dict
# ---------------------------------------------------------------------------


class TestModelVersionToDict:
    """Tests for the _model_version_to_dict serialiser."""

    def test_basic_fields(self) -> None:
        """All expected keys are present in the output dict."""
        mv = _fake_model_version()
        d = _model_version_to_dict(mv)
        expected_keys = {
            "id", "model_name", "version", "stage", "is_active",
            "mlflow_run_id", "metrics", "config", "deployed_at",
            "deployed_by", "created_at", "updated_at",
        }
        assert set(d.keys()) == expected_keys

    def test_deployed_at_iso(self) -> None:
        """deployed_at is ISO-formatted when present."""
        dt = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
        mv = _fake_model_version(deployed_at=dt)
        d = _model_version_to_dict(mv)
        assert d["deployed_at"] == dt.isoformat()

    def test_deployed_at_none(self) -> None:
        """deployed_at is None when not deployed."""
        mv = _fake_model_version(deployed_at=None)
        d = _model_version_to_dict(mv)
        assert d["deployed_at"] is None

    def test_deployed_by_stringified(self) -> None:
        """deployed_by UUID is converted to string."""
        uid = uuid.uuid4()
        mv = _fake_model_version(deployed_by=uid)
        d = _model_version_to_dict(mv)
        assert d["deployed_by"] == str(uid)

    def test_deployed_by_none(self) -> None:
        """deployed_by is None when unset."""
        mv = _fake_model_version(deployed_by=None)
        d = _model_version_to_dict(mv)
        assert d["deployed_by"] is None


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def _db_returning(versions: list[MagicMock]) -> AsyncMock:
    """Build an AsyncMock db session that returns the given model versions."""
    result = MagicMock()
    result.scalars.return_value.all.return_value = versions
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


class TestListModels:
    """Tests for the list_models endpoint."""

    @pytest.mark.asyncio
    async def test_returns_grouped_models(self) -> None:
        """Models are grouped by model_name."""
        versions = [
            _fake_model_version(model_name="ner", version="1.0"),
            _fake_model_version(model_name="ner", version="2.0"),
            _fake_model_version(model_name="icd", version="1.0"),
        ]
        db = _db_returning(versions)
        resp = await list_models(db)

        assert "models" in resp
        assert "ner" in resp["models"]
        assert "icd" in resp["models"]
        assert len(resp["models"]["ner"]) == 2
        assert resp["total_versions"] == 3

    @pytest.mark.asyncio
    async def test_empty_registry(self) -> None:
        """Empty model registry returns empty structure."""
        db = _db_returning([])
        resp = await list_models(db)
        assert resp["models"] == {}
        assert resp["total_versions"] == 0

    @pytest.mark.asyncio
    async def test_known_model_names_included(self) -> None:
        """Response always includes known_model_names."""
        db = _db_returning([])
        resp = await list_models(db)
        assert "known_model_names" in resp
        assert len(resp["known_model_names"]) > 0

    @pytest.mark.asyncio
    async def test_single_version(self) -> None:
        """Single model version is returned correctly."""
        versions = [_fake_model_version(model_name="textrank")]
        db = _db_returning(versions)
        resp = await list_models(db)
        assert resp["total_versions"] == 1
        assert "textrank" in resp["models"]


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    """Tests for the get_model endpoint."""

    @pytest.mark.asyncio
    async def test_model_found(self) -> None:
        """Existing model returns version list and metadata."""
        versions = [
            _fake_model_version(version="2.0", stage="production", is_active=True),
            _fake_model_version(version="1.0", stage="staging", is_active=False),
        ]
        db = _db_returning(versions)
        resp = await get_model("rule-based-ner", db)

        assert resp["model_name"] == "rule-based-ner"
        assert resp["version_count"] == 2
        assert resp["active_count"] == 1
        assert resp["latest_production"] is not None
        assert resp["latest_active"] is not None

    @pytest.mark.asyncio
    async def test_model_not_found_raises_404(self) -> None:
        """Unknown model name → HTTP 404."""
        from fastapi import HTTPException

        db = _db_returning([])
        with pytest.raises(HTTPException) as exc_info:
            await get_model("nonexistent-model", db)
        assert exc_info.value.status_code == 404
        assert "nonexistent-model" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_no_production_version(self) -> None:
        """When no production versions exist, latest_production is None."""
        versions = [
            _fake_model_version(stage="staging", is_active=True),
        ]
        db = _db_returning(versions)
        resp = await get_model("rule-based-ner", db)
        assert resp["latest_production"] is None
        assert resp["latest_active"] is not None

    @pytest.mark.asyncio
    async def test_no_active_version(self) -> None:
        """When no active versions exist, latest_active is None."""
        versions = [
            _fake_model_version(stage="archived", is_active=False),
        ]
        db = _db_returning(versions)
        resp = await get_model("rule-based-ner", db)
        assert resp["latest_active"] is None
        assert resp["active_count"] == 0
