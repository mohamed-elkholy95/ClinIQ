"""Unit tests for the authentication middleware dependencies.

Tests the require_role factory and get_optional_user dependency
using lightweight mocks (no database required).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.middleware.auth import get_current_active_user, get_current_superuser, require_role


def _mock_user(
    *,
    is_active: bool = True,
    is_superuser: bool = False,
    role: str = "user",
) -> MagicMock:
    """Create a mock User object."""
    user = MagicMock()
    user.id = uuid4()
    user.is_active = is_active
    user.is_superuser = is_superuser
    user.role = role
    return user


class TestGetCurrentActiveUser:
    """Validate the active-user gate."""

    @pytest.mark.asyncio
    async def test_active_user_passes_through(self):
        user = _mock_user(is_active=True)
        result = await get_current_active_user(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_inactive_user_raises_403(self):
        user = _mock_user(is_active=False)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user=user)
        assert exc_info.value.status_code == 403


class TestGetCurrentSuperuser:
    """Validate the superuser gate."""

    @pytest.mark.asyncio
    async def test_superuser_passes_through(self):
        user = _mock_user(is_superuser=True)
        result = await get_current_superuser(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_non_superuser_raises_403(self):
        user = _mock_user(is_superuser=False)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user=user)
        assert exc_info.value.status_code == 403


class TestRequireRole:
    """Validate the role-based access control factory."""

    @pytest.mark.asyncio
    async def test_allowed_role_passes(self):
        dep = require_role(["admin", "clinician"])
        user = _mock_user(role="clinician")
        result = await dep(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_disallowed_role_raises_403(self):
        dep = require_role(["admin"])
        user = _mock_user(role="viewer")
        with pytest.raises(HTTPException) as exc_info:
            await dep(current_user=user)
        assert exc_info.value.status_code == 403
        assert "viewer" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_superuser_bypasses_role_check(self):
        dep = require_role(["admin"])
        user = _mock_user(role="viewer", is_superuser=True)
        result = await dep(current_user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_empty_allowed_roles_rejects_non_superuser(self):
        dep = require_role([])
        user = _mock_user(role="clinician")
        with pytest.raises(HTTPException):
            await dep(current_user=user)
