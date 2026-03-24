"""Unit tests for app.api.v1.deps.get_current_user and get_current_active_user.

Covers the JWT auth path, API key auth path, missing credentials,
inactive users, and the active-user wrapper — targeting the uncovered
lines 30–73 and 80–82 in deps.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.v1.deps import get_current_active_user, get_current_user


def _mock_user(*, is_active: bool = True) -> MagicMock:
    """Create a mock User ORM object."""
    user = MagicMock()
    user.id = uuid4()
    user.is_active = is_active
    return user


def _mock_settings() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# JWT Bearer token path
# ---------------------------------------------------------------------------


class TestDepsJWTAuth:
    """JWT token authentication through deps.get_current_user."""

    @pytest.mark.asyncio
    async def test_valid_jwt_returns_user(self) -> None:
        """Valid JWT with matching user returns the user object."""
        user = _mock_user()
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="good-token"
        )

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with patch(
            "app.api.v1.deps.decode_access_token",
            return_value={"sub": str(user.id)},
        ):
            result = await get_current_user(
                db=mock_db,
                credentials=credentials,
                api_key=None,
                settings=_mock_settings(),
            )
        assert result is user

    @pytest.mark.asyncio
    async def test_invalid_jwt_raises_401(self) -> None:
        """decode_access_token returning None → 401."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="bad"
        )
        mock_db = AsyncMock()

        with patch("app.api.v1.deps.decode_access_token", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=credentials,
                    api_key=None,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_missing_sub_raises_401(self) -> None:
        """Token payload without 'sub' claim → 401."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="token"
        )
        mock_db = AsyncMock()

        with patch(
            "app.api.v1.deps.decode_access_token", return_value={"exp": 9999}
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=credentials,
                    api_key=None,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_user_not_found_raises_401(self) -> None:
        """Valid token but user does not exist in DB → 401."""
        user_id = str(uuid4())
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="token"
        )

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with patch(
            "app.api.v1.deps.decode_access_token", return_value={"sub": user_id}
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=credentials,
                    api_key=None,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_inactive_user_raises_403(self) -> None:
        """Authenticated but inactive user → 403."""
        user = _mock_user(is_active=False)
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="token"
        )

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with patch(
            "app.api.v1.deps.decode_access_token",
            return_value={"sub": str(user.id)},
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=credentials,
                    api_key=None,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# API key path
# ---------------------------------------------------------------------------


class TestDepsAPIKeyAuth:
    """API key authentication through deps.get_current_user."""

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_user(self) -> None:
        """Valid API key → resolved user returned, last_used_at updated."""
        user = _mock_user()
        mock_api_key_obj = MagicMock()
        mock_api_key_obj.hashed_key = "hashed-value"
        mock_api_key_obj.user_id = user.id
        mock_api_key_obj.is_active = True
        mock_api_key_obj.last_used_at = None

        mock_db = AsyncMock()

        # First execute: find API key by prefix
        api_key_result = MagicMock()
        api_key_result.scalar_one_or_none.return_value = mock_api_key_obj
        # Second execute: find user by ID
        user_result = MagicMock()
        user_result.scalar_one_or_none.return_value = user
        mock_db.execute.side_effect = [api_key_result, user_result]

        with patch("app.api.v1.deps.verify_api_key", return_value=True):
            result = await get_current_user(
                db=mock_db,
                credentials=None,
                api_key="sk-1234567890-rest-of-key",
                settings=_mock_settings(),
            )
        assert result is user
        # last_used_at should have been updated
        assert mock_api_key_obj.last_used_at is not None
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_key_not_found_raises_401(self) -> None:
        """API key prefix not in DB → 401."""
        mock_db = AsyncMock()
        api_key_result = MagicMock()
        api_key_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = api_key_result

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                db=mock_db,
                credentials=None,
                api_key="sk-unknown-key-prefix",
                settings=_mock_settings(),
            )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_hash_mismatch_raises_401(self) -> None:
        """API key found but hash doesn't match → 401."""
        mock_api_key_obj = MagicMock()
        mock_api_key_obj.hashed_key = "hashed"
        mock_api_key_obj.is_active = True

        mock_db = AsyncMock()
        api_key_result = MagicMock()
        api_key_result.scalar_one_or_none.return_value = mock_api_key_obj
        mock_db.execute.return_value = api_key_result

        with patch("app.api.v1.deps.verify_api_key", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=None,
                    api_key="sk-bad-hash-key-xx",
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_user_not_found_raises_401(self) -> None:
        """API key valid but owning user gone → 401."""
        mock_api_key_obj = MagicMock()
        mock_api_key_obj.hashed_key = "hashed"
        mock_api_key_obj.user_id = uuid4()
        mock_api_key_obj.is_active = True
        mock_api_key_obj.last_used_at = None

        mock_db = AsyncMock()
        api_key_result = MagicMock()
        api_key_result.scalar_one_or_none.return_value = mock_api_key_obj
        user_result = MagicMock()
        user_result.scalar_one_or_none.return_value = None
        mock_db.execute.side_effect = [api_key_result, user_result]

        with patch("app.api.v1.deps.verify_api_key", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    db=mock_db,
                    credentials=None,
                    api_key="sk-orphan-key-xxx",
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# No credentials
# ---------------------------------------------------------------------------


class TestDepsNoCredentials:
    """No auth provided at all."""

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self) -> None:
        mock_db = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                db=mock_db,
                credentials=None,
                api_key=None,
                settings=_mock_settings(),
            )
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_active_user
# ---------------------------------------------------------------------------


class TestGetCurrentActiveUser:
    """Tests for the get_current_active_user wrapper."""

    @pytest.mark.asyncio
    async def test_active_user_passes_through(self) -> None:
        user = _mock_user(is_active=True)
        result = await get_current_active_user(user)
        assert result is user

    @pytest.mark.asyncio
    async def test_inactive_user_raises_400(self) -> None:
        user = _mock_user(is_active=False)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(user)
        assert exc_info.value.status_code == 400
