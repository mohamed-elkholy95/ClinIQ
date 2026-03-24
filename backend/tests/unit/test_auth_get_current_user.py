"""Unit tests for get_current_user and get_optional_user.

Tests the JWT Bearer token path, API key path, missing credentials,
inactive users, and the optional auth wrapper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.middleware.auth import get_current_user, get_optional_user


def _mock_user(*, is_active: bool = True, is_superuser: bool = False) -> MagicMock:
    user = MagicMock()
    user.id = uuid4()
    user.is_active = is_active
    user.is_superuser = is_superuser
    user.role = "user"
    return user


def _mock_request() -> MagicMock:
    return MagicMock()


def _mock_settings() -> MagicMock:
    return MagicMock()


class TestGetCurrentUserJWT:
    """Tests for JWT Bearer token authentication path."""

    @pytest.mark.asyncio
    async def test_valid_token(self) -> None:
        user = _mock_user()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with patch("app.middleware.auth.decode_access_token", return_value={"sub": str(user.id)}):
            result = await get_current_user(
                request=_mock_request(),
                credentials=credentials,
                api_key=None,
                db=mock_db,
                settings=_mock_settings(),
            )
        assert result is user

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self) -> None:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
        mock_db = AsyncMock()

        with patch("app.middleware.auth.decode_access_token", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    request=_mock_request(),
                    credentials=credentials,
                    api_key=None,
                    db=mock_db,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401
        assert "Invalid or expired" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_token_missing_sub_raises_401(self) -> None:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
        mock_db = AsyncMock()

        with patch("app.middleware.auth.decode_access_token", return_value={"exp": 9999}):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    request=_mock_request(),
                    credentials=credentials,
                    api_key=None,
                    db=mock_db,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401
        assert "Invalid token payload" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self) -> None:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
        user_id = str(uuid4())

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with patch("app.middleware.auth.decode_access_token", return_value={"sub": user_id}):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    request=_mock_request(),
                    credentials=credentials,
                    api_key=None,
                    db=mock_db,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_inactive_user_raises_401(self) -> None:
        user = _mock_user(is_active=False)
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with patch("app.middleware.auth.decode_access_token", return_value={"sub": str(user.id)}):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    request=_mock_request(),
                    credentials=credentials,
                    api_key=None,
                    db=mock_db,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401


class TestGetCurrentUserAPIKey:
    """Tests for API key authentication path."""

    @pytest.mark.asyncio
    async def test_valid_api_key(self) -> None:
        user = _mock_user()
        mock_api_key = MagicMock()
        mock_api_key.hashed_key = "hashed"
        mock_api_key.user_id = user.id
        mock_api_key.last_used_at = None

        mock_db = AsyncMock()
        # First execute returns API keys, second returns user
        mock_keys_result = MagicMock()
        mock_keys_result.scalars.return_value.all.return_value = [mock_api_key]
        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = user
        mock_db.execute.side_effect = [mock_keys_result, mock_user_result]

        with patch("app.middleware.auth.verify_api_key", return_value=True):
            result = await get_current_user(
                request=_mock_request(),
                credentials=None,
                api_key="sk-test-key",
                db=mock_db,
                settings=_mock_settings(),
            )
        assert result is user

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self) -> None:
        mock_db = AsyncMock()
        mock_keys_result = MagicMock()
        mock_keys_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_keys_result

        with patch("app.middleware.auth.verify_api_key", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(
                    request=_mock_request(),
                    credentials=None,
                    api_key="bad-key",
                    db=mock_db,
                    settings=_mock_settings(),
                )
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self) -> None:
        mock_db = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=_mock_request(),
                credentials=None,
                api_key=None,
                db=mock_db,
                settings=_mock_settings(),
            )
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail


class TestGetOptionalUser:
    """Tests for the optional authentication wrapper."""

    @pytest.mark.asyncio
    async def test_no_auth_returns_none(self) -> None:
        mock_db = AsyncMock()
        result = await get_optional_user(
            credentials=None,
            api_key=None,
            db=mock_db,
            settings=_mock_settings(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_auth_error_returns_none(self) -> None:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad")
        mock_db = AsyncMock()

        with patch("app.middleware.auth.get_current_user", side_effect=HTTPException(status_code=401)):
            result = await get_optional_user(
                credentials=credentials,
                api_key=None,
                db=mock_db,
                settings=_mock_settings(),
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_auth_returns_user(self) -> None:
        user = _mock_user()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="good")
        mock_db = AsyncMock()

        with patch("app.middleware.auth.get_current_user", return_value=user):
            result = await get_optional_user(
                credentials=credentials,
                api_key=None,
                db=mock_db,
                settings=_mock_settings(),
            )
        assert result is user
