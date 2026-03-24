"""Unit tests for the authentication route module.

Tests cover:
- ``POST /auth/token`` — login with valid/invalid credentials, disabled accounts
- ``POST /auth/register`` — new user creation, duplicate email rejection
- ``POST /auth/api-keys`` — API key creation (requires authenticated user)
- ``GET /auth/me`` — current user profile retrieval

All database interactions are mocked; these are pure logic tests for the route
handlers (schema validation, password verification, token generation, duplicate
checks, response shaping).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.schemas.auth import (
    APIKeyCreate,
    TokenRequest,
    UserCreate,
)
from app.api.v1.routes.auth import (
    create_api_key,
    get_current_user_profile,
    login,
    register_user,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(
    *,
    user_id: uuid.UUID | None = None,
    email: str = "doc@hospital.org",
    hashed_password: str = "$2b$12$hashed",
    is_active: bool = True,
    is_superuser: bool = False,
    role: str = "user",
    full_name: str | None = "Dr. Test",
) -> MagicMock:
    """Create a mock User ORM object."""
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.email = email
    user.hashed_password = hashed_password
    user.is_active = is_active
    user.is_superuser = is_superuser
    user.role = role
    user.full_name = full_name
    user.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    user.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return user


def _make_settings() -> MagicMock:
    """Create a mock Settings object with sensible defaults."""
    settings = MagicMock()
    settings.access_token_expire_minutes = 30
    settings.secret_key.get_secret_value.return_value = "test-secret-key-at-least-32-chars-long"
    settings.algorithm = "HS256"
    return settings


def _mock_scalar_result(value):
    """Create a mock query result whose .scalar_one_or_none() returns *value*."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


# ---------------------------------------------------------------------------
# POST /auth/token
# ---------------------------------------------------------------------------


class TestLogin:
    """Tests for the login (token issuance) endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(self) -> None:
        """Valid credentials return a JWT access token."""
        user = _make_user()
        settings = _make_settings()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(user))

        payload = TokenRequest(username="doc@hospital.org", password="Passw0rd!")

        with patch("app.api.v1.routes.auth.verify_password", return_value=True), \
             patch("app.api.v1.routes.auth.create_access_token", return_value="jwt.token.here"):
            result = await login(payload, mock_db, settings)

        assert result.access_token == "jwt.token.here"
        assert result.token_type == "bearer"
        assert result.expires_in == 30 * 60

    @pytest.mark.asyncio
    async def test_login_wrong_password(self) -> None:
        """Incorrect password raises HTTP 401."""
        user = _make_user()
        settings = _make_settings()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(user))

        payload = TokenRequest(username="doc@hospital.org", password="WrongPass")

        with patch("app.api.v1.routes.auth.verify_password", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await login(payload, mock_db, settings)

        assert exc_info.value.status_code == 401
        assert "Incorrect" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_login_unknown_email(self) -> None:
        """Non-existent email raises HTTP 401."""
        settings = _make_settings()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(None))

        payload = TokenRequest(username="nobody@example.com", password="Passw0rd!")

        with pytest.raises(HTTPException) as exc_info:
            await login(payload, mock_db, settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_login_disabled_account(self) -> None:
        """Inactive account raises HTTP 401 even with correct password."""
        user = _make_user(is_active=False)
        settings = _make_settings()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(user))

        payload = TokenRequest(username="doc@hospital.org", password="Passw0rd!")

        with patch("app.api.v1.routes.auth.verify_password", return_value=True):
            with pytest.raises(HTTPException) as exc_info:
                await login(payload, mock_db, settings)

        assert exc_info.value.status_code == 401
        assert "disabled" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------


class TestRegisterUser:
    """Tests for the user registration endpoint."""

    @pytest.mark.asyncio
    async def test_register_success(self) -> None:
        """New email creates user and returns UserResponse."""
        settings = _make_settings()

        mock_db = AsyncMock()
        # First query: duplicate check → no existing user
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(None))
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        payload = UserCreate(
            email="new@hospital.org",
            password="Secur3Pa$$",
            full_name="New Doctor",
        )

        with patch("app.api.v1.routes.auth.get_password_hash", return_value="$2b$hashed"), \
             patch("app.api.v1.routes.auth.UserResponse") as MockResponse:
            # Mock UserResponse.model_validate to return a simple object
            mock_resp = MagicMock()
            MockResponse.model_validate.return_value = mock_resp

            result = await register_user(payload, mock_db, settings)

        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()
        assert result is mock_resp

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self) -> None:
        """Existing email raises HTTP 409."""
        existing_user = _make_user()
        settings = _make_settings()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=_mock_scalar_result(existing_user))

        payload = UserCreate(
            email="doc@hospital.org",
            password="Secur3Pa$$",
            full_name="Dup Doctor",
        )

        with pytest.raises(HTTPException) as exc_info:
            await register_user(payload, mock_db, settings)

        assert exc_info.value.status_code == 409
        assert "already registered" in exc_info.value.detail


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------


class TestGetCurrentUserProfile:
    """Tests for the current user profile endpoint."""

    @pytest.mark.asyncio
    async def test_returns_user_profile(self) -> None:
        """Authenticated user gets their own profile."""
        user = _make_user()

        with patch("app.api.v1.routes.auth.UserResponse") as MockResponse:
            mock_resp = MagicMock()
            MockResponse.model_validate.return_value = mock_resp

            result = await get_current_user_profile(user)

        assert result is mock_resp
        MockResponse.model_validate.assert_called_once_with(user)


# ---------------------------------------------------------------------------
# POST /auth/api-keys
# ---------------------------------------------------------------------------


class TestCreateAPIKey:
    """Tests for the API key creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_api_key_success(self) -> None:
        """Authenticated user can create an API key."""
        import uuid as _uuid
        from datetime import datetime, timezone

        user = _make_user()
        settings = _make_settings()

        _fake_id = _uuid.uuid4()
        _fake_now = datetime.now(timezone.utc)

        mock_db = AsyncMock()

        def _add_side_effect(obj):
            """Simulate DB server-assigned defaults on the APIKey instance."""
            obj.id = _fake_id
            obj.created_at = _fake_now

        mock_db.add = MagicMock(side_effect=_add_side_effect)
        mock_db.flush = AsyncMock()

        payload = APIKeyCreate(name="test-key", rate_limit=100)

        with patch("app.api.v1.routes.auth.generate_api_key", return_value="cliniq_testkey1234567890"), \
             patch("app.api.v1.routes.auth.hash_api_key", return_value="$2b$hashed_key"):
            result = await create_api_key(payload, mock_db, settings, user)

        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()
        assert result.plain_key == "cliniq_testkey1234567890"
        assert result.name == "test-key"
        assert result.key_prefix == "cliniq_tes"
