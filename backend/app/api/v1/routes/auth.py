"""Authentication and user management endpoints.

Provides OAuth2 password-flow token issuance, user registration, API key
management, and current-user profile retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.auth import (
    APIKeyCreate,
    APIKeyResponse,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.api.v1.deps import get_current_user
from app.core.config import Settings, get_settings
from app.core.security import (
    create_access_token,
    generate_api_key,
    get_password_hash,
    hash_api_key,
    verify_password,
)
from app.db.models import APIKey, User
from app.db.session import get_db_session

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/auth/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Obtain JWT access token",
    description=(
        "Exchange a user email + password for a signed JWT access token. "
        "Submit as JSON body (`username` = email address, `password`). "
        "The returned `access_token` must be sent as an `Authorization: Bearer <token>` "
        "header on all protected endpoints."
    ),
    responses={
        200: {"description": "Token issued successfully"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login(
    payload: TokenRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenResponse:
    """Authenticate a user and return a JWT access token."""
    result = await db.execute(select(User).where(User.email == payload.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled. Contact your administrator.",
        )

    token = create_access_token(
        data={"sub": str(user.id), "email": user.email, "role": user.role},
        settings=settings,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        scope="cliniq:read cliniq:write",
    )


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description=(
        "Create a new ClinIQ user account. The `email` must be unique. "
        "Passwords are hashed with bcrypt before storage and are never returned "
        "by the API."
    ),
    responses={
        201: {"description": "User created"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error"},
    },
)
async def register_user(
    payload: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserResponse:
    """Create a new user account."""
    # Check uniqueness.
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The email address '{payload.email}' is already registered.",
        )

    user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        is_active=True,
        is_superuser=False,
        role="user",
    )
    db.add(user)
    # The session is flushed so we can read auto-generated values before the
    # response is serialised; the commit happens via get_db_session.
    await db.flush()

    return UserResponse.model_validate(user)


@router.post(
    "/auth/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description=(
        "Generate a new API key for programmatic access. "
        "The full plaintext key is returned **only at creation time** and cannot "
        "be retrieved again — store it securely immediately. "
        "Requires an authenticated session (Bearer token)."
    ),
    responses={
        201: {"description": "API key created — store the `plain_key` value now"},
        401: {"description": "Authentication required"},
        422: {"description": "Validation error"},
    },
)
async def create_api_key(
    payload: APIKeyCreate,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> APIKeyResponse:
    """Create a new API key for the authenticated user."""
    plain_key = generate_api_key()
    hashed = hash_api_key(plain_key)
    prefix = plain_key[:10]

    expires_at: datetime | None = None
    if payload.expires_at is not None:
        expires_at = payload.expires_at

    api_key = APIKey(
        user_id=current_user.id,
        name=payload.name,
        hashed_key=hashed,
        key_prefix=prefix,
        is_active=True,
        rate_limit=payload.rate_limit,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.flush()

    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=prefix,
        plain_key=plain_key,  # Shown exactly once.
        is_active=api_key.is_active,
        rate_limit=api_key.rate_limit,
        expires_at=api_key.expires_at,
        last_used_at=api_key.last_used_at,
        created_at=api_key.created_at,
    )


@router.get(
    "/auth/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Current user profile",
    description=(
        "Return the profile of the currently authenticated user. "
        "Requires a valid Bearer token."
    ),
    responses={
        200: {"description": "User profile returned"},
        401: {"description": "Authentication required"},
    },
)
async def get_current_user_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Return the authenticated user's profile."""
    return UserResponse.model_validate(current_user)
