"""Authentication middleware and FastAPI dependency functions.

Design decisions:
    - **Dual auth schemes** — Supports both JWT Bearer tokens (for browser/SPA
      sessions) and API key headers (for programmatic/SDK access).  The JWT
      path is tried first because it's cheaper to validate (no DB round-trip
      for key comparison).
    - **Dependency injection** — Each auth level (``get_current_user``,
      ``get_current_active_user``, ``get_current_superuser``) is a FastAPI
      ``Depends``-compatible async function, composable in route signatures.
    - **RBAC factory** — ``require_role()`` returns a dependency closure that
      checks the user's ``role`` field.  Superusers bypass role checks entirely,
      following the principle of least surprise.
    - **API key timing** — We iterate all active keys and use constant-time
      ``verify_api_key`` to avoid timing-based enumeration.  For large
      key sets, consider indexing by a key prefix (first 8 chars).
    - **Optional auth** — ``get_optional_user`` allows public endpoints to
      still benefit from user context when auth is present.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token, verify_api_key
from app.db.models import APIKey, User
from app.db.session import get_db_session

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> User:
    """Extract and validate the current user from JWT token or API key."""
    # Try JWT token first
    if credentials:
        payload = decode_access_token(credentials.credentials, settings)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        result = await db.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        return user

    # Try API key
    if api_key:
        result = await db.execute(select(APIKey).where(APIKey.is_active.is_(True)))
        api_keys = result.scalars().all()

        for key_record in api_keys:
            if verify_api_key(api_key, key_record.hashed_key):
                # Update last used timestamp
                key_record.last_used_at = datetime.now(timezone.utc)

                # Fetch the associated user
                user_result = await db.execute(
                    select(User).where(User.id == key_record.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user and user.is_active:
                    return user

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the current user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Ensure the current user is a superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required",
        )
    return current_user


def require_role(allowed_roles: list[str]):
    """Dependency factory for role-based access control."""

    async def check_role(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Verify the authenticated user holds one of the *allowed_roles*.

        Superusers bypass the role check. Raises 403 Forbidden when the
        user's role is not in the allowed set.
        """
        if current_user.role not in allowed_roles and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' not authorized. Required: {allowed_roles}",
            )
        return current_user

    return check_role


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
    db: AsyncSession = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Optionally extract user - returns None if no auth provided."""
    if not credentials and not api_key:
        return None

    try:
        return await get_current_user(
            request=None,  # type: ignore[arg-type]
            credentials=credentials,
            api_key=api_key,
            db=db,
            settings=settings,
        )
    except HTTPException:
        return None
