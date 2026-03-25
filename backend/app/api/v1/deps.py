"""API dependencies for dependency injection."""

from datetime import UTC
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import decode_access_token, verify_api_key
from app.db.models import APIKey, User
from app.db.session import get_db_session
from app.ml.pipeline import ClinicalPipeline

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    api_key: Annotated[str | None, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """Get current user from JWT token or API key."""
    user = None

    # Try JWT authentication first
    if credentials:
        payload = decode_access_token(credentials.credentials, settings)
        if payload and "sub" in payload:
            user_id = payload["sub"]
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

    # Try API key authentication
    elif api_key:
        # Find API key by prefix
        key_prefix = api_key[:10]
        result = await db.execute(
            select(APIKey).where(APIKey.key_prefix == key_prefix, APIKey.is_active)
        )
        api_key_obj = result.scalar_one_or_none()

        if api_key_obj and verify_api_key(api_key, api_key_obj.hashed_key):
            result = await db.execute(
                select(User).where(User.id == api_key_obj.user_id)
            )
            user = result.scalar_one_or_none()

            # Update last used
            from datetime import datetime
            api_key_obj.last_used_at = datetime.now(UTC)
            await db.commit()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Verify user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_superuser(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require superuser privileges."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required",
        )
    return current_user


def get_ml_pipeline() -> ClinicalPipeline:
    """Get ML pipeline instance.

    Returns a :class:`ClinicalPipeline` configured with components from
    the model registry.  The pipeline uses lazy loading, so this function
    is fast and safe to call in dependency injection.

    Design decision: We import the model registry here (rather than at
    module level) to avoid circular imports and to allow the registry
    singletons to initialise only when the first request arrives.
    """
    from app.services.model_registry import (
        get_icd_model,
        get_ner_model,
        get_risk_scorer,
        get_summarizer,
    )

    return ClinicalPipeline(
        ner_model=get_ner_model(),
        icd_classifier=get_icd_model(),
        summarizer=get_summarizer(),
        risk_scorer=get_risk_scorer(),
    )


# Type aliases for cleaner dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db_session)]
CurrentUser = Annotated[User, Depends(get_current_user)]
SuperUser = Annotated[User, Depends(get_superuser)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[ClinicalPipeline, Depends(get_ml_pipeline)]
