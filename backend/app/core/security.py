"""Security utilities for authentication and authorization."""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import Settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password[:72], hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password (bcrypt truncates at 72 bytes)."""
    return pwd_context.hash(password[:72])


def create_access_token(
    data: dict[str, Any],
    settings: Settings,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key.get_secret_value(),
        algorithm=settings.algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str, settings: Settings) -> dict[str, Any] | None:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
        )
        return payload
    except JWTError:
        return None


def generate_api_key() -> str:
    """Generate a secure API key."""
    import secrets
    return f"cliniq_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage (truncated to 72 bytes for bcrypt)."""
    return pwd_context.hash(api_key[:72])


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return pwd_context.verify(plain_key[:72], hashed_key)
