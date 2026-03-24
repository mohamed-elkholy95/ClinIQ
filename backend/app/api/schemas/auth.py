"""Authentication and authorisation schemas."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# EmailStr requires the optional 'email-validator' package.  Use an annotated
# str with a pattern constraint so the schema works without that extra dep while
# still expressing the intent clearly in the OpenAPI spec.
EmailStr = Annotated[
    str,
    Field(
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        max_length=255,
        description="Valid email address",
        examples=["clinician@hospital.org"],
    ),
]

# ---------------------------------------------------------------------------
# Token schemas
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    """OAuth2 password-flow token request (maps to /auth/token form data)."""

    username: str = Field(
        description="User's email address used as the login credential",
        examples=["clinician@hospital.org"],
    )
    password: str = Field(
        min_length=8,
        description="User's password",
    )
    grant_type: Literal["password"] = Field(
        default="password",
        description="OAuth2 grant type; must be 'password'",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "clinician@hospital.org",
                "password": "S3cureP@ssw0rd",
                "grant_type": "password",
            }
        }
    }


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str = Field(description="Signed JWT access token")
    refresh_token: str | None = Field(
        default=None,
        description="Opaque refresh token for obtaining a new access token after expiry",
    )
    token_type: Literal["bearer"] = Field(
        default="bearer",
        description="OAuth2 token type; always 'bearer'",
    )
    expires_in: int = Field(
        ge=1,
        description="Access token lifetime in seconds from the time of issue",
    )
    scope: str = Field(
        default="cliniq:read cliniq:write",
        description="Space-separated list of OAuth2 scopes granted to this token",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4...",
                "token_type": "bearer",
                "expires_in": 1800,
                "scope": "cliniq:read cliniq:write",
            }
        }
    }


class RefreshTokenRequest(BaseModel):
    """Request body for refreshing an expired access token."""

    refresh_token: str = Field(description="Refresh token previously issued by /auth/token")

    model_config = {
        "json_schema_extra": {
            "example": {"refresh_token": "dGhpcyBpcyBhIHJlZnJlc2ggdG9rZW4..."}
        }
    }


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Request body for registering a new user account."""

    email: EmailStr  # type: ignore[valid-type]  # annotated alias defined above
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Plain-text password (minimum 8 characters; hashed server-side before storage)",
    )
    full_name: str | None = Field(
        default=None,
        max_length=255,
        description="User's full name (optional)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "clinician@hospital.org",
                "password": "S3cureP@ssw0rd",
                "full_name": "Dr. Jane Smith",
            }
        }
    }


class UserUpdate(BaseModel):
    """Request body for updating an existing user account (all fields optional)."""

    full_name: str | None = Field(
        default=None,
        max_length=255,
        description="New full name",
    )
    password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="New password (hashed server-side before storage)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"full_name": "Dr. Jane Smith", "password": None}
        }
    }


class UserResponse(BaseModel):
    """Public user representation returned from the API (no password hash)."""

    id: UUID = Field(description="Server-assigned unique user identifier (UUID v4)")
    email: str = Field(description="User's email address")
    full_name: str | None = Field(default=None, description="User's full display name")
    is_active: bool = Field(description="Whether the account is currently active")
    is_superuser: bool = Field(description="Whether the user has super-admin privileges")
    role: str = Field(description="Role assigned to the user (e.g. 'user', 'admin', 'clinician')")
    created_at: datetime = Field(description="UTC timestamp when the account was created")
    updated_at: datetime = Field(description="UTC timestamp of the last account update")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "email": "clinician@hospital.org",
                "full_name": "Dr. Jane Smith",
                "is_active": True,
                "is_superuser": False,
                "role": "clinician",
                "created_at": "2026-01-10T08:00:00Z",
                "updated_at": "2026-03-20T14:30:00Z",
            }
        },
    }


# ---------------------------------------------------------------------------
# API key schemas
# ---------------------------------------------------------------------------


class APIKeyCreate(BaseModel):
    """Request body for creating a new API key."""

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Human-readable label for the key (e.g. 'EHR integration prod')",
    )
    rate_limit: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum number of API requests allowed per 24-hour period for this key",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiry timestamp (UTC). Omit for a non-expiring key.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "EHR integration prod",
                "rate_limit": 500,
                "expires_at": None,
            }
        }
    }


class APIKeyResponse(BaseModel):
    """API key representation returned from the API."""

    id: UUID = Field(description="Server-assigned unique key identifier (UUID v4)")
    name: str = Field(description="Human-readable label for the key")
    key_prefix: str = Field(
        description="First few characters of the key (e.g. 'ciq_abc1'), safe to display",
    )
    plain_key: str | None = Field(
        default=None,
        description=(
            "The full plaintext key value. ONLY returned at creation time. "
            "Store it securely — it cannot be retrieved again."
        ),
    )
    is_active: bool = Field(description="Whether the key is currently active")
    rate_limit: int = Field(description="Maximum requests per 24-hour period")
    expires_at: datetime | None = Field(default=None, description="Key expiry timestamp (null = no expiry)")
    last_used_at: datetime | None = Field(default=None, description="UTC timestamp of the most recent use")
    created_at: datetime = Field(description="UTC timestamp when the key was created")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "name": "EHR integration prod",
                "key_prefix": "ciq_abc1",
                "plain_key": "ciq_abc1xxxxxxxxxxxxxxxxxxxxxxxx",
                "is_active": True,
                "rate_limit": 500,
                "expires_at": None,
                "last_used_at": None,
                "created_at": "2026-03-24T10:00:00Z",
            }
        },
    }
