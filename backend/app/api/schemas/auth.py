"""Authentication and user management schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth / token schemas
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    """OAuth2-compatible password-grant request body."""

    username: str = Field(description="User email address used as the username")
    password: str = Field(description="User password (transmitted only over HTTPS)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "clinician@hospital.org",
                "password": "s3cur3P@ssword",
            }
        }
    }


class TokenResponse(BaseModel):
    """JWT token pair returned on successful authentication."""

    access_token: str = Field(description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token type; always 'bearer'")
    expires_in: int = Field(
        ge=0,
        description="Seconds until the access token expires",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
            }
        }
    }


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Request body for new user registration."""

    email: EmailStr = Field(description="Unique email address for the new account")
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Account password (minimum 8 characters)",
    )
    full_name: str | None = Field(
        default=None,
        max_length=255,
        description="Optional display name for the user",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "email": "clinician@hospital.org",
                "password": "s3cur3P@ssword",
                "full_name": "Dr. Jane Smith",
            }
        }
    }


class UserResponse(BaseModel):
    """Public user profile returned by auth endpoints."""

    id: UUID = Field(description="Unique user identifier")
    email: str = Field(description="User email address")
    full_name: str | None = Field(default=None, description="Display name")
    is_active: bool = Field(description="Whether the account is enabled")
    role: str = Field(description="User role (e.g. 'user', 'admin')")
    created_at: datetime = Field(description="UTC timestamp of account creation")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "email": "clinician@hospital.org",
                "full_name": "Dr. Jane Smith",
                "is_active": True,
                "role": "user",
                "created_at": "2026-03-24T09:00:00Z",
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
        description="Human-readable label for this API key (e.g. 'CI pipeline key')",
    )
    rate_limit: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum number of API requests allowed per 24-hour period",
    )
    expires_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="Key lifetime in days from now. Omit for a non-expiring key.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "production-etl-key",
                "rate_limit": 1000,
                "expires_days": 365,
            }
        }
    }


class APIKeyResponse(BaseModel):
    """Newly created API key; the raw key is shown only once."""

    id: UUID = Field(description="API key unique identifier")
    name: str = Field(description="Human-readable label")
    key: str | None = Field(
        default=None,
        description=(
            "Raw API key value — returned only at creation time and never again. "
            "Store it securely immediately."
        ),
    )
    key_prefix: str = Field(description="First few characters of the key for identification")
    rate_limit: int = Field(description="Requests allowed per 24-hour period")
    is_active: bool = Field(description="Whether the key is currently active")
    expires_at: datetime | None = Field(default=None, description="UTC expiry timestamp, or null for non-expiring")
    created_at: datetime = Field(description="UTC creation timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "production-etl-key",
                "key": "cliniq_abc123...",
                "key_prefix": "cliniq_a",
                "rate_limit": 1000,
                "is_active": True,
                "expires_at": "2027-03-24T00:00:00Z",
                "created_at": "2026-03-24T09:00:00Z",
            }
        },
    }
