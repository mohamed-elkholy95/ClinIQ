"""Unit tests for core security utilities."""

from datetime import timedelta

import pytest

from app.core.config import Settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_api_key,
    get_password_hash,
    hash_api_key,
    verify_api_key,
    verify_password,
)


@pytest.fixture
def test_settings() -> Settings:
    """Provide minimal Settings for security tests."""
    return Settings(
        secret_key="super-secret-test-key-32-chars-ok",
        algorithm="HS256",
        access_token_expire_minutes=30,
        environment="development",
    )


class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_get_password_hash_returns_string(self):
        """Test that hashing returns a non-empty string."""
        hashed = get_password_hash("testpassword123")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_get_password_hash_is_not_plaintext(self):
        """Test that the hash is not the same as the plaintext."""
        password = "my_secure_password"
        hashed = get_password_hash(password)
        assert hashed != password

    def test_get_password_hash_different_each_call(self):
        """Test that hashing the same password twice yields different hashes (salted)."""
        password = "same_password"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        # bcrypt generates a new salt each call, so hashes differ
        assert hash1 != hash2

    def test_verify_password_correct(self):
        """Test that verify_password returns True for a correct password."""
        password = "correct_horse_battery_staple"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that verify_password returns False for a wrong password."""
        hashed = get_password_hash("original_password")
        assert verify_password("wrong_password", hashed) is False

    def test_verify_password_empty_plain(self):
        """Test verify_password with empty plaintext."""
        hashed = get_password_hash("actual_password")
        assert verify_password("", hashed) is False

    def test_verify_password_case_sensitive(self):
        """Test that password verification is case sensitive."""
        password = "CaseSensitive"
        hashed = get_password_hash(password)
        assert verify_password("casesensitive", hashed) is False
        assert verify_password("CASESENSITIVE", hashed) is False
        assert verify_password(password, hashed) is True

    def test_roundtrip_hash_verify(self):
        """Test a full hash-then-verify roundtrip with various passwords."""
        passwords = [
            "simple",
            "with spaces and special chars!@#",
            "unicode_тест",
            "1234567890",
            "a" * 100,  # Long password
        ]
        for pwd in passwords:
            hashed = get_password_hash(pwd)
            assert verify_password(pwd, hashed) is True


class TestJWTTokens:
    """Tests for JWT access token creation and decoding."""

    def test_create_access_token_returns_string(self, test_settings: Settings):
        """Test that create_access_token returns a non-empty string."""
        token = create_access_token({"sub": "user@example.com"}, test_settings)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_is_jwt_format(self, test_settings: Settings):
        """Test that the token has three dot-separated segments (JWT format)."""
        token = create_access_token({"sub": "user@example.com"}, test_settings)
        parts = token.split(".")
        assert len(parts) == 3, "JWT should have header.payload.signature format"

    def test_decode_access_token_returns_payload(self, test_settings: Settings):
        """Test that decoding a valid token returns the payload dict."""
        payload = {"sub": "test@example.com", "role": "user"}
        token = create_access_token(payload, test_settings)
        decoded = decode_access_token(token, test_settings)

        assert decoded is not None
        assert decoded["sub"] == "test@example.com"
        assert decoded["role"] == "user"

    def test_decode_access_token_contains_exp(self, test_settings: Settings):
        """Test that decoded payload contains an expiration claim."""
        token = create_access_token({"sub": "user@example.com"}, test_settings)
        decoded = decode_access_token(token, test_settings)

        assert decoded is not None
        assert "exp" in decoded

    def test_decode_invalid_token_returns_none(self, test_settings: Settings):
        """Test that decoding a tampered/invalid token returns None."""
        result = decode_access_token("not.a.valid.jwt.token", test_settings)
        assert result is None

    def test_decode_empty_token_returns_none(self, test_settings: Settings):
        """Test that decoding an empty string returns None."""
        result = decode_access_token("", test_settings)
        assert result is None

    def test_decode_token_wrong_secret_returns_none(self, test_settings: Settings):
        """Test that a token signed with a different secret cannot be decoded."""
        token = create_access_token({"sub": "user@example.com"}, test_settings)

        wrong_settings = Settings(
            secret_key="completely-different-secret-key-x",
            algorithm="HS256",
            access_token_expire_minutes=30,
            environment="development",
        )
        result = decode_access_token(token, wrong_settings)
        assert result is None

    def test_create_access_token_with_custom_expiry(self, test_settings: Settings):
        """Test creating a token with a custom expiry delta."""
        expires = timedelta(hours=1)
        token = create_access_token(
            {"sub": "user@example.com"}, test_settings, expires_delta=expires
        )
        decoded = decode_access_token(token, test_settings)

        assert decoded is not None
        assert decoded["sub"] == "user@example.com"

    def test_create_access_token_preserves_all_payload_fields(
        self, test_settings: Settings
    ):
        """Test that all payload fields survive the encode-decode roundtrip."""
        payload = {
            "sub": "user@example.com",
            "role": "admin",
            "org_id": "42",
            "custom_field": "value",
        }
        token = create_access_token(payload, test_settings)
        decoded = decode_access_token(token, test_settings)

        assert decoded is not None
        for key, value in payload.items():
            assert decoded[key] == value

    def test_different_users_get_different_tokens(self, test_settings: Settings):
        """Test that two different users get distinct tokens."""
        token1 = create_access_token({"sub": "user1@example.com"}, test_settings)
        token2 = create_access_token({"sub": "user2@example.com"}, test_settings)
        assert token1 != token2


class TestAPIKeyGeneration:
    """Tests for API key generation and verification."""

    def test_generate_api_key_returns_string(self):
        """Test that generate_api_key returns a string."""
        key = generate_api_key()
        assert isinstance(key, str)

    def test_generate_api_key_has_prefix(self):
        """Test that generated API keys have the expected 'cliniq_' prefix."""
        key = generate_api_key()
        assert key.startswith("cliniq_"), (
            f"API key {key!r} does not start with 'cliniq_'"
        )

    def test_generate_api_key_minimum_length(self):
        """Test that generated API keys have a reasonable minimum length."""
        key = generate_api_key()
        # 'cliniq_' (7) + base64(32 bytes) ≈ 43 chars → total ≈ 50
        assert len(key) >= 40, f"API key too short: {len(key)} characters"

    def test_generate_api_key_uniqueness(self):
        """Test that two generated API keys are unique."""
        keys = {generate_api_key() for _ in range(20)}
        # Very high probability all 20 are unique with a 32-byte random token
        assert len(keys) == 20, "Generated API keys are not unique"

    def test_hash_api_key_returns_string(self):
        """Test that hash_api_key returns a non-empty string."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_api_key_not_plaintext(self):
        """Test that the API key hash is not the plaintext key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert hashed != key

    def test_verify_api_key_correct(self):
        """Test that verify_api_key returns True for the correct key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_incorrect(self):
        """Test that verify_api_key returns False for a wrong key."""
        key = generate_api_key()
        hashed = hash_api_key(key)
        wrong_key = generate_api_key()  # A different key
        assert verify_api_key(wrong_key, hashed) is False

    def test_api_key_roundtrip(self):
        """Test full generate → hash → verify roundtrip."""
        key = generate_api_key()
        hashed = hash_api_key(key)

        assert verify_api_key(key, hashed) is True
        assert verify_api_key("wrong_key", hashed) is False
