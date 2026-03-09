"""Tests for authentication module — token creation, verification, password hashing."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from datetime import timedelta
from authentication import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    create_password_reset_token,
    verify_password_reset_token,
    SECRET_KEY,
    ALGORITHM,
)
import jwt


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "mysecretpassword"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)

    def test_different_hashes_for_same_password(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # bcrypt uses random salt


class TestJWTTokens:
    def test_create_and_decode_access_token(self):
        token = create_access_token(data={"sub": "testuser"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["type"] == "access"
        assert "exp" in payload

    def test_access_token_custom_expiry(self):
        token = create_access_token(
            data={"sub": "user"}, expires_delta=timedelta(minutes=5)
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "user"

    def test_create_and_decode_refresh_token(self):
        token = create_refresh_token(data={"sub": "testuser"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["type"] == "refresh"

    def test_decode_invalid_token_raises(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_decode_expired_token_raises(self):
        from fastapi import HTTPException

        token = create_access_token(
            data={"sub": "user"}, expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)
        assert exc_info.value.status_code == 401


class TestAPIKey:
    def test_generate_api_key_unique(self):
        k1 = generate_api_key()
        k2 = generate_api_key()
        assert k1 != k2
        assert len(k1) > 20


class TestPasswordResetToken:
    def test_create_and_verify_reset_token(self):
        email = "test@example.com"
        token = create_password_reset_token(email)
        result = verify_password_reset_token(token)
        assert result == email

    def test_invalid_reset_token_returns_none(self):
        result = verify_password_reset_token("invalid.token")
        assert result is None
