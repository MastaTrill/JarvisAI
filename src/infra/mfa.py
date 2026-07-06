"""
MFA/TOTP utilities for JarvisAI authentication system
"""
import pyotp
import qrcode
import io
import base64
import secrets
import string
from typing import Tuple, Optional, List

# NOTE: Database imports are commented out to avoid initialization issues
# In a full implementation, these would be used for persistent storage
# from sqlalchemy.orm import Session
# from fastapi import HTTPException, status

# from src.infra.database_models import User
# from src.infra.database import get_db


def generate_mfa_secret() -> str:
    """Generate a new random secret for TOTP"""
    return pyotp.random_base32()


def get_totp_uri(email: str, secret: str, issuer: str = "JarvisAI") -> str:
    """
    Generate the TOTP provisioning URI for QR code generation
    
    Args:
        email: User's email address
        secret: TOTP secret key
        issuer: Service name (defaults to "JarvisAI")
    
    Returns:
        otpauth:// URL for QR code
    """
    return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name=issuer)


def verify_totp(secret: str, token: str, window: int = 1) -> bool:
    """
    Verify a TOTP token against the secret
    
    Args:
        secret: User's TOTP secret
        token: 6-digit code from user
        window: Number of time steps to check (default 1 allows for clock skew)
    
    Returns:
        True if token is valid, False otherwise
    """
    try:
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    except Exception:
        return False


def generate_backup_codes(count: int = 10) -> List[str]:
    """
    Generate backup codes for account recovery
    
    Args:
        count: Number of backup codes to generate
    
    Returns:
        List of 8-character alphanumeric codes
    """
    return [''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8)) for _ in range(count)]


# Database-related functions would go here in a full implementation
# For now, we provide stubs that indicate what would be implemented

def setup_mfa_for_user_stub() -> dict:
    """
    Stub for MFA setup function - in full implementation would:
    1. Generate secret for user
    2. Store in database
    3. Generate QR code
    4. Generate backup codes
    5. Return provisioning info
    """
    # This is a stub showing what the function would do
    secret = generate_mfa_secret()
    # In real implementation: save secret to user record in DB
    return {
        "secret": secret,
        "qr_code": "data:image/png;base64,PLACEHOLDER",  # Would be real QR code
        "backup_codes": generate_backup_codes(),
        "manual_entry_key": secret
    }


def verify_mfa_token_stub(user_has_mfa: bool, token: str, secret: str = None) -> bool:
    """
    Stub for MFA token verification
    
    Args:
        user_has_mfa: Whether user has MFA enabled
        token: TOTP token from user
        secret: User's TOTP secret (if MFA enabled)
    
    Returns:
        True if token is valid or MFA not required
    """
    if not user_has_mfa or not secret:
        # MFA not enabled for user
        return True
    
    return verify_totp(secret, token)