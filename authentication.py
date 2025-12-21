"""
JarvisAI Authentication System
JWT-based authentication with role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session
import hashlib
import secrets

from database import get_db
from database_models import User

# Security configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    username: str = payload.get("sub")
    token_type: str = payload.get("type")
    
    if username is None or token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user


def get_current_user_from_api_key(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from API key"""
    if api_key is None:
        return None
    
    user = db.query(User).filter(User.api_key == api_key).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Role-based access control
class RoleChecker:
    """Check if user has required role"""
    
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_active_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return current_user


# Pre-configured role checkers
require_creator = RoleChecker(["creator"])
require_family = RoleChecker(["creator", "family"])
require_admin = RoleChecker(["creator", "family", "admin"])
require_user = RoleChecker(["creator", "family", "admin", "user"])


def authenticate_user(username: str, password: str, db: Session) -> Optional[User]:
    """Authenticate user with username and password"""
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if not user.is_active:
        return None
    
    return user


def generate_api_key() -> str:
    """Generate a new API key"""
    return secrets.token_urlsafe(32)


def revoke_api_key(user_id: str, db: Session):
    """Revoke user's API key"""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.api_key = generate_api_key()
        db.commit()
        return user.api_key
    return None


# Password reset functionality
def create_password_reset_token(email: str) -> str:
    """Create password reset token"""
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode = {"email": email, "exp": expire, "type": "password_reset"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_type: str = payload.get("type")
        
        if token_type != "password_reset":
            return None
        
        email: str = payload.get("email")
        return email
    except JWTError:
        return None


def reset_password(token: str, new_password: str, db: Session) -> bool:
    """Reset user password using reset token"""
    email = verify_password_reset_token(token)
    if not email:
        return False
    
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return False
    
    user.hashed_password = hash_password(new_password)
    db.commit()
    return True


# User permissions
class Permission:
    """Permission constants"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    CREATOR = "creator"


def has_permission(user: User, permission: str) -> bool:
    """Check if user has specific permission"""
    role_permissions = {
        "creator": [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN, Permission.CREATOR],
        "family": [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
        "admin": [Permission.READ, Permission.WRITE, Permission.DELETE],
        "user": [Permission.READ, Permission.WRITE]
    }
    
    user_permissions = role_permissions.get(user.role, [])
    return permission in user_permissions


def check_permission(user: User, permission: str):
    """Check permission and raise exception if not authorized"""
    if not has_permission(user, permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied. Required permission: {permission}"
        )


if __name__ == "__main__":
    # Test authentication system
    print("🔐 JarvisAI Authentication System")
    print(f"✅ Secret Key Generated: {SECRET_KEY[:20]}...")
    print(f"✅ Algorithm: {ALGORITHM}")
    print(f"✅ Token Expiry: {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    
    # Test token creation
    test_data = {"sub": "creator", "role": "creator"}
    test_token = create_access_token(test_data)
    print(f"✅ Test Token: {test_token[:50]}...")
    
    # Test token decoding
    decoded = decode_token(test_token)
    print(f"✅ Decoded Token: {decoded}")
    
    # Test password hashing
    test_password = "Test123"
    hashed = hash_password(test_password)
    print(f"✅ Password Hash: {hashed[:20]}...")
    print(f"✅ Password Verified: {verify_password(test_password, hashed)}")
