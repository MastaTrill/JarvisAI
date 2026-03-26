"""
Authentication helper functions for FastAPI JWT authentication and role-based access control.
"""

import os
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

try:
    import jwt
    from jwt import InvalidTokenError as JWTError
except ImportError as exc:
    raise ImportError(
        "PyJWT is not installed. Please install it with 'pip install PyJWT'"
    ) from exc


from database import get_db as _get_user_db

SECRET_KEY = os.environ.get("JARVIS_SECRET_KEY", "")
if not SECRET_KEY:
    # Import from the canonical source so all modules share the same key
    from authentication import SECRET_KEY  # noqa: F811
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(_get_user_db),
):
    """Retrieve the current user from the JWT token."""
    # Import User here to avoid circular import
    User = __import__("models_user").User
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def require_role(required_role: str):
    """
    Dependency to require a specific user role.
    """

    def role_checker(user=Depends(get_current_user)):
        # Check if user.role exists and is not a SQLAlchemy Column object
        role = getattr(user, "role", None)
        role_name = getattr(role, "name", role) if role else None
        if role_name != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return role_checker


def admin_required(current_user=Depends(get_current_user)):
    """Verify that the current user has admin role."""
    role = getattr(current_user, "role", None)
    role_name = getattr(role, "name", role) if role else None
    if role_name != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
