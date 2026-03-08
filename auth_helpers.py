"""
Authentication helper functions for FastAPI JWT authentication and role-based access control.
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

try:
    from jose import JWTError, jwt
except ImportError:
    raise ImportError(
        "python-jose is not installed. Please install it with 'pip install python-jose'"
    )


SECRET_KEY = "your-secret-key"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(lambda: __import__('models_user').get_db())
):
    # Import User here to avoid circular import
    User = __import__('models_user').User
    """
    Retrieve the current user from the JWT token.
    """
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

    def role_checker(user = Depends(get_current_user)):
        # Check if user.role exists and is not a SQLAlchemy Column object
        role = getattr(user, "role", None)
        role_name = getattr(role, "name", None) if role else None
        if role_name != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return role_checker
