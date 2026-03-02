"""User and Role Models and API endpoints for Jarvis AI."""

# Standard library imports
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
# from enum import Enum  # Enum is not used, so remove

# Third-party imports
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship, Session
try:
    from passlib.context import CryptContext
except ImportError:
    raise ImportError("passlib is not installed. Install with 'pip install passlib[bcrypt]'")

try:
    from jose import JWTError, jwt
except ImportError:
    raise ImportError("python-jose is not installed. Install with 'pip install python-jose'")

# First-party imports
from db_config import Base, SessionLocal

# Import exceptions at module level
from sqlalchemy.exc import NoResultFound


# Models
class Role(Base):
    """Role model for user roles."""

    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    users = relationship("User", back_populates="role")

    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self):
        return f"<Role(id={self.id}, name={self.name})>"

    def get_description(self):
        return self.description


class User(Base):
    """User model for Jarvis AI."""

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    role_id = Column(Integer, ForeignKey("roles.id"))
    role = relationship("Role", back_populates="users")

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary."""
        result = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "role": self.role.name if self.role else None,
        }
        if include_sensitive:
            result["hashed_password"] = self.hashed_password
        return result

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

    def get_email(self):
        return self.email


# Password hashing helpers
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


# FastAPI router
router = APIRouter(prefix="/users", tags=["Users"])


# Database dependency
def get_db():
    """Provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# JWT and authentication
SECRET_KEY = "your-secret-key"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not isinstance(username, str):
            raise credentials_exception
    except JWTError as exc:
        raise credentials_exception from exc
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def require_role(required_role: str):
    """Dependency to require a specific user role."""

    def role_checker(current_user: User = Depends(get_current_user)):
        if (
            not getattr(current_user, "role", None)
            or current_user.role.name != required_role
        ):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user

    return role_checker


# User update endpoint for admin
@router.post("/update")
def update_user(
    user_id: int,
    email: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """Update user details (admin only)."""
    from audit_trail import log_audit_event

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    changes = []
    if email is not None and (user.email != email if not hasattr(user.email, 'compare') else user.email.compare(email) is False):
        changes.append(f"email: {user.email} -> {email}")
        user.email = email
    if role:
        try:
            role_obj = db.query(Role).filter(Role.name == role).one()
        except NoResultFound:
            role_obj = Role(name=role, description="")
            db.add(role_obj)
            db.commit()
            db.refresh(role_obj)
        user_role = getattr(user, "role", None)
        if user_role is None or user_role.name != role:
            changes.append(f"role: {user_role.name if user_role else None} -> {role}")
        user.role = role_obj
    if is_active is not None:
        new_active = is_active.lower() == "true"
        if bool(user.is_active) != new_active:
            changes.append(f"is_active: {user.is_active} -> {new_active}")
        user.is_active = new_active
    db.commit()
    if changes:
        log_audit_event(
            admin.username, "update_user", f"user_id={user_id}", "; ".join(changes)
        )
    return {"message": "User updated"}


@router.post("/register")
def register_user(
    username: str, password: str, email: str, db: Session = Depends(get_db)
):
    """Register a new user."""
    existing_user = (
        db.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )
    if existing_user is not None:
        raise HTTPException(
            status_code=400, detail="Username or email already registered"
        )
    hashed_pw = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_pw, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered", "user_id": user.id}


@router.post("/login")
def login_user(username: str, password: str, db: Session = Depends(get_db)):
    """Login a user and return an access token."""
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, str(user.hashed_password)):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_role = getattr(user, "role", None)
    access_token = create_access_token(
        {"sub": user.username, "role": user_role.name if user_role else "user"}
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Admin dashboard scaffolding (API endpoints)
@router.get("/admin/users")
def list_users(
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role.name if u.role else None,
            "is_active": u.is_active,
        }
        for u in users
    ]


@router.get("/admin/models")
def list_models(
    db: Session = Depends(get_db), _current_user: User = Depends(require_role("admin"))
):
    """List all models (admin only)."""
    from models_registry import ModelRegistry
    
        models = db.query(ModelRegistry).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "accuracy": m.accuracy,
        }
        for m in models
    ]


@router.get("/admin/jobs")
def list_jobs(
    db: Session = Depends(get_db), _current_user: User = Depends(require_role("admin"))
):
    """List all jobs (admin only)."""
    from jobs_persistent import Job
    
        jobs = db.query(Job).all()
    return [
        {
            "id": j.id,
            "job_id": j.job_id,
            "status": j.status,
            "result": j.result,
            "completed_at": j.completed_at,
            "cancelled": j.cancelled,
        }
        for j in jobs
    ]
