# User update endpoint for admin
@router.post("/update")
def update_user(user_id: int, email: str = None, role: str = None, is_active: str = None, db: Session = Depends(get_db), admin: User = Depends(require_role("admin"))):
    from audit_trail import log_audit_event
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    changes = []
    if email and user.email != email:
        changes.append(f"email: {user.email} -> {email}")
        user.email = email
    if role:
        from sqlalchemy.orm.exc import NoResultFound
        try:
            role_obj = db.query(Role).filter(Role.name == role).one()
        except NoResultFound:
            role_obj = Role(name=role, description="")
            db.add(role_obj)
            db.commit()
            db.refresh(role_obj)
        if not user.role or user.role.name != role:
            changes.append(f"role: {user.role.name if user.role else None} -> {role}")
        user.role = role_obj
    if is_active is not None:
        new_active = (is_active.lower() == "true")
        if user.is_active != new_active:
            changes.append(f"is_active: {user.is_active} -> {new_active}")
        user.is_active = new_active
    db.commit()
    if changes:
        log_audit_event(admin.username, "update_user", f"user_id={user_id}", "; ".join(changes))
    return {"message": "User updated"}
"""
User and Role Models for Jarvis AI
- SQLAlchemy ORM models for users and roles
- For use with persistent database (see db_config.py)
"""
from db_config import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship

class Role(Base):
    __tablename__ = "roles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    users = relationship("User", back_populates="role")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    role_id = Column(Integer, ForeignKey("roles.id"))
    role = relationship("Role", back_populates="users")


# Password hashing helpers
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)


# Registration, login, and RBAC logic
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from db_config import SessionLocal
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "your-secret-key"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

router = APIRouter(prefix="/users", tags=["Users"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/register")
def register_user(username: str, password: str, email: str, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=400, detail="Username or email already registered")
    hashed_pw = get_password_hash(password)
    user = User(username=username, hashed_password=hashed_pw, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered", "user_id": user.id}

@router.post("/login")
def login_user(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token({"sub": user.username, "role": user.role.name if user.role else "user"})
    return {"access_token": access_token, "token_type": "bearer"}

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def require_role(required_role: str):
    def role_checker(user: User = Depends(get_current_user)):
        if not user.role or user.role.name != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

# Admin dashboard scaffolding (API endpoints)
@router.get("/admin/users")
def list_users(db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    users = db.query(User).all()
    return [
        {"id": u.id, "username": u.username, "email": u.email, "role": u.role.name if u.role else None, "is_active": u.is_active}
        for u in users
    ]

@router.get("/admin/models")
def list_models(db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    from models_registry import ModelRegistry
    models = db.query(ModelRegistry).all()
    return [
        {"id": m.id, "name": m.name, "description": m.description, "accuracy": m.accuracy}
        for m in models
    ]

@router.get("/admin/jobs")
def list_jobs(db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    from jobs_persistent import Job
    jobs = db.query(Job).all()
    return [
        {"id": j.id, "job_id": j.job_id, "status": j.status, "result": j.result, "completed_at": j.completed_at, "cancelled": j.cancelled}
        for j in jobs
    ]
