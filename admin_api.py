"""
Model and Job Management Endpoints for Admin Dashboard (scaffold)
- List, create, update, delete models and jobs
- Admin-only (RBAC)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from jobs_persistent import Job
from models_user import User
from api import get_current_user

router = APIRouter(prefix="/admin/api", tags=["Admin"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def admin_required(current_user: User = Depends(get_current_user)):
    if not getattr(current_user.role, 'name', None) == 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.get("/models")
def list_models(db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    return db.query(ModelRegistry).all()

@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    return db.query(Job).all()

# TODO: Add create/update/delete endpoints for models and jobs
