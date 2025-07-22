"""
Model and Job Management Endpoints for Admin Dashboard (scaffold)
- List, create, update, delete models and jobs
- Admin-only (RBAC)
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from jobs_persistent import Job
from models_user import User
from api import get_current_user
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/admin/api", tags=["Admin"])

# --- Pydantic Schemas (minimal, adjust as needed) ---
class ModelRegistryCreate(BaseModel):
    name: str
    description: str = ""
    # ...add other fields as needed...

class ModelRegistryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    # ...add other fields as needed...

class JobCreate(BaseModel):
    name: str
    status: str = "pending"
    # ...add other fields as needed...

class JobUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    # ...add other fields as needed...

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

# --- Model Endpoints ---

@router.post("/models")
def create_model(
    model: ModelRegistryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_model = ModelRegistry(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.put("/models/{model_id}")
def update_model(
    model_id: int,
    model: ModelRegistryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    for field, value in model.dict(exclude_unset=True).items():
        setattr(db_model, field, value)
    db.commit()
    db.refresh(db_model)
    return db_model

@router.delete("/models/{model_id}")
def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(db_model)
    db.commit()
    return {"detail": "Model deleted"}

# --- Job Endpoints ---

@router.post("/jobs")
def create_job(
    job: JobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_job = Job(**job.dict())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job

@router.put("/jobs/{job_id}")
def update_job(
    job_id: int,
    job: JobUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_job = db.query(Job).filter(Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    for field, value in job.dict(exclude_unset=True).items():
        setattr(db_job, field, value)
    db.commit()
    db.refresh(db_job)
    return db_job

@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required)
):
    db_job = db.query(Job).filter(Job.id == job_id).first()
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.delete(db_job)
    db.commit()
    return {"detail": "Job deleted"}
