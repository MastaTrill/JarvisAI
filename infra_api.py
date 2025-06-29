"""
Self-Healing & Auto-Scaling Endpoints
- Health check for all model endpoints
- Trigger scale up/down (stub)
- List model endpoint status
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from models_user import User
from api import get_current_user
from datetime import datetime
import random

router = APIRouter(prefix="/infra", tags=["Self-Healing & Auto-Scaling"])

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

@router.get("/health")
def health_all_models(db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    models = db.query(ModelRegistry).all()
    # Stub: In production, check real health endpoints
    return [{"model_id": m.id, "name": m.name, "status": random.choice(["healthy", "unhealthy"]) } for m in models]

@router.post("/scale/{model_id}")
def scale_model(model_id: int, replicas: int, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    # Stub: In production, trigger K8s or infra scaling
    return {"message": f"Scaling model {model_id} to {replicas} replicas (stub)"}

@router.get("/status/{model_id}")
def model_status(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    # Stub: In production, return real status
    return {"model_id": model_id, "status": random.choice(["healthy", "unhealthy", "scaling"]) }
