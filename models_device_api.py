"""
GPU/Accelerator-Aware Model Serving Endpoints
- List available devices (CPU/GPU)
- Register model with device
- Route inference to device
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from models_user import User
from api import get_current_user
import os

router = APIRouter(prefix="/models/device", tags=["Model Device Serving"])

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

@router.get("/available")
def list_devices():
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {"devices": devices}

@router.post("/register")
def register_model_device(model_id: int, device: str, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.device = device
    db.commit()
    return {"message": f"Model {model_id} set to device {device}"}

@router.post("/infer")
def infer_on_device(model_id: int, data: dict, db: Session = Depends(get_db)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    # Stub: Route to correct device (cpu/gpu)
    return {"message": f"Inference for model {model_id} on device {model.device}", "result": None}
