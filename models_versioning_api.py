"""
Model Versioning and Rollback Endpoints for Jarvis
- Register new model version
- List all versions
- Rollback to previous version
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from models_user import User
from api import get_current_user
from datetime import datetime

router = APIRouter(prefix="/models/versioning", tags=["Model Versioning"])

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

@router.post("/register")
def register_version(name: str, version: str, description: str = "", device: str = "cpu", parent_id: int = None, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    model = ModelRegistry(
        name=name,
        version=version,
        description=description,
        device=device,
        registered_at=datetime.utcnow(),
        parent_id=parent_id
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return {"message": f"Model '{name}' version '{version}' registered.", "model": model.id}

@router.get("/list/{name}")
def list_versions(name: str, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    versions = db.query(ModelRegistry).filter(ModelRegistry.name == name).order_by(ModelRegistry.registered_at.desc()).all()
    return [{"id": m.id, "version": m.version, "registered_at": m.registered_at.isoformat()} for m in versions]

@router.post("/rollback/{model_id}")
def rollback_version(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model or not model.parent_id:
        raise HTTPException(status_code=404, detail="No previous version to rollback to.")
    parent = db.query(ModelRegistry).filter(ModelRegistry.id == model.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent version not found.")
    model.active = False
    parent.active = True
    db.commit()
    return {"message": f"Rolled back to version '{parent.version}' for model '{parent.name}'."}
