"""
External Model Server Integration Endpoints
- Register external model server endpoint
- Health check for external server
- Reload external model
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_registry import ModelRegistry
from models_user import User
from api import get_current_user
import requests

router = APIRouter(prefix="/models/external", tags=["External Model Server"])

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
def register_external(model_id: int, endpoint: str, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.external_endpoint = endpoint
    db.commit()
    return {"message": f"External endpoint set for model {model_id}"}

@router.get("/health/{model_id}")
def check_external_health(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model or not model.external_endpoint:
        raise HTTPException(status_code=404, detail="External endpoint not set")
    try:
        resp = requests.get(f"{model.external_endpoint}/health", timeout=3)
        return {"status": resp.status_code, "response": resp.json()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@router.post("/reload/{model_id}")
def reload_external_model(model_id: int, db: Session = Depends(get_db), current_user: User = Depends(admin_required)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model or not model.external_endpoint:
        raise HTTPException(status_code=404, detail="External endpoint not set")
    try:
        resp = requests.post(f"{model.external_endpoint}/reload", timeout=5)
        return {"status": resp.status_code, "response": resp.json()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
