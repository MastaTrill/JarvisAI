"""
Drift Detection Endpoints
- Set drift score for a model
- Get drift score
- Trigger drift monitoring (stub)
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import get_db
from models_registry import ModelRegistry
from models_user import User
from auth_helpers import admin_required

router = APIRouter(prefix="/models/drift", tags=["Drift Detection"])


@router.post("/set/{model_id}")
def set_drift_score(
    model_id: int,
    drift_score: float,
    db: Session = Depends(get_db),
    _current_user: User = Depends(admin_required),
):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.drift_score = drift_score
    db.commit()
    return {
        "message": f"Drift score set for model {model_id}",
        "drift_score": drift_score,
    }


@router.get("/get/{model_id}")
def get_drift_score(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "drift_score": model.drift_score}


@router.post("/monitor/{model_id}")
def trigger_drift_monitoring(
    model_id: int,
    _db: Session = Depends(get_db),
    _current_user: User = Depends(admin_required),
):
    # Stub: In production, trigger background drift monitoring job
    return {"message": f"Drift monitoring triggered for model {model_id}"}
