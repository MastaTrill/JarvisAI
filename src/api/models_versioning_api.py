"""
Model Versioning and Rollback Endpoints for Jarvis
- Register new model version
- List all versions
- Rollback to previous version
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from src.infra.db_config import get_db
from src.ml.models_registry import ModelRegistry
from src.infra.models_user import User
from src.infra.auth_helpers import admin_required
from datetime import datetime, timezone

router = APIRouter(prefix="/models/versioning", tags=["Model Versioning"])


@router.post("/register")
def register_version(
    name: str,
    version: str,
    description: str = "",
    device: str = "cpu",
    parent_id: int = None,
    db: Session = Depends(get_db),
    _current_user: User = Depends(admin_required),
):
    model = ModelRegistry(
        name=name,
        version=version,
        description=description,
        device=device,
        created_at=datetime.now(timezone.utc),
        parent_id=parent_id,
    )
    db.add(model)
    try:
        db.commit()
        db.refresh(model)
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.name == name, ModelRegistry.version == version)
            .first()
        )
        if existing:
            return {
                "message": f"Model '{name}' version '{version}' already registered.",
                "model": existing.id,
            }
        raise HTTPException(status_code=409, detail="Could not register model version")
    return {
        "message": f"Model '{name}' version '{version}' registered.",
        "model": model.id,
    }


@router.get("/list/{name}")
def list_versions(
    name: str,
    db: Session = Depends(get_db),
    _current_user: User = Depends(admin_required),
):
    versions = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.name == name)
        .order_by(ModelRegistry.created_at.desc())
        .all()
    )
    return [
        {
            "id": m.id,
            "version": m.version,
            "registered_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in versions
    ]


@router.post("/rollback/{model_id}")
def rollback_version(
    model_id: int,
    db: Session = Depends(get_db),
    _current_user: User = Depends(admin_required),
):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model or not model.parent_id:
        raise HTTPException(
            status_code=404, detail="No previous version to rollback to."
        )
    parent = db.query(ModelRegistry).filter(ModelRegistry.id == model.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent version not found.")
    model.active = False
    parent.active = True
    db.commit()
    return {
        "message": f"Rolled back to version '{parent.version}' for model '{parent.name}'."
    }
