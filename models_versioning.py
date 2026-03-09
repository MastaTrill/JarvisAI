"""
Model Versioning and Advanced Serving for Jarvis AI
- Track multiple versions of each model
- Support rollback and promotion
- Placeholder for GPU inference and external model server integration
"""

from db_config import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime, timezone


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    version = Column(String, index=True)
    path = Column(String)
    accuracy = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=False)


from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db_config import get_db

router = APIRouter(prefix="/model_versions", tags=["Model Versioning"])


@router.get("/list/{model_name}")
def list_versions(model_name: str, db: Session = Depends(get_db)):
    versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_name == model_name)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )
    return [
        {
            "id": v.id,
            "model_name": v.model_name,
            "version": v.version,
            "accuracy": v.accuracy,
            "created_at": v.created_at,
            "is_active": v.is_active,
        }
        for v in versions
    ]


# Accept POST and GET for rollback/promote for dashboard form compatibility
from fastapi import Request


@router.post("/rollback/{model_name}/{version}")
@router.get("/rollback/{model_name}/{version}")
def rollback_version(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    request: Request = None,
):
    target = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_name == model_name, ModelVersion.version == version)
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="Version not found")
    db.query(ModelVersion).filter(ModelVersion.model_name == model_name).update(
        {ModelVersion.is_active: False}
    )
    target.is_active = True
    db.commit()
    if request and request.method == "GET":
        return {"message": f"Rolled back {model_name} to version {version}"}
    return {"message": f"Rolled back {model_name} to version {version}"}


@router.post("/promote/{model_name}/{version}")
@router.get("/promote/{model_name}/{version}")
def promote_version(
    model_name: str,
    version: str,
    db: Session = Depends(get_db),
    request: Request = None,
):
    target = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_name == model_name, ModelVersion.version == version)
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="Version not found")
    db.query(ModelVersion).filter(ModelVersion.model_name == model_name).update(
        {ModelVersion.is_active: False}
    )
    target.is_active = True
    db.commit()
    if request and request.method == "GET":
        return {"message": f"Promoted {model_name} version {version} to active"}
    return {"message": f"Promoted {model_name} version {version} to active"}


# GPU inference support (PyTorch example)
def is_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run_inference(model, input_data):
    """
    Example inference function that uses GPU if available (PyTorch).
    """
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_data = input_data.to(device)
        with torch.no_grad():
            output = model(input_data)
        return output.cpu()
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU inference support.") from exc


# Model serving integration scaffolds
import requests


def predict_with_tensorflow_serving(
    model_name, data, tf_serving_url="http://localhost:8501/v1/models"
):
    """
    Send prediction request to TensorFlow Serving REST API.
    """
    url = f"{tf_serving_url}/{model_name}:predict"
    response = requests.post(url, json={"instances": data}, timeout=30)
    response.raise_for_status()
    return response.json()


def predict_with_torchserve(
    model_name, data, torchserve_url="http://localhost:8080/predictions"
):
    """
    Send prediction request to TorchServe REST API.
    """
    url = f"{torchserve_url}/{model_name}"
    response = requests.post(url, data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def predict_with_triton(model_name, data, triton_url="http://localhost:8000/v2/models"):
    """
    Send prediction request to NVIDIA Triton Inference Server REST API.
    """
    url = f"{triton_url}/{model_name}/infer"
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()
