"""
A/B Testing Framework for Jarvis AI.

Supports:
- Creating experiments with multiple variants
- Assigning users to variants (sticky via session/user ID)
- Tracking conversion events and metrics
- Statistical significance calculation
- Admin controls for starting/stopping experiments
"""

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import Session

from db_config import Base as ConfigBase
from database import get_db
from auth_helpers import admin_required, get_current_user
from models_user import User


# --- Database Models ---

class ABExperiment(ConfigBase):
    __tablename__ = "ab_experiments"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    status = Column(String(20), default="draft")  # draft, running, paused, completed
    variants = Column(JSON, nullable=False)  # [{"name": "control", "weight": 0.5}, ...]
    traffic_allocation = Column(Float, default=1.0)  # 0.0-1.0, % of traffic included
    primary_metric = Column(String(100), default="conversion")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_by = Column(String(100))
    winner = Column(String(100), nullable=True)


class ABEvent(ConfigBase):
    __tablename__ = "ab_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    variant = Column(String(100), nullable=False)
    event_type = Column(String(50), nullable=False)  # exposure, conversion, custom
    event_name = Column(String(100), default="")
    event_value = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# --- Pydantic Schemas ---

class VariantSpec(BaseModel):
    name: str
    weight: float = 0.5
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    variants: List[VariantSpec]
    traffic_allocation: float = 1.0
    primary_metric: str = "conversion"


class EventTrack(BaseModel):
    experiment_id: str
    event_type: str = "conversion"  # exposure, conversion, custom
    event_name: str = ""
    event_value: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: str
    variants: List[Dict[str, Any]]
    primary_metric: str
    created_at: str


# --- Router ---

router = APIRouter(prefix="/experiments", tags=["A/B Testing"])


@router.post("/create", response_model=ExperimentResponse)
def create_experiment(
    body: ExperimentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Create a new A/B experiment."""
    import uuid

    if len(body.variants) < 2:
        raise HTTPException(400, "At least 2 variants required")

    total_weight = sum(v.weight for v in body.variants)
    if abs(total_weight - 1.0) > 0.01:
        raise HTTPException(400, f"Variant weights must sum to 1.0, got {total_weight}")

    exp_id = str(uuid.uuid4())[:8]
    exp = ABExperiment(
        id=exp_id,
        name=body.name,
        description=body.description,
        variants=[v.model_dump() for v in body.variants],
        traffic_allocation=body.traffic_allocation,
        primary_metric=body.primary_metric,
        status="draft",
        created_by=current_user.username,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _exp_to_response(exp)


@router.post("/{experiment_id}/start")
def start_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Start a draft experiment."""
    exp = db.query(ABExperiment).filter_by(id=experiment_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status not in ("draft", "paused"):
        raise HTTPException(400, f"Cannot start experiment in '{exp.status}' state")
    exp.status = "running"
    exp.started_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": f"Experiment '{exp.name}' started", "id": exp.id}


@router.post("/{experiment_id}/stop")
def stop_experiment(
    experiment_id: str,
    winner: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Stop a running experiment and optionally declare a winner."""
    exp = db.query(ABExperiment).filter_by(id=experiment_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status != "running":
        raise HTTPException(400, f"Experiment is '{exp.status}', not running")
    exp.status = "completed"
    exp.ended_at = datetime.now(timezone.utc)
    exp.winner = winner
    db.commit()
    return {"message": f"Experiment '{exp.name}' stopped", "winner": winner}


@router.get("/list")
def list_experiments(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all experiments, optionally filtered by status."""
    query = db.query(ABExperiment)
    if status:
        query = query.filter_by(status=status)
    experiments = query.order_by(ABExperiment.created_at.desc()).all()
    return [_exp_to_response(e) for e in experiments]


@router.get("/{experiment_id}/results")
def experiment_results(
    experiment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed results for an experiment including per-variant metrics."""
    exp = db.query(ABExperiment).filter_by(id=experiment_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")

    events = db.query(ABEvent).filter_by(experiment_id=experiment_id).all()

    variant_names = [v["name"] for v in exp.variants]
    results = {}
    for name in variant_names:
        variant_events = [e for e in events if e.variant == name]
        exposures = [e for e in variant_events if e.event_type == "exposure"]
        conversions = [e for e in variant_events if e.event_type == "conversion"]
        conversion_values = [e.event_value for e in conversions if e.event_value is not None]

        results[name] = {
            "exposures": len(exposures),
            "conversions": len(conversions),
            "conversion_rate": len(conversions) / len(exposures) if exposures else 0.0,
            "avg_value": sum(conversion_values) / len(conversion_values) if conversion_values else 0.0,
            "total_value": sum(conversion_values),
        }

    # Calculate statistical significance (z-test for proportions) between first two variants
    significance = None
    if len(variant_names) >= 2:
        v1, v2 = variant_names[0], variant_names[1]
        r1, r2 = results[v1], results[v2]
        if r1["exposures"] > 0 and r2["exposures"] > 0:
            p1 = r1["conversion_rate"]
            p2 = r2["conversion_rate"]
            n1 = r1["exposures"]
            n2 = r2["exposures"]
            p_pool = (r1["conversions"] + r2["conversions"]) / (n1 + n2)
            se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 0
            z = (p1 - p2) / se if se > 0 else 0
            significance = {
                "z_score": round(z, 4),
                "p_value": round(2 * (1 - _normal_cdf(abs(z))), 6),
                "significant_at_95": abs(z) > 1.96,
                "lift": round((p1 - p2) / p2 * 100, 2) if p2 > 0 else None,
            }

    return {
        "experiment": _exp_to_response(exp),
        "results": results,
        "significance": significance,
        "total_events": len(events),
    }


@router.post("/track")
def track_event(
    body: EventTrack,
    request,
    db: Session = Depends(get_db),
):
    """Track an event for an experiment. Called by client-side or server-side."""
    user_id = _get_user_id(request)

    event = ABEvent(
        experiment_id=body.experiment_id,
        user_id=user_id,
        variant="",
        event_type=body.event_type,
        event_name=body.event_name,
        event_value=body.event_value,
        metadata_json=body.metadata,
    )

    # Auto-assign variant if experiment is running
    exp = db.query(ABExperiment).filter_by(id=body.experiment_id).first()
    if exp and exp.status == "running":
        event.variant = _assign_variant(user_id, exp)

    db.add(event)
    db.commit()
    return {"message": "Event tracked", "variant": event.variant}


@router.get("/{experiment_id}/variant")
def get_variant(
    experiment_id: str,
    request,
    db: Session = Depends(get_db),
):
    """Get the assigned variant for the current user in an experiment."""
    user_id = _get_user_id(request)
    exp = db.query(ABExperiment).filter_by(id=experiment_id).first()
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status != "running":
        raise HTTPException(400, f"Experiment is '{exp.status}'")

    variant = _assign_variant(user_id, exp)

    # Auto-track exposure
    existing = (
        db.query(ABEvent)
        .filter_by(experiment_id=experiment_id, user_id=user_id, event_type="exposure")
        .first()
    )
    if not existing:
        db.add(
            ABEvent(
                experiment_id=experiment_id,
                user_id=user_id,
                variant=variant,
                event_type="exposure",
            )
        )
        db.commit()

    variant_config = {}
    for v in exp.variants:
        if v["name"] == variant:
            variant_config = v.get("config", {})
            break

    return {
        "experiment_id": experiment_id,
        "variant": variant,
        "config": variant_config,
    }


# --- Helpers ---

def _get_user_id(request) -> str:
    """Extract a stable user ID from the request."""
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "username"):
        return str(user.username)
    # Fall back to IP-based anonymous ID
    client_ip = request.client.host if request.client else "unknown"
    return hashlib.sha256(client_ip.encode()).hexdigest()[:16]


def _assign_variant(user_id: str, exp: ABExperiment) -> str:
    """Deterministically assign a user to a variant based on hash."""
    hash_input = f"{exp.id}:{user_id}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

    # Check traffic allocation
    if (hash_val % 10000) / 10000 > exp.traffic_allocation:
        return "control"

    # Weighted variant selection
    hash_val = int(hashlib.sha256(f"{hash_input}:variant".encode()).hexdigest(), 16)
    point = (hash_val % 10000) / 10000.0

    cumulative = 0.0
    for v in exp.variants:
        cumulative += v["weight"]
        if point <= cumulative:
            return v["name"]
    return exp.variants[-1]["name"]


def _exp_to_response(exp: ABExperiment) -> dict:
    return {
        "id": exp.id,
        "name": exp.name,
        "status": exp.status,
        "variants": exp.variants,
        "primary_metric": exp.primary_metric,
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "ended_at": exp.ended_at.isoformat() if exp.ended_at else None,
        "winner": exp.winner,
    }


def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
