"""
Model Comparison System for Jarvis AI.

Provides side-by-side comparison of registered models including:
- Accuracy, F1, precision, recall metrics
- Inference speed
- Memory usage
- Training time
- Visual comparison data
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Float, DateTime, JSON

from db_config import Base as ConfigBase
from database import get_db
from auth_helpers import get_current_user
from models_user import User


# --- Database Model ---

class ModelComparison(ConfigBase):
    __tablename__ = "model_comparisons"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    model_names = Column(JSON, nullable=False)  # ["model_a", "model_b"]
    dataset = Column(String(500))
    metrics = Column(JSON)  # {"model_a": {"accuracy": 0.95, ...}, ...}
    winner = Column(String(200), nullable=True)
    notes = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))


# --- Pydantic Schemas ---

class ComparisonRequest(BaseModel):
    name: str = "Model Comparison"
    model_names: List[str]
    dataset: Optional[str] = None
    metrics_to_compare: List[str] = Field(
        default_factory=lambda: ["accuracy", "f1_score", "inference_time_ms"]
    )


# --- Router ---

router = APIRouter(prefix="/models/compare", tags=["Model Comparison"])


@router.post("/run")
def run_comparison(
    body: ComparisonRequest,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run a side-by-side comparison of registered models."""
    import uuid
    import numpy as np

    from main_api import app
    from database import SessionLocal as DBSession
    from database_models import ModelRun

    if len(body.model_names) < 2:
        raise HTTPException(400, "At least 2 models required for comparison")

    comparison_id = str(uuid.uuid4())[:8]
    results = {}
    model_objects = {}

    # Load models
    for name in body.model_names:
        model = app.state.models.get(name)
        if model:
            model_objects[name] = model

    # Generate test data for inference benchmarking
    test_input = np.random.randn(100, 10).tolist()

    for name in body.model_names:
        model_results = {}

        # Get historical metrics from DB
        session = DBSession()
        try:
            runs = (
                session.query(ModelRun)
                .filter_by(model_name=name, status="completed")
                .order_by(ModelRun.created_at.desc())
                .limit(5)
                .all()
            )
            if runs:
                model_results["accuracy"] = runs[0].accuracy
                model_results["f1_score"] = runs[0].f1_score
                model_results["precision"] = runs[0].precision_score
                model_results["recall"] = runs[0].recall_score
                model_results["training_time_s"] = runs[0].training_time
                model_results["avg_loss"] = runs[0].loss
                model_results["runs_count"] = len(runs)
        finally:
            session.close()

        # Live inference benchmark
        model = model_objects.get(name)
        if model and hasattr(model, "predict"):
            latencies = []
            for sample in test_input[:20]:
                start = time.perf_counter()
                try:
                    model.predict([sample])
                except Exception:
                    pass
                latencies.append((time.perf_counter() - start) * 1000)

            if latencies:
                model_results["inference_time_ms"] = round(
                    sum(latencies) / len(latencies), 3
                )
                model_results["inference_p95_ms"] = round(
                    sorted(latencies)[int(len(latencies) * 0.95)], 3
                )

        # Model metadata
        if model:
            model_results["model_type"] = type(model).__name__
            try:
                model_results["parameters"] = sum(
                    p.size for p in model.parameters()
                ) if hasattr(model, "parameters") else None
            except Exception:
                model_results["parameters"] = None

        results[name] = model_results

    # Determine winner based on accuracy (or first available metric)
    winner = None
    best_metric = body.metrics_to_compare[0] if body.metrics_to_compare else "accuracy"
    best_value = -float("inf")
    for name, metrics in results.items():
        val = metrics.get(best_metric)
        if val is not None and val > best_value:
            best_value = val
            winner = name

    # Save comparison
    comp = ModelComparison(
        id=comparison_id,
        name=body.name,
        model_names=body.model_names,
        dataset=body.dataset,
        metrics=results,
        winner=winner,
        created_by=current_user.username if current_user else "anonymous",
    )
    db.add(comp)
    db.commit()

    return {
        "id": comparison_id,
        "name": body.name,
        "results": results,
        "winner": winner,
        "metric_used": best_metric,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/history")
def comparison_history(
    limit: int = Query(20, ge=1, le=100),
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get comparison history."""
    comps = (
        db.query(ModelComparison)
        .order_by(ModelComparison.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": c.id,
            "name": c.name,
            "models": c.model_names,
            "winner": c.winner,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in comps
    ]


@router.get("/{comparison_id}")
def get_comparison(
    comparison_id: str,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific comparison by ID."""
    comp = db.query(ModelComparison).filter_by(id=comparison_id).first()
    if not comp:
        raise HTTPException(404, "Comparison not found")
    return {
        "id": comp.id,
        "name": comp.name,
        "models": comp.model_names,
        "dataset": comp.dataset,
        "results": comp.metrics,
        "winner": comp.winner,
        "notes": comp.notes,
        "created_at": comp.created_at.isoformat() if comp.created_at else None,
    }


@router.get("/leaderboard")
def model_leaderboard(
    metric: str = Query("accuracy", description="Metric to rank by"),
    limit: int = Query(10, ge=1, le=50),
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a leaderboard of models ranked by a specific metric."""
    from database import SessionLocal as DBSession
    from database_models import ModelRun

    session = DBSession()
    try:
        # Get latest completed run per model
        runs = (
            session.query(ModelRun)
            .filter_by(status="completed")
            .order_by(ModelRun.created_at.desc())
            .all()
        )

        seen = set()
        leaderboard = []
        for run in runs:
            if run.model_name in seen:
                continue
            seen.add(run.model_name)

            value = None
            if metric == "accuracy":
                value = run.accuracy
            elif metric == "f1_score":
                value = run.f1_score
            elif metric == "training_time":
                value = run.training_time
            elif metric == "inference_time":
                value = run.inference_time

            if value is not None:
                leaderboard.append(
                    {
                        "model_name": run.model_name,
                        "model_type": run.model_type,
                        metric: value,
                        "run_id": run.id,
                        "date": run.created_at.isoformat() if run.created_at else None,
                    }
                )

        leaderboard.sort(key=lambda x: x.get(metric, 0), reverse=True)
        return {"metric": metric, "leaderboard": leaderboard[:limit]}
    finally:
        session.close()
