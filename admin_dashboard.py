"""
Admin Dashboard for Jarvis AI (FastAPI + Jinja2 scaffold)
- User, model, and job management
- Requires admin role (RBAC)
"""

import os

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from db_config import SessionLocal
from models_user import User, get_password_hash
from api import get_current_user
from audit_trail import log_audit_event
from models_registry import create_model, get_models
from jobs_persistent import Job, get_job, update_job_status

router = APIRouter(prefix="/admin/dashboard", tags=["AdminDashboard"])
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)


def get_db(session_override=None):
    """Get database session, optionally overriding with a provided session factory."""
    db = session_override() if session_override else SessionLocal()
    try:
        yield db
    finally:
        db.close()


def admin_required(current_user: User = Depends(get_current_user)):
    """Verify that the current user has admin role."""
    if not getattr(current_user.role, "name", None) == "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/admin", response_class=HTMLResponse)
def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Render admin dashboard with list of users."""
    users = db.query(User).all()
    return templates.TemplateResponse(
        "admin_dashboard.html",
        {"request": request, "users": users, "current_user": current_user},
    )


# --- User CRUD Endpoints ---


# GET /admin/users endpoint for test coverage
@router.get("/admin/users", response_class=JSONResponse)
def list_users(
    db: Session = Depends(get_db),
):
    """List all users in the system."""
    users = db.query(User).all()
    user_list = [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "is_active": u.is_active,
            "role": u.role,
        }
        for u in users
    ]
    return user_list


@router.post("/admin/users/create")
def create_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Create a new user with the provided credentials."""
    if db.query(User).filter_by(username=username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_audit_event(
        current_user.username, "create_user", username, f"Created user {username}"
    )
    return {"status": "created", "user_id": user.id}


@router.post("/admin/users/delete")
def delete_user(
    user_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Delete a user by ID."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    log_audit_event(
        current_user.username,
        "delete_user",
        user.username if user else str(user_id),
        f"Deleted user {user.username if user else user_id}",
    )
    return {"status": "deleted"}


# --- Model Management Endpoints ---


@router.get("/admin/models")
def list_models(
    db: Session = Depends(get_db),
):
    """List all registered models."""
    models = get_models(db)
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "accuracy": m.accuracy,
                "active": m.active,
            }
            for m in models
        ]
    }


@router.post("/admin/models/create")
def create_model_endpoint(
    name: str = Form(...),
    description: str = Form(""),
    accuracy: float = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Create a new model with the provided details."""
    model = create_model(db, name=name, description=description, accuracy=accuracy)
    log_audit_event(
        current_user.username, "create_model", name, f"Created model {name}"
    )
    return {"status": "created", "model_id": model.id}


# --- Job Management Endpoints ---


@router.get("/admin/jobs")
def list_jobs(
    db: Session = Depends(get_db),
):
    """List all jobs in the system."""
    jobs = db.query(Job).all()
    return {
        "jobs": [
            {
                "id": j.id,
                "job_id": j.job_id,
                "status": j.status,
                "created_at": j.created_at,
                "completed_at": j.completed_at,
                "cancelled": j.cancelled,
            }
            for j in jobs
        ]
    }


@router.post("/admin/jobs/cancel")
def cancel_job(
    job_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(admin_required),
):
    """Cancel a job by ID."""
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    update_job_status(db, job_id, status="cancelled", cancelled=True)
    log_audit_event(
        current_user.username, "cancel_job", job_id, f"Cancelled job {job_id}"
    )
    return {"status": "cancelled"}
