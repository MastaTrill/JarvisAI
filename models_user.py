from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth_helpers import get_current_user, require_role
from authentication import hash_password, verify_password
from database import get_db
from database_models import User

# Define router for admin endpoints
router = APIRouter()


def get_password_hash(password: str) -> str:
    return hash_password(password)


# Add any user model or schema definitions here if needed

# Example placeholder for user model (replace with actual implementation if required)
# class User(Base):
#     ...


# Admin dashboard scaffolding (API endpoints)
@router.get("/admin/users")
def list_users(
    db: Session = Depends(get_db), user: User = Depends(require_role("admin"))
):
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role.name if u.role else None,
            "is_active": u.is_active,
        }
        for u in users
    ]


@router.get("/admin/models")
def list_models(
    db: Session = Depends(get_db), user: User = Depends(require_role("admin"))
):
    from models_registry import ModelRegistry

    models = db.query(ModelRegistry).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "accuracy": m.accuracy,
        }
        for m in models
    ]


@router.get("/admin/jobs")
def list_jobs(
    db: Session = Depends(get_db), user: User = Depends(require_role("admin"))
):
    from jobs_persistent import Job

    jobs = db.query(Job).all()
    return [
        {
            "id": j.id,
            "job_id": j.job_id,
            "status": j.status,
            "result": j.result,
            "completed_at": j.completed_at,
            "cancelled": j.cancelled,
        }
        for j in jobs
    ]
