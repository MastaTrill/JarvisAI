"""
Admin Audit Log API for Jarvis AI
- Exposes endpoints to view audit logs for admin actions
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db_config import get_db
from audit_trail import AuditTrail
from models_user import require_role, User

router = APIRouter(prefix="/admin/audit", tags=["Audit"])


@router.get("/logs")
def get_audit_logs(
    db: Session = Depends(get_db), _user: User = Depends(require_role("admin"))
):
    logs = db.query(AuditTrail).order_by(AuditTrail.timestamp.desc()).limit(200).all()
    return [
        {
            "id": log.id,
            "user": log.user,
            "action": log.action,
            "target": log.target,
            "timestamp": log.timestamp,
            "details": log.details,
        }
        for log in logs
    ]
