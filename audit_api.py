"""
Audit Trail & Compliance Dashboard Endpoints
- Log actions to audit trail
- List/export audit logs
- Compliance report stub
"""

from fastapi import APIRouter, Depends
from models_user import User
from auth_helpers import get_current_user
from datetime import datetime
import csv
import os

router = APIRouter(prefix="/audit", tags=["Audit Trail & Compliance"])

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.csv")
if not os.path.exists(AUDIT_LOG_PATH):
    with open(AUDIT_LOG_PATH, "w", newline="", encoding="utf-8") as init_file:
        init_writer = csv.writer(init_file)
        init_writer.writerow(["timestamp", "user", "action", "detail"])


def log_audit(user: str, action: str, detail: str):
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as audit_file:
        audit_writer = csv.writer(audit_file)
        audit_writer.writerow([datetime.utcnow().isoformat(), user, action, detail])


@router.post("/log")
def log_action(
    action: str, detail: str = "", _current_user: User = Depends(get_current_user)
):
    log_audit(_current_user.username, action, detail)
    return {"message": "Action logged"}


@router.get("/list")
def list_audit_logs(_current_user: User = Depends(get_current_user)):
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as audit_file:
        reader = csv.DictReader(audit_file)
        return list(reader)


@router.get("/export")
def export_audit_logs(_current_user: User = Depends(get_current_user)):
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as audit_file:
        return {"csv": audit_file.read()}


@router.get("/compliance-report")
def compliance_report(_current_user: User = Depends(get_current_user)):
    # Stub: In production, generate a full compliance report
    return {"message": "Compliance report generated (stub)"}
