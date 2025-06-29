"""
Audit Trail & Compliance Dashboard Endpoints
- Log actions to audit trail
- List/export audit logs
- Compliance report stub
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_user import User
from api import get_current_user
from datetime import datetime
import csv
import os

router = APIRouter(prefix="/audit", tags=["Audit Trail & Compliance"])

AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_log.csv")
if not os.path.exists(AUDIT_LOG_PATH):
    with open(AUDIT_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user", "action", "detail"])

def log_audit(user: str, action: str, detail: str):
    with open(AUDIT_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), user, action, detail])

@router.post("/log")
def log_action(action: str, detail: str = "", current_user: User = Depends(get_current_user)):
    log_audit(current_user.username, action, detail)
    return {"message": "Action logged"}

@router.get("/list")
def list_audit_logs(current_user: User = Depends(get_current_user)):
    with open(AUDIT_LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

@router.get("/export")
def export_audit_logs(current_user: User = Depends(get_current_user)):
    with open(AUDIT_LOG_PATH, "r") as f:
        return {"csv": f.read()}

@router.get("/compliance-report")
def compliance_report(current_user: User = Depends(get_current_user)):
    # Stub: In production, generate a full compliance report
    return {"message": "Compliance report generated (stub)"}
