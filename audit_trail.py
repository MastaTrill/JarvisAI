"""
Audit Trail Template for Jarvis AI
- Log data access, changes, and deletions
- Store audit logs in database or secure storage
"""
from db_config import Base
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

class AuditTrail(Base):
    __tablename__ = "audit_trail"
    id = Column(Integer, primary_key=True, index=True)
    user = Column(String)
    action = Column(String)
    target = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(String)


# Utility to log audit events
from db_config import SessionLocal
def log_audit_event(user, action, target, details=None):
    db = SessionLocal()
    try:
        entry = AuditTrail(user=user, action=action, target=target, details=details)
        db.add(entry)
        db.commit()
    finally:
        db.close()
