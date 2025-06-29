"""
Persistent Job Management for Jarvis AI
- SQLAlchemy ORM model for jobs
- CRUD utilities for job operations
"""
from db_config import Base, SessionLocal
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from datetime import datetime

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    cancelled = Column(Boolean, default=False)

# CRUD utilities

def create_job(session, job_id, status="queued"):
    job = Job(job_id=job_id, status=status)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job

def update_job_status(session, job_id, status, result=None, completed_at=None, cancelled=False):
    job = session.query(Job).filter_by(job_id=job_id).first()
    if job:
        job.status = status
        job.result = result
        job.completed_at = completed_at
        job.cancelled = cancelled
        session.commit()
        return job
    return None

def get_job(session, job_id):
    return session.query(Job).filter_by(job_id=job_id).first()
