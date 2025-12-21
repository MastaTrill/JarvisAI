"""
Job management API endpoints for admin dashboard
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from db_config import SessionLocal
from jobs_persistent import Job

router = APIRouter(prefix="/jobs", tags=["Jobs"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/cancel")
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancelled = True
    job.status = "cancelled"
    db.commit()
    return {"message": f"Job {job_id} cancelled"}

@router.post("/retry")
def retry_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancelled = False
    job.status = "queued"
    job.result = None
    job.completed_at = None
    db.commit()
    return {"message": f"Job {job_id} set to retry"}