"""
Real-Time Collaboration & Annotation Endpoints
- Add annotation/feedback to model predictions
- List annotations for a model
- (Stub) WebSocket for live collaboration
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_user import User
from api import get_current_user
from datetime import datetime
import os
import json

router = APIRouter(prefix="/collab", tags=["Collaboration & Annotation"])

ANNOTATION_DIR = os.path.join(os.path.dirname(__file__), "annotations")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/annotate/{model_id}")
def add_annotation(model_id: int, feedback: str, current_user: User = Depends(get_current_user)):
    ann_path = os.path.join(ANNOTATION_DIR, f"{model_id}.jsonl")
    entry = {
        "user": str(current_user.username),
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(ann_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"message": "Annotation added"}

@router.get("/list/{model_id}")
def list_annotations(model_id: int, current_user: User = Depends(get_current_user)):
    ann_path = os.path.join(ANNOTATION_DIR, f"{model_id}.jsonl")
    if not os.path.exists(ann_path):
        return []
    with open(ann_path, "r") as f:
        return [json.loads(line) for line in f]


from fastapi import WebSocket, WebSocketDisconnect

# In-memory set of active connections (for demo purposes)
active_connections = set()

@router.websocket("/ws/collab")
async def websocket_collaboration(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast to all connected clients
            for conn in active_connections:
                if conn != websocket:
                    await conn.send_text(data)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception:
        active_connections.remove(websocket)
