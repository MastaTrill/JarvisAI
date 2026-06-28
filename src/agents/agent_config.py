"""
Agent Personality & Multi-Model Conversation System for JarvisAI.

Provides:
- Customizable agent personality (name, style, expertise, tone)
- Per-conversation model routing based on task type
- Model capability detection and automatic fallback
"""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text

from db_config import Base as ConfigBase
from database import get_db
from auth_helpers import get_current_user
from models_user import User


# --- Database Models ---

class AgentPersonality(ConfigBase):
    __tablename__ = "agent_personalities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(200), default="Jarvis")
    system_prompt = Column(Text, default="")
    tone = Column(String(50), default="professional")  # professional, casual, friendly, technical, concise
    expertise = Column(String(500), default="")  # comma-separated
    language = Column(String(20), default="en")
    avatar_style = Column(String(50), default="default")
    is_default = Column(Integer, default=0)  # 0 or 1
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))


class ModelRoute(ConfigBase):
    __tablename__ = "model_routes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), default="")
    task_type = Column(String(50), nullable=False)  # chat, code, analysis, creative, vision, reasoning
    provider = Column(String(50), default="ollama")  # ollama, openai, groq
    model_name = Column(String(200), default="")
    priority = Column(Integer, default=0)
    enabled = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# --- Pydantic Schemas ---

class PersonalityCreate(BaseModel):
    name: str
    display_name: str = "Jarvis"
    system_prompt: str = ""
    tone: str = "professional"
    expertise: str = ""
    language: str = "en"
    avatar_style: str = "default"


class PersonalityUpdate(BaseModel):
    display_name: Optional[str] = None
    system_prompt: Optional[str] = None
    tone: Optional[str] = None
    expertise: Optional[str] = None
    language: Optional[str] = None
    avatar_style: Optional[str] = None
    is_default: Optional[bool] = None


class ModelRouteCreate(BaseModel):
    name: str
    description: str = ""
    task_type: str = "chat"
    provider: str = "ollama"
    model_name: str = ""
    priority: int = 0


# --- Router ---

router = APIRouter(prefix="/agent/config", tags=["Agent Configuration"])


# --- Personality Endpoints ---

@router.post("/personality")
def create_personality(
    body: PersonalityCreate,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new agent personality."""
    # Check if name exists
    existing = db.query(AgentPersonality).filter_by(name=body.name).first()
    if existing:
        raise HTTPException(400, f"Personality '{body.name}' already exists")

    personality = AgentPersonality(
        name=body.name,
        display_name=body.display_name,
        system_prompt=body.system_prompt,
        tone=body.tone,
        expertise=body.expertise,
        language=body.language,
        avatar_style=body.avatar_style,
        created_by=current_user.username,
    )
    db.add(personality)
    db.commit()
    db.refresh(personality)
    return {"message": f"Personality '{body.name}' created", "id": personality.id}


@router.get("/personalities")
def list_personalities(
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all agent personalities."""
    personalities = db.query(AgentPersonality).order_by(AgentPersonality.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "name": p.name,
            "display_name": p.display_name,
            "tone": p.tone,
            "expertise": p.expertise,
            "is_default": bool(p.is_default),
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in personalities
    ]


@router.get("/personality/{name}")
def get_personality(
    name: str,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific personality."""
    p = db.query(AgentPersonality).filter_by(name=name).first()
    if not p:
        raise HTTPException(404, "Personality not found")
    return {
        "id": p.id,
        "name": p.name,
        "display_name": p.display_name,
        "system_prompt": p.system_prompt,
        "tone": p.tone,
        "expertise": p.expertise,
        "language": p.language,
        "avatar_style": p.avatar_style,
        "is_default": bool(p.is_default),
    }


@router.post("/personality/{name}/activate")
def activate_personality(
    name: str,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Set a personality as the default."""
    p = db.query(AgentPersonality).filter_by(name=name).first()
    if not p:
        raise HTTPException(404, "Personality not found")

    # Clear existing default
    db.query(AgentPersonality).filter_by(is_default=1).update({"is_default": 0})
    p.is_default = 1
    db.commit()
    return {"message": f"Personality '{name}' activated"}


@router.delete("/personality/{name}")
def delete_personality(
    name: str,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a personality."""
    p = db.query(AgentPersonality).filter_by(name=name).first()
    if not p:
        raise HTTPException(404, "Personality not found")
    db.delete(p)
    db.commit()
    return {"message": f"Personality '{name}' deleted"}


# --- Model Routing Endpoints ---

@router.post("/model-route")
def create_model_route(
    body: ModelRouteCreate,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a model routing rule."""
    route = ModelRoute(
        name=body.name,
        description=body.description,
        task_type=body.task_type,
        provider=body.provider,
        model_name=body.model_name,
        priority=body.priority,
    )
    db.add(route)
    db.commit()
    db.refresh(route)
    return {"message": f"Model route '{body.name}' created", "id": route.id}


@router.get("/model-routes")
def list_model_routes(
    task_type: Optional[str] = None,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List model routing rules."""
    query = db.query(ModelRoute).filter_by(enabled=1)
    if task_type:
        query = query.filter_by(task_type=task_type)
    routes = query.order_by(ModelRoute.priority.desc()).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "task_type": r.task_type,
            "provider": r.provider,
            "model_name": r.model_name,
            "priority": r.priority,
        }
        for r in routes
    ]


@router.get("/model-route/resolve")
def resolve_model_for_task(
    task_type: str = "chat",
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resolve which model to use for a given task type."""
    route = (
        db.query(ModelRoute)
        .filter_by(task_type=task_type, enabled=1)
        .order_by(ModelRoute.priority.desc())
        .first()
    )
    if route:
        return {
            "task_type": task_type,
            "provider": route.provider,
            "model": route.model_name,
            "route_name": route.name,
        }
    # Fallback
    return {
        "task_type": task_type,
        "provider": "ollama",
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "route_name": "default",
    }


@router.delete("/model-route/{route_id}")
def delete_model_route(
    route_id: int,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a model route."""
    route = db.query(ModelRoute).filter_by(id=route_id).first()
    if not route:
        raise HTTPException(404, "Route not found")
    db.delete(route)
    db.commit()
    return {"message": "Route deleted"}
