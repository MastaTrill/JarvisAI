"""
Advanced Security Endpoints
- API key management (create, revoke, list)
- SSO (OAuth2) login stub
- RBAC: assign/revoke roles
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db_config import SessionLocal
from models_user import User
from api import get_current_user
from datetime import datetime
import secrets

router = APIRouter(prefix="/security", tags=["Advanced Security"])

# In-memory API key store (replace with persistent DB in production)
api_keys = {}

# RBAC role assignment (in-memory for now)
user_roles = {}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def admin_required(current_user: User = Depends(get_current_user)):
    if not getattr(current_user.role, 'name', None) == 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.post("/apikey/create")
def create_api_key(current_user: User = Depends(admin_required)):
    key = secrets.token_urlsafe(32)
    api_keys[key] = {"user": str(current_user.username), "created_at": datetime.utcnow().isoformat(), "revoked": False}
    return {"api_key": key}

@router.post("/apikey/revoke")
def revoke_api_key(key: str, current_user: User = Depends(admin_required)):
    if key in api_keys:
        api_keys[key]["revoked"] = True
        return {"message": "API key revoked"}
    raise HTTPException(status_code=404, detail="API key not found")

@router.get("/apikey/list")
def list_api_keys(current_user: User = Depends(admin_required)):
    return api_keys

@router.post("/role/assign")
def assign_role(username: str, role: str, current_user: User = Depends(admin_required)):
    user_roles[username] = role
    return {"message": f"Role '{role}' assigned to {username}"}

@router.post("/role/revoke")
def revoke_role(username: str, current_user: User = Depends(admin_required)):
    if username in user_roles:
        del user_roles[username]
        return {"message": f"Role revoked for {username}"}
    raise HTTPException(status_code=404, detail="User not found")

@router.get("/role/list")
def list_roles(current_user: User = Depends(admin_required)):
    return user_roles

@router.get("/sso/login")
def sso_login():
    # Stub: In production, redirect to OAuth2/SAML provider
    return {"message": "SSO login (stub)"}
