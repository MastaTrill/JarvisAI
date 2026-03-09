"""
Advanced Security Endpoints
- API key management (create, revoke, list)
- SSO (OAuth2) login stub
- RBAC: assign/revoke roles
"""

from fastapi import APIRouter, Depends, HTTPException
from models_user import User
from auth_helpers import admin_required
from datetime import datetime, timezone
import secrets

router = APIRouter(prefix="/security", tags=["Advanced Security"])

# In-memory API key store (replace with persistent DB in production)
api_keys = {}

# RBAC role assignment (in-memory for now)
user_roles = {}


@router.post("/apikey/create")
def create_api_key(_current_user: User = Depends(admin_required)):
    key = secrets.token_urlsafe(32)
    api_keys[key] = {
        "user": str(_current_user.username),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revoked": False,
    }
    return {"api_key": key}


@router.post("/apikey/revoke")
def revoke_api_key(key: str, _current_user: User = Depends(admin_required)):
    if key in api_keys:
        api_keys[key]["revoked"] = True
        return {"message": "API key revoked"}
    raise HTTPException(status_code=404, detail="API key not found")


@router.get("/apikey/list")
def list_api_keys(_current_user: User = Depends(admin_required)):
    return api_keys


@router.post("/role/assign")
def assign_role(
    username: str, role: str, _current_user: User = Depends(admin_required)
):
    user_roles[username] = role
    return {"message": f"Role '{role}' assigned to {username}"}


@router.post("/role/revoke")
def revoke_role(username: str, _current_user: User = Depends(admin_required)):
    if username in user_roles:
        del user_roles[username]
        return {"message": f"Role revoked for {username}"}
    raise HTTPException(status_code=404, detail="User not found")


@router.get("/role/list")
def list_roles(_current_user: User = Depends(admin_required)):
    return user_roles


@router.get("/sso/login")
def sso_login():
    # Stub: In production, redirect to OAuth2/SAML provider
    return {"message": "SSO login (stub)"}
