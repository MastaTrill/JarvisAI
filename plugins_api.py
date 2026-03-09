"""
Plugin & Extension Marketplace Endpoints
- Register plugin
- List plugins
- Approve/reject plugins (admin)
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from models_user import User
from auth_helpers import admin_required, get_current_user
from datetime import datetime
import os

router = APIRouter(prefix="/plugins", tags=["Plugin Marketplace"])

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "plugins")
os.makedirs(PLUGIN_DIR, exist_ok=True)

# In-memory plugin registry (replace with persistent DB as needed)
plugin_registry = {}


@router.post("/register")
def register_plugin(
    name: str,
    description: str = "",
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    raw_filename = (
        file.filename
        if file and file.filename
        else f"plugin_{datetime.utcnow().timestamp()}"
    )
    # Sanitize filename to prevent path traversal
    filename = os.path.basename(raw_filename)
    if not filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    plugin_path = os.path.join(PLUGIN_DIR, filename)
    with open(plugin_path, "wb") as f:
        f.write(file.file.read())
    plugin_registry[name] = {
        "description": description,
        "filename": filename,
        "uploaded_by": current_user.username,
        "approved": False,
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    return {"message": f"Plugin '{name}' uploaded for review."}


@router.get("/list")
def list_plugins():
    return plugin_registry


@router.post("/approve/{name}")
def approve_plugin(name: str, _current_user: User = Depends(admin_required)):
    if name not in plugin_registry:
        raise HTTPException(status_code=404, detail="Plugin not found")
    plugin_registry[name]["approved"] = True
    return {"message": f"Plugin '{name}' approved."}


@router.post("/reject/{name}")
def reject_plugin(name: str, _current_user: User = Depends(admin_required)):
    if name not in plugin_registry:
        raise HTTPException(status_code=404, detail="Plugin not found")
    plugin_registry[name]["approved"] = False
    return {"message": f"Plugin '{name}' rejected."}
