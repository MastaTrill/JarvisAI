"""
Minimal Jarvis AI API - serves mission page and dashboard only.
"""

from pathlib import Path
import os
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from agent_api import router

# Global app start time
app_start_time = None


async def monitor_system():
    """Placeholder system monitor."""
    while True:
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    global app_start_time
    app_start_time = datetime.now()
    asyncio.create_task(monitor_system())
    yield
    # Shutdown


app = FastAPI(
    title="Jarvis AI API",
    description="Jarvis AI Control Surface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include routers
app.include_router(router)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
try:
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
        name="static",
    )
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")


# Serve mission page at root
@app.get("/")
async def get_mission():
    """Serve the mission page."""
    mission_file = Path(__file__).parent / "static" / "dashboard" / "mission.html"
    if mission_file.exists():
        return FileResponse(
            mission_file,
            media_type="text/html",
            headers={
                "Permissions-Policy": "microphone=(self)",
                "Content-Security-Policy": "script-src 'self' 'unsafe-inline' 'unsafe-eval' cdnjs.cloudflare.com cdn.jsdelivr.net fonts.googleapis.com fonts.gstatic.com cdn.plot.ly; style-src 'self' 'unsafe-inline' fonts.googleapis.com cdnjs.cloudflare.com; font-src fonts.gstatic.com; connect-src 'self' *.openai.com",
            },
        )
    return {"error": "Mission page not found"}, 404


# Serve dashboard at /dashboard
@app.get("/dashboard")
async def get_dashboard():
    """Serve the dashboard page."""
    dashboard_file = Path(__file__).parent / "static" / "dashboard" / "index.html"
    if dashboard_file.exists():
        return FileResponse(
            dashboard_file,
            media_type="text/html",
            headers={
                "Permissions-Policy": "microphone=(self)",
                "Content-Security-Policy": "script-src 'self' 'unsafe-inline' 'unsafe-eval' cdnjs.cloudflare.com cdn.jsdelivr.net fonts.googleapis.com fonts.gstatic.com cdn.plot.ly; style-src 'self' 'unsafe-inline' fonts.googleapis.com cdnjs.cloudflare.com; font-src fonts.gstatic.com; connect-src 'self' *.openai.com",
            },
        )
    return {"error": "Dashboard page not found"}, 404


# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "uptime": (
            (datetime.now() - app_start_time).total_seconds() if app_start_time else 0
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7071)
