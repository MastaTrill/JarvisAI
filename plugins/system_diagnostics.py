import time
import platform
import datetime

from fastapi import APIRouter

router = APIRouter(tags=["Diagnostics"])


@router.get("/system/info")
def system_info():
    """Return platform and runtime information."""
    boot = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "uptime_since": boot,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@router.get("/system/resources")
def system_resources():
    """Return current CPU, memory, and disk usage."""
    import psutil
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "memory_available_gb": round(mem.available / (1024 ** 3), 2),
        "memory_percent": mem.percent,
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_percent": disk.percent,
    }


@router.get("/system/endpoints")
def list_endpoints(request):
    """Return all registered API endpoints with their methods."""
    from fastapi.routing import APIRoute
    endpoints = []
    app = request.app
    for route in app.routes:
        if isinstance(route, APIRoute):
            endpoints.append({
                "path": route.path,
                "methods": sorted(route.methods),
                "name": route.name,
                "summary": route.summary or "",
            })
    endpoints.sort(key=lambda e: e["path"])
    return {
        "total": len(endpoints),
        "endpoints": endpoints,
    }
