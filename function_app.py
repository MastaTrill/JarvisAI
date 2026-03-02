"""Azure Functions entrypoint for Jarvis AI FastAPI app."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import azure.functions as func


def _load_fastapi_app():
    api_file = Path(__file__).with_name("api.py")
    spec = spec_from_file_location("jarvis_api_main", api_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FastAPI module from {api_file}")
    module = module_from_spec(spec)
    sys.modules["api"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "app"):
        raise RuntimeError("Loaded module does not expose FastAPI 'app'")
    return module.app


app = func.AsgiFunctionApp(
    app=_load_fastapi_app(),
    http_auth_level=func.AuthLevel.ANONYMOUS,
)
