"""Compatibility API entrypoint.

This module re-exports the canonical FastAPI app defined in top-level
``api.py`` so existing imports (tests, scripts, and tooling) continue to work
while routes are maintained in one place.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_api_path = Path(__file__).with_name("api.py")
_api_spec = spec_from_file_location("jarvis_legacy_api_module", _api_path)
if _api_spec is None or _api_spec.loader is None:
    raise ImportError(f"Unable to load API module from {_api_path}")

_api_module = module_from_spec(_api_spec)
_api_spec.loader.exec_module(_api_module)

app = _api_module.app
get_current_user = _api_module.get_current_user

__all__ = ["app", "get_current_user"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8080, reload=True)
