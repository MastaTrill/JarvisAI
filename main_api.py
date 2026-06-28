"""Compatibility API entrypoint."""

import os

os.environ.setdefault('OPENAI_API_KEY', '')
os.environ.setdefault('GROQ_API_KEY', '')
os.environ.setdefault('REDIS_URL', 'memory://')

from jarvis_api import app, get_current_user

__all__ = ["app", "get_current_user"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_api:app", host="0.0.0.0", port=8080, reload=True)
