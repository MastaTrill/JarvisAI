# Changelog

## Unreleased

- Fix AI module import consistency by migrating legacy root-level imports to `src.*` paths in `jarvis_api.py`, `tests/conftest.py`, and `src/infra/auth_helpers.py`.
- Add `advanced_features` package initializer and compatibility wrapper for `src.ai.multimodal_ai`.
- Update FastAPI auth helper to import `User` from `src.infra.models_user` directly.
