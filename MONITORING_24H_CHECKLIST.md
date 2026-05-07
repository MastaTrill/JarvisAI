# Jarvis AI 24-Hour Monitoring Handoff

## Scope
This checklist covers the first 24 hours after releasing checkpoint tag `stable-2026-05-07-a8cc25d`.

## Baseline (T+0)
- [x] Confirm branch/tag state:
  - `git status --short --branch`
  - `git rev-parse --short HEAD`
  - `git tag --list 'stable-*' | tail -n 10`
- [x] Confirm critical smoke checks:
  - `python -m pytest -o addopts='' tests/test_agent_tools.py::test_due_reminder_dispatch_and_voice_feed tests/test_auth.py::TestPasswordHashing::test_hash_and_verify tests/test_auth.py::TestJWTTokens::test_create_and_decode_access_token tests/test_api_coverage.py::TestLogin::test_login_success tests/test_api_coverage.py::TestMe::test_me_authenticated -q --maxfail=1`

## Monitoring Cadence
- T+15m: API health + logs
- T+1h: auth/reminder path sanity checks
- T+4h: error budget and queue sanity
- T+12h: repeat smoke checks
- T+24h: final stabilization review

## Commands (Run at each checkpoint)
- Health:
  - `curl -sS http://127.0.0.1:8000/health | cat`
- Auth sanity:
  - `curl -sS -X POST http://127.0.0.1:8000/token -H 'Content-Type: application/x-www-form-urlencoded' -d 'username=admin&password=admin' | cat`
- Reminder dispatch regression guard:
  - `python -m pytest -o addopts='' tests/test_agent_tools.py::test_due_reminder_dispatch_and_voice_feed -q --maxfail=1`

## Watch Items
- Error spikes in authentication endpoints.
- Reminder dispatch ordering regressions (voice-announced reminders should be prioritized).
- Rate limiter warnings when Redis is unavailable and fallback to in-memory limiter occurs.

## Escalation and Rollback
- Trigger rollback if:
  - Critical endpoint availability drops below agreed SLO.
  - Reproducible auth failures in healthy infrastructure.
  - Reminder dispatch regression reproduced in smoke test.
- Rollback reference:
  - Tag: `stable-2026-05-07-a8cc25d`
  - Commit: `a8cc25d`
