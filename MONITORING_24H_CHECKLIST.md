# Jarvis AI 24-Hour Monitoring Handoff

## Scope
This checklist covers the first 24 hours after releasing checkpoint tag `stable-2026-05-07-a8cc25d`.

## Baseline (T+0)
- [x] Confirm branch/tag state:
  - Current branch: main (synced with origin/main)
  - HEAD commit: a34f63c (docs: add release checkpoint and 24h monitoring handoff)
  - Stable tag: stable-2026-05-07-a8cc25d → a8cc25d
- [x] Confirm critical smoke checks (pytest):
  - test_due_reminder_dispatch_and_voice_feed: PASSED
  - test_hash_and_verify: PASSED
  - test_create_and_decode_access_token: PASSED
  - test_login_success: PASSED
  - test_me_authenticated: PASSED
  - Result: 5/5 passed in 11.04s
- [x] Full test suite validation:
  - Result: 218 passed in 190.48s
  - Covers: all agent tools, API endpoints, auth, training, data processing, versioning

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
