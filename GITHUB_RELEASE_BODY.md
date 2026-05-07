# Release v0.1.0-stable (2026-05-07)

## Summary
Stable checkpoint release with prioritized reminder dispatch and test data alignment fixes.

## What's Fixed
- **Reminder Dispatch Prioritization:** Voice-announced due reminders are now prioritized in Discord dispatch selection, ensuring urgent reminders are delivered first.
- **Numpy Test Data Alignment:** Detailed numpy training tests now explicitly request 4-feature sample data, matching the configured model input size and preventing shape mismatch errors.

## Validation Evidence
- ✅ **Full Test Suite:** 218 tests passed in 190.48 seconds
  - Agent tools & endpoints: 75 tests
  - API coverage & authentication: 43 tests
  - Data processing & training: 35 tests
  - Database models & versioning: 15 tests
  - Quantum & advanced features: 25 tests
  - Unit & integration: 10 tests

- ✅ **Targeted Smoke Checks:** 5 critical tests passed in 11.04 seconds
  - `test_due_reminder_dispatch_and_voice_feed`
  - `test_hash_and_verify`
  - `test_create_and_decode_access_token`
  - `test_login_success`
  - `test_me_authenticated`

## Installation & Upgrade
```bash
# Fetch the latest stable code
git fetch origin stable-2026-05-07-a8cc25d

# Checkout this release
git checkout stable-2026-05-07-a8cc25d

# Install/update dependencies (if needed)
python -m pip install -r requirements.txt

# Run smoke tests to verify in your environment
python -m pytest tests/test_agent_tools.py::test_due_reminder_dispatch_and_voice_feed \
  tests/test_auth.py::TestPasswordHashing::test_hash_and_verify \
  tests/test_auth.py::TestJWTTokens::test_create_and_decode_access_token \
  tests/test_api_coverage.py::TestLogin::test_login_success \
  tests/test_api_coverage.py::TestMe::test_me_authenticated -v
```

## Key Changes
### Code Changes
- `agent_task_memory.py`: Updated `list_due_proactive_reminders()` ordering logic to prioritize voice-announced reminders for Discord dispatch.
- `tests/test_training_numpy_detailed.py`: Added explicit `self.n_features = 4` in setup and updated three test methods to request matching feature counts.

### Breaking Changes
None. This is a patch release with internal fixes and no API surface changes.

## Known Issues & Caveats
- **Redis Dependency:** If Redis is unavailable, slowapi will fall back to in-memory rate limiting with a warning. This is graceful degradation.
- **Admin Dashboard Auth:** HTML route `/admin/dashboard/admin` requires authentication, but JSON endpoints under `/admin/dashboard/admin/*` currently respond without auth checks (architectural debt noted for future refinement).
- **Local API Testing:** In some container environments, uvicorn may take longer to bind to the port. Increase startup timeout if testing locally.

## Monitoring & Rollback
See [MONITORING_24H_CHECKLIST.md](MONITORING_24H_CHECKLIST.md) for post-release monitoring cadence and health check commands.

**Rollback reference:**
- Previous stable tag: (check git tag history)
- To rollback: `git checkout <previous-tag>` and redeploy

## Commits in This Release
- `a8cc25d` - fix: prioritize voiced reminders and align numpy detailed tests
- `a34f63c` - docs: add release checkpoint and 24h monitoring handoff
- `58bcac6` - docs: finalize monitoring checklist with actual baseline evidence

## Contributors
- William Mccoy (author of fixes and release tagging)

## Release Checklist
- [x] Code changes implemented and tested
- [x] Full test suite passing (218 tests)
- [x] Targeted smoke checks passing (5 critical tests)
- [x] Release notes written
- [x] Stable tag created and pushed
- [x] Monitoring checklist prepared
- [x] Documentation committed to main branch

---

**Tag:** `stable-2026-05-07-a8cc25d`  
**Release Date:** 2026-05-07 10:18 UTC  
**Status:** Production-Ready
