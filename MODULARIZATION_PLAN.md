# Modularization Plan for agent_api.py

## Current State
- File: `agent_api.py` (~12,200 lines)
- Routes: 179 endpoints
- All routes registered on single `router = APIRouter(prefix="/agent")`

## Proposed Structure

```
api/
├── __init__.py          # Package init (already exists)
├── routes/
│   ├── __init__.py      # Export all routers
│   ├── chat.py          # /chat, /chat/stream endpoints
│   ├── memory.py        # /memory/* endpoints (20+)
│   ├── quantum.py       # /quantum/* endpoints (60+)
│   ├── autonomy.py      # /autonomy/* endpoints
│   ├── tasks.py         # /tasks/* endpoints
│   ├── tools.py         # /tools endpoint
│   ├── profile.py       # /profile, /tools, /policy endpoints
│   ├── vision.py        # /vision/* endpoints
│   ├── voice.py         # /voice/* endpoints
│   ├── desktop.py       # /desktop/* endpoints
│   ├── control.py       # /control/* endpoints
│   ├── integrations.py  # /integrations/* endpoints
│   ├── goals.py         # /goals/* endpoints
│   ├── briefing.py      # /memory/briefing/* endpoints
│   ├── workspace.py     # /memory/workspaces/* endpoints
│   └── self.py          # /self_test, /next-action endpoints
└── agent_api.py         # Refactored to import and include routers
```

## Migration Steps
1. Create `api/routes/` directory
2. Extract route groups to separate modules
3. Each module imports necessary models/helpers from parent
4. Update `agent_api.py` to import and include routers
5. Ensure all imports remain backward-compatible

## Dependencies
- Requires working Python environment with fastapi, pydantic, etc.
- Test all endpoints after migration