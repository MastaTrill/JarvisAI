from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import zipfile
from io import BytesIO
from PIL import Image
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import agent_api
import llm_ollama
import requests
from agent_api import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_tools_endpoint_lists_core_tools() -> None:
    client = _client()
    res = client.get("/agent/tools")
    assert res.status_code == 200
    names = {t["name"] for t in res.json()}
    assert "get_time" in names
    assert "shell_run" in names
    assert "repo_write_file" in names
    assert "browser_open" in names
    assert "browser_search" in names
    assert "desktop_launch" in names
    assert "desktop_control" in names
    assert "quantum_superposition" in names
    assert "quantum_entangle" in names
    assert "quantum_measure" in names
    assert "quantum_decipher" in names
    assert "quantum_experiment" in names
    assert "quantum_remediate" in names


def test_profile_roundtrip() -> None:
    client = _client()
    res = client.post("/agent/profile", json={"profile": "safe"})
    assert res.status_code == 200
    assert res.json()["profile"] == "safe"

    res2 = client.get("/agent/profile")
    assert res2.status_code == 200
    assert res2.json()["profile"] == "safe"


def test_chat_returns_plan() -> None:
    client = _client()
    res = client.post("/agent/chat", json={"message": "list files under data"})
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body.get("plan"), list)
    assert len(body["plan"]) >= 1


def test_basic_chat_mode_supports_normal_conversation(monkeypatch) -> None:
    client = _client()
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(agent_api, "is_openai_configured", lambda: False)
    monkeypatch.setattr(agent_api, "is_ollama_configured", lambda: False)

    res = client.post("/agent/chat", json={"message": "can we just talk normally?", "session_id": "basic-open-chat"})
    assert res.status_code == 200
    reply = res.json()["reply"].lower()
    assert "normal conversation" in reply or "lightweight mode" in reply
    assert "use /tool get_time" not in reply


def test_ollama_list_models_falls_back_to_localhost(monkeypatch) -> None:
    calls = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append(url)
        if url == "http://ollama:11434/api/tags":
            raise requests.ConnectionError(
                "HTTPConnectionPool(host='ollama', port=11434): "
                "Failed to resolve 'ollama' ([Errno 11001] getaddrinfo failed)"
            )
        if url == "http://127.0.0.1:11434/api/tags":
            return _Response({"models": [{"name": "llama3.2:3b"}]})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setattr(llm_ollama.requests, "request", fake_request)

    assert llm_ollama.ollama_list_models(timeout_s=2) == ["llama3.2:3b"]
    assert calls[:2] == [
        "http://ollama:11434/api/tags",
        "http://127.0.0.1:11434/api/tags",
    ]


def test_agent_chat_ollama_failure_is_human_readable(monkeypatch) -> None:
    client = _client()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(agent_api, "ollama_list_models", lambda timeout_s=5: [])
    monkeypatch.setattr(agent_api, "ollama_model", lambda: "llama3.2:3b")
    monkeypatch.setattr(
        agent_api,
        "ollama_chat",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Ollama is not running at http://127.0.0.1:11434. Start Ollama locally or update OLLAMA_BASE_URL.")
        ),
    )

    res = client.post("/agent/chat", json={"message": "hello", "session_id": "ollama-friendly-error"})
    assert res.status_code == 200
    reply = res.json()["reply"]
    assert "Ollama is not running at http://127.0.0.1:11434." in reply
    assert "Falling back to basic mode." in reply
    assert "HTTPConnectionPool" not in reply


def test_agent_chat_stream_ollama_failure_falls_back_to_basic(monkeypatch) -> None:
    client = _client()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(agent_api, "ollama_list_models", lambda timeout_s=5: [])
    monkeypatch.setattr(agent_api, "ollama_model", lambda: "llama3.2:3b")
    monkeypatch.setattr(
        agent_api,
        "ollama_chat_stream",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("Ollama timed out at http://127.0.0.1:11434. Start Ollama locally or update OLLAMA_BASE_URL.")
        ),
    )

    with client.stream(
        "POST",
        "/agent/chat/stream",
        json={"message": "hello", "session_id": "ollama-stream-friendly-error"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: error" not in body
    assert "event: done" in body
    assert "Ollama timed out at http://127.0.0.1:11434." in body
    assert "Falling back to basic mode." in body


def test_risky_shell_requires_confirm_in_safe_profile() -> None:
    client = _client()
    client.post("/agent/profile", json={"profile": "safe"})
    res = client.post(
        "/agent/chat",
        json={"message": '/tool shell_run {"command":"git reset --hard"}'},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["tool_result"]["result"]["blocked"] is True


def test_self_test_endpoint() -> None:
    client = _client()
    res = client.post("/agent/self_test")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] >= 3


def test_goal_runner_and_history() -> None:
    client = _client()
    run = client.post(
        "/agent/goals/run",
        json={"goal": "list files under data and show system info", "session_id": "goal-test"},
    )
    assert run.status_code == 200
    rb = run.json()
    assert rb["status"] in {"done", "blocked", "failed"}
    assert isinstance(rb["steps"], list)

    hist = client.get("/agent/goals/history?session_id=goal-test&limit=5")
    assert hist.status_code == 200
    hb = hist.json()
    assert isinstance(hb.get("runs"), list)


def test_goal_schedule_crud() -> None:
    client = _client()
    created = client.post(
        "/agent/goals/schedule",
        json={
            "goal": "list files under data",
            "interval_minutes": 30,
            "session_id": "sched-test",
            "auto_approve": False,
            "enabled": True,
        },
    )
    assert created.status_code == 200
    sid = int(created.json()["id"])

    listed = client.get("/agent/goals/schedule?limit=50")
    assert listed.status_code == 200
    schedules = listed.json().get("schedules", [])
    assert any(int(s["id"]) == sid for s in schedules)

    upd = client.post(f"/agent/goals/schedule/{sid}", json={"enabled": False})
    assert upd.status_code == 200


def test_quantum_status_endpoint() -> None:
    client = _client()
    res = client.get("/agent/quantum/status")
    assert res.status_code == 200
    body = res.json()
    assert "available" in body
    assert "creator_authorized" in body


def test_quantum_goal_template_runs() -> None:
    client = _client()
    res = client.post(
        "/agent/goals/run",
        json={"goal": "quantum measure basis computational", "session_id": "quant-goal", "auto_approve": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"done", "blocked", "failed"}


def test_quantum_decipher_goal_runs() -> None:
    client = _client()
    client.post("/agent/quantum/measure", json={"measurement_basis": "computational"})
    res = client.post(
        "/agent/goals/run",
        json={"goal": "quantum decipher 24h", "session_id": "quant-decipher-goal", "auto_approve": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"done", "blocked", "failed"}


def test_quantum_experiment_and_remediation_goals_run() -> None:
    client = _client()
    r1 = client.post(
        "/agent/goals/run",
        json={"goal": "quantum experiment quick", "session_id": "quant-exp-goal", "auto_approve": True},
    )
    assert r1.status_code == 200
    assert r1.json()["status"] in {"done", "blocked", "failed"}

    r2 = client.post(
        "/agent/goals/run",
        json={"goal": "quantum remediation 24h", "session_id": "quant-rem-goal", "auto_approve": True},
    )
    assert r2.status_code == 200
    assert r2.json()["status"] in {"done", "blocked", "failed"}


def test_quantum_history_and_stats_endpoints() -> None:
    client = _client()
    client.post("/agent/quantum/superposition", json={"states": ["a", "b"]})
    client.post("/agent/quantum/entangle", json={"system_a": "core", "system_b": "memory"})
    client.post("/agent/quantum/measure", json={"measurement_basis": "computational"})

    h = client.get("/agent/quantum/history?limit=10")
    assert h.status_code == 200
    assert isinstance(h.json().get("events"), list)

    s = client.get("/agent/quantum/stats?hours=24")
    assert s.status_code == 200
    sb = s.json()
    assert "total_events" in sb
    assert "measurement_outcomes" in sb


def test_quantum_alert_endpoints() -> None:
    client = _client()
    cfg = client.post(
        "/agent/quantum/alerts/config",
        json={
            "enabled": True,
            "window_hours": 24,
            "min_measurements": 1,
            "min_entangles": 1,
            "outcome_one_min_pct": 0.0,
            "outcome_one_max_pct": 100.0,
            "entanglement_strength_min": 0.0,
        },
    )
    assert cfg.status_code == 200
    g = client.get("/agent/quantum/alerts")
    assert g.status_code == 200
    body = g.json()
    assert "config" in body
    assert "alerts" in body


def test_quantum_export_endpoints() -> None:
    client = _client()
    client.post("/agent/quantum/superposition", json={"states": ["left", "right"]})
    j = client.get("/agent/quantum/export?format=json&hours=24")
    assert j.status_code == 200
    assert "events" in j.json()

    c = client.get("/agent/quantum/export?format=csv&hours=24")
    assert c.status_code == 200
    assert "id,event_type,created_at" in c.text

    aj = client.get("/agent/quantum/alerts/export?format=json")
    assert aj.status_code == 200
    assert "config" in aj.json()

    ac = client.get("/agent/quantum/alerts/export?format=csv")
    assert ac.status_code == 200
    assert "generated_at,active,code,severity,message,value,window_hours" in ac.text
    assert "attachment; filename=" in (ac.headers.get("content-disposition") or "")


def test_quantum_bundle_and_decipher_endpoints() -> None:
    client = _client()
    client.post("/agent/quantum/superposition", json={"states": ["left", "right"]})
    client.post("/agent/quantum/entangle", json={"system_a": "core", "system_b": "memory"})
    client.post("/agent/quantum/measure", json={"measurement_basis": "computational"})

    d = client.get("/agent/quantum/decipher?hours=24")
    assert d.status_code == 200
    db = d.json()
    assert "patterns" in db
    assert "recommendations" in db

    alias = client.get("/agent/quantum/decypher?hours=24")
    assert alias.status_code == 200
    assert "signals" in alias.json()

    z = client.get("/agent/quantum/export/all?format=csv&hours=24")
    assert z.status_code == 200
    assert z.headers.get("content-type", "").startswith("application/zip")
    with zipfile.ZipFile(BytesIO(z.content), mode="r") as zf:
        names = set(zf.namelist())
        assert "quantum_history.csv" in names
        assert "quantum_alerts.csv" in names
        assert "quantum_decipher.csv" in names

    zjson = client.get("/agent/quantum/export/all?format=json&hours=24")
    assert zjson.status_code == 200
    with zipfile.ZipFile(BytesIO(zjson.content), mode="r") as zf:
        names = set(zf.namelist())
        assert "quantum_history.json" in names
        assert "quantum_alerts.json" in names
        assert "quantum_decipher.json" in names


def test_quantum_export_date_filters() -> None:
    client = _client()
    client.post("/agent/quantum/superposition", json={"states": ["left", "right"]})
    j = client.get(
        "/agent/quantum/export?format=json&hours=24"
        "&start_at=2099-01-01T00:00:00Z&end_at=2099-01-01T01:00:00Z"
    )
    assert j.status_code == 200
    assert j.json().get("events") == []


def test_quantum_advanced_endpoints() -> None:
    client = _client()
    client.post("/agent/quantum/superposition", json={"states": ["a", "b"]})
    client.post("/agent/quantum/entangle", json={"system_a": "core", "system_b": "memory"})
    client.post("/agent/quantum/measure", json={"measurement_basis": "computational"})

    tl = client.get("/agent/quantum/timeline?hours=24&bucket_minutes=60")
    assert tl.status_code == 200
    assert "points" in tl.json()

    an = client.get("/agent/quantum/anomalies?hours=24&z_threshold=1.0")
    assert an.status_code == 200
    assert "anomalies" in an.json()

    ba = client.get("/agent/quantum/basis-analysis?hours=24")
    assert ba.status_code == 200
    assert "bases" in ba.json()

    exp = client.post("/agent/quantum/experiment/run", json={"preset": "quick"})
    assert exp.status_code == 200
    assert "snapshot_id" in exp.json()

    snap = client.post("/agent/quantum/decipher/snapshot?hours=24")
    assert snap.status_code == 200
    assert "id" in snap.json()

    snaps = client.get("/agent/quantum/decipher/snapshots?limit=5")
    assert snaps.status_code == 200
    assert isinstance(snaps.json().get("snapshots"), list)

    rem_cfg = client.post(
        "/agent/quantum/remediation/config",
        json={
            "enabled": True,
            "bias_threshold_pct": 10.0,
            "entanglement_min": 0.9,
            "measure_iterations": 1,
            "entangle_iterations": 1,
            "preferred_basis": "computational",
        },
    )
    assert rem_cfg.status_code == 200

    rem = client.post("/agent/quantum/remediation/run?hours=24&force=true")
    assert rem.status_code == 200
    assert "ran" in rem.json()

    hs = client.get("/agent/quantum/health-score?hours=24")
    assert hs.status_code == 200
    assert "score" in hs.json()

    replay = client.get("/agent/quantum/replay?hours=24")
    assert replay.status_code == 200
    assert "window" in replay.json()

    noc = client.get("/agent/quantum/noc?hours=24")
    assert noc.status_code == 200
    assert "health_score" in noc.json()

    tune = client.post("/agent/quantum/remediation/tune?hours=168&apply=false")
    assert tune.status_code == 200
    assert "suggested_config" in tune.json() or "config" in tune.json()

    pdf = client.get("/agent/quantum/summary.pdf?hours=24")
    assert pdf.status_code == 200
    assert pdf.headers.get("content-type", "").startswith("application/pdf")

    ncfg = client.post(
        "/agent/quantum/notifications/config",
        json={
            "enabled": False,
            "channel": "generic",
            "webhook_url": "",
            "webhook_url_warning": "",
            "webhook_url_critical": "",
            "min_severity": "warning",
        },
    )
    assert ncfg.status_code == 200

    ncfg2 = client.post(
        "/agent/quantum/notifications/config",
        json={
            "enabled": True,
            "channel": "discord",
            "webhook_url": "https://example.com/default",
            "webhook_url_warning": "https://example.com/warn",
            "webhook_url_critical": "https://example.com/crit",
            "min_severity": "warning",
        },
    )
    assert ncfg2.status_code == 200
    conf = ncfg2.json()
    assert conf.get("webhook_url_warning") == "https://example.com/warn"
    assert conf.get("webhook_url_critical") == "https://example.com/crit"

    ntest = client.post("/agent/quantum/notifications/test")
    assert ntest.status_code == 200
    assert "result" in ntest.json()

    ndispatch = client.post("/agent/quantum/notifications/dispatch?hours=24")
    assert ndispatch.status_code == 200
    assert "sent" in ndispatch.json()

    audit = client.get("/agent/quantum/ops-audit?limit=20")
    assert audit.status_code == 200
    assert isinstance(audit.json().get("items"), list)

    graph = client.get("/agent/quantum/memory-graph?hours=24")
    assert graph.status_code == 200
    assert "nodes" in graph.json()

    replay_agent = client.get("/agent/quantum/replay/agent?limit=10")
    assert replay_agent.status_code == 200
    assert "timeline" in replay_agent.json()

    sim = client.get("/agent/quantum/simulate?hours=24&outcome_one_min_pct=25&outcome_one_max_pct=75&entanglement_strength_min=0.88")
    assert sim.status_code == 200
    assert "simulated_config" in sim.json()

    packs = client.get("/agent/quantum/policy-packs")
    assert packs.status_code == 200
    assert "packs" in packs.json()

    apply_pack = client.post("/agent/quantum/policy-packs/safe/apply")
    assert apply_pack.status_code == 200
    assert apply_pack.json().get("pack") == "safe"

    runbook = client.get("/agent/quantum/runbook/entanglement_strength_low")
    assert runbook.status_code == 200
    assert "steps" in runbook.json()

    ann = client.post(
        "/agent/quantum/annotations",
        json={"item_type": "incident", "item_id": "latest", "note": "watch this", "author": "tester"},
    )
    assert ann.status_code == 200
    anns = client.get("/agent/quantum/annotations?limit=20")
    assert anns.status_code == 200
    assert isinstance(anns.json().get("annotations"), list)

    incidents = client.get("/agent/quantum/incidents?limit=20")
    assert incidents.status_code == 200
    assert isinstance(incidents.json().get("incidents"), list)

    ws = client.get("/agent/quantum/incidents/workspace")
    assert ws.status_code == 200
    wsb = ws.json()
    if wsb.get("incident"):
        inc_id = str(wsb["incident"]["id"])
        st = client.post(f"/agent/quantum/incidents/{inc_id}/status?status=acked")
        assert st.status_code == 200
        inc2 = st.json().get("incident") or {}
        assert inc2.get("status") in {"acked", "open", "closed"}
        checklist = inc2.get("checklist") or []
        if checklist:
            item_id = checklist[0].get("id")
            if item_id:
                chk = client.post(f"/agent/quantum/incidents/{inc_id}/checklist?item_id={item_id}&done=true")
                assert chk.status_code == 200


def test_memory_quality_endpoint() -> None:
    client = _client()
    sid = "mem-quality-test"
    client.post("/agent/chat", json={"message": "hello there", "session_id": sid})
    client.post("/agent/chat", json={"message": "help me plan next step", "session_id": sid})
    res = client.get(f"/agent/memory/quality?session_id={sid}")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == sid
    assert body["messages"] >= 2
    assert "overall_score_pct" in body


def test_long_term_memory_endpoints() -> None:
    client = _client()
    ident = client.post(
        "/agent/memory/remember",
        json={
            "content": "My name is Will.",
            "tags": ["profile", "identity"],
            "importance": 5,
            "memory_type": "identity",
            "subject": "user_name",
            "source": "test",
            "session_id": "mem-long",
        },
    )
    assert ident.status_code == 200
    created = client.post(
        "/agent/memory/remember",
        json={
            "content": "User is building Jarvis AI as a local assistant.",
            "tags": ["project", "profile"],
            "importance": 5,
            "memory_type": "project",
            "subject": "jarvis",
            "source": "test",
            "session_id": "mem-long",
        },
    )
    assert created.status_code == 200
    search = client.get("/agent/memory/search?query=Jarvis%20assistant&limit=5")
    assert search.status_code == 200
    items = search.json().get("items")
    assert isinstance(items, list)
    assert items
    assert items[0]["memory_type"] == "project"
    assert items[0]["subject"] == "jarvis"
    assert "effective_importance" in items[0]

    listed = client.get("/agent/memory/long-term?memory_type=project")
    assert listed.status_code == 200
    filtered = listed.json().get("items") or []
    assert filtered
    assert all(item["memory_type"] == "project" for item in filtered)

    dup = client.post(
        "/agent/memory/remember",
        json={
            "content": "User is building Jarvis AI as a local assistant with local tools.",
            "tags": ["project"],
            "importance": 4,
            "memory_type": "project",
            "subject": "jarvis",
            "source": "test",
            "session_id": "mem-long",
        },
    )
    assert dup.status_code == 200
    consolidated = client.post("/agent/memory/consolidate?limit=50")
    assert consolidated.status_code == 200
    assert "merged" in consolidated.json()

    overview = client.get("/agent/memory/overview?limit_per_group=4")
    assert overview.status_code == 200
    ob = overview.json()
    assert ob["total"] >= 2
    assert isinstance(ob.get("profile"), list)
    assert isinstance(ob.get("projects"), list)
    assert any(item["memory_type"] == "identity" for item in ob.get("profile", []))
    assert any(item["memory_type"] == "project" for item in ob.get("projects", []))

    profile = client.get("/agent/memory/profile?limit=5")
    assert profile.status_code == 200
    assert any(item["memory_type"] == "identity" for item in profile.json().get("items", []))

    projects = client.get("/agent/memory/projects?limit=5")
    assert projects.status_code == 200
    assert any(item["memory_type"] == "project" for item in projects.json().get("items", []))

    pinned_create = client.post(
        "/agent/memory/remember",
        json={
            "content": "Pin this memory for quick retrieval.",
            "tags": ["profile"],
            "importance": 4,
            "memory_type": "profile",
            "subject": "pin-test",
            "pinned": True,
            "source": "test",
            "session_id": "mem-long",
        },
    )
    assert pinned_create.status_code == 200
    pinned_id = int(pinned_create.json()["id"])

    pinned_list = client.get("/agent/memory/long-term?pinned=true&limit=20")
    assert pinned_list.status_code == 200
    assert any(int(item["id"]) == pinned_id and item["pinned"] is True for item in pinned_list.json().get("items", []))

    pin_toggle = client.post(f"/agent/memory/{pinned_id}/pin?pinned=false")
    assert pin_toggle.status_code == 200
    edited = client.post(
        f"/agent/memory/{pinned_id}",
        json={"content": "Edited memory content", "pinned": True},
    )
    assert edited.status_code == 200

    bulk = client.post(
        "/agent/memory/actions/bulk",
        json={"ids": [pinned_id], "action": "unpin"},
    )
    assert bulk.status_code == 200
    assert bulk.json()["updated"] >= 1

    since = client.get("/agent/memory/overview?limit_per_group=4&since_hours=24")
    assert since.status_code == 200
    assert since.json().get("since_hours") == 24

    deleted = client.request("DELETE", f"/agent/memory/{pinned_id}")
    assert deleted.status_code == 200
    assert deleted.json().get("archived") is True
    post_delete = client.get("/agent/memory/long-term?limit=50")
    assert all(int(item["id"]) != pinned_id for item in post_delete.json().get("items", []))
    archived = client.get("/agent/memory/long-term?limit=50&archived=true")
    assert any(int(item["id"]) == pinned_id and item["archived"] is True for item in archived.json().get("items", []))

    restored = client.post(
        "/agent/memory/actions/bulk",
        json={"ids": [pinned_id], "action": "restore"},
    )
    assert restored.status_code == 200
    restored_items = client.get("/agent/memory/long-term?limit=50").json().get("items", [])
    assert any(int(item["id"]) == pinned_id and item["archived"] is False for item in restored_items)


def test_memory_briefing_endpoint_uses_pinned_and_project_memory() -> None:
    client = _client()
    client.post(
        "/agent/memory/remember",
        json={
            "content": "Remember to prioritize the Jarvis dashboard refresh.",
            "tags": ["priority", "dashboard"],
            "importance": 5,
            "memory_type": "workflow",
            "subject": "dashboard-refresh",
            "pinned": True,
            "source": "test",
            "session_id": "briefing-test",
        },
    )
    client.post(
        "/agent/memory/remember",
        json={
            "content": "Jarvis AI project is focused on smart local orchestration.",
            "tags": ["project", "jarvis"],
            "importance": 5,
            "memory_type": "project",
            "subject": "jarvis",
            "source": "test",
            "session_id": "briefing-test",
        },
    )
    res = client.get("/agent/memory/briefing?period=morning&recent_project_hours=24")
    assert res.status_code == 200
    body = res.json()
    assert body["period"] == "morning"
    assert isinstance(body.get("pinned"), list)
    assert isinstance(body.get("recent_projects"), list)
    assert "Jarvis" in body.get("text", "")


def test_archived_memory_filters_and_pin_reorder() -> None:
    client = _client()
    created_ids = []
    for idx, payload in enumerate(
        [
            {
                "content": "Pinned alpha memory for ordering.",
                "tags": ["alpha", "priority"],
                "importance": 5,
                "memory_type": "profile",
                "subject": "alpha-order",
                "pinned": True,
                "lane": "critical",
            },
            {
                "content": "Pinned beta memory for ordering.",
                "tags": ["beta", "priority"],
                "importance": 5,
                "memory_type": "profile",
                "subject": "beta-order",
                "pinned": True,
                "lane": "personal",
            },
            {
                "content": "Archived gamma workflow snapshot.",
                "tags": ["gamma", "archive"],
                "importance": 3,
                "memory_type": "workflow",
                "subject": "gamma-archive",
                "pinned": False,
            },
        ],
        start=1,
    ):
        res = client.post("/agent/memory/remember", json={**payload, "source": "test", "session_id": f"mem-order-{idx}"})
        assert res.status_code == 200
        created_ids.append(int(res.json()["id"]))

    archive_res = client.request("DELETE", f"/agent/memory/{created_ids[2]}")
    assert archive_res.status_code == 200

    reorder = client.post("/agent/memory/pinned/reorder", json={"ids": [created_ids[1], created_ids[0]]})
    assert reorder.status_code == 200
    ordered = reorder.json()["ordered_ids"]
    assert ordered[:2] == [created_ids[1], created_ids[0]]

    overview = client.get("/agent/memory/overview?limit_per_group=6")
    assert overview.status_code == 200
    pinned = overview.json().get("pinned", [])
    assert len(pinned) >= 2
    assert int(pinned[0]["id"]) == created_ids[1]
    assert int(pinned[1]["id"]) == created_ids[0]
    pinned_lanes = overview.json().get("pinned_lanes", {})
    assert any(int(item["id"]) == created_ids[0] for item in pinned_lanes.get("critical", []))

    archived = client.get("/agent/memory/long-term?archived=true&query=gamma&tag=archive&memory_type=workflow")
    assert archived.status_code == 200
    archived_items = archived.json().get("items", [])
    assert any(int(item["id"]) == created_ids[2] for item in archived_items)

    saved = client.post(
        "/agent/memory/filters/saved",
        json={"name": "Gamma Archive", "query": "gamma", "tag": "archive", "memory_type": "workflow"},
    )
    assert saved.status_code == 200
    assert any(item["name"] == "Gamma Archive" for item in saved.json().get("items", []))
    saved_list = client.get("/agent/memory/filters/saved")
    assert saved_list.status_code == 200
    assert any(item["name"] == "Gamma Archive" for item in saved_list.json().get("items", []))
    saved_delete = client.request("DELETE", "/agent/memory/filters/saved?name=Gamma%20Archive")
    assert saved_delete.status_code == 200
    assert all(item["name"] != "Gamma Archive" for item in saved_delete.json().get("items", []))


def test_project_memory_reorder() -> None:
    client = _client()
    project_ids = []
    for idx, subject in enumerate(["alpha-project", "beta-project", "gamma-project"], start=1):
        res = client.post(
            "/agent/memory/remember",
            json={
                "content": f"Project memory {subject}",
                "tags": ["project", subject],
                "importance": 4,
                "memory_type": "project",
                "subject": subject,
                "source": "test",
                "session_id": f"project-order-{idx}",
            },
        )
        assert res.status_code == 200
        project_ids.append(int(res.json()["id"]))

    reorder = client.post("/agent/memory/projects/reorder", json={"ids": [project_ids[2], project_ids[0], project_ids[1]]})
    assert reorder.status_code == 200
    listed = client.get("/agent/memory/long-term?memory_type=project&limit=50")
    assert listed.status_code == 200
    ordered_ids = [int(item["id"]) for item in listed.json().get("items", []) if int(item["id"]) in project_ids]
    assert ordered_ids[:3] == [project_ids[2], project_ids[0], project_ids[1]]


def test_memory_briefing_automations_create_list_and_run() -> None:
    client = _client()
    created = client.post(
        "/agent/memory/briefings/automations",
        json={
            "session_id": "briefing-auto",
            "timezone_name": "America/Chicago",
            "morning_hour": 7,
            "evening_hour": 19,
        },
    )
    assert created.status_code == 200
    items = created.json().get("items", [])
    assert len(items) >= 2
    assert all(item["mode"] == "briefing" for item in items)
    assert any(item.get("metadata", {}).get("period") == "morning" for item in items)
    assert any(item.get("metadata", {}).get("period") == "evening" for item in items)

    listed = client.get("/agent/memory/briefings/automations")
    assert listed.status_code == 200
    listed_items = listed.json().get("items", [])
    assert len(listed_items) >= 2

    morning = next(item for item in listed_items if item.get("metadata", {}).get("period") == "morning")
    run = client.post(f"/agent/autonomy/jobs/{morning['id']}/run")
    assert run.status_code == 200
    result = run.json().get("result", {})
    assert result.get("mode") == "briefing"
    assert "briefing" in (result.get("text") or "").lower()


def test_memory_briefing_delivery_config_and_dispatch(monkeypatch) -> None:
    client = _client()

    monkeypatch.setattr(agent_api, "_dispatch_briefing_discord", lambda webhook_url, payload: {"sent": True, "target": webhook_url})
    monkeypatch.setattr(agent_api, "_dispatch_briefing_email", lambda email_to, subject, body_text: {"sent": True, "target": email_to, "subject": subject})
    monkeypatch.setattr(agent_api, "_dispatch_briefing_mobile", lambda push_url, payload, channel="ntfy": {"sent": True, "target": push_url, "channel": channel})

    cfg = client.post(
        "/agent/memory/briefings/delivery",
        json={
            "enabled": True,
            "discord_enabled": True,
            "discord_webhook_url": "https://discord.example/webhook",
            "email_enabled": True,
            "email_to": "jarvis@example.com",
            "mobile_enabled": True,
            "mobile_push_url": "https://ntfy.example/topic",
            "mobile_channel": "ntfy",
        },
    )
    assert cfg.status_code == 200
    assert cfg.json()["enabled"] is True

    test_res = client.post("/agent/memory/briefings/delivery/test?period=morning")
    assert test_res.status_code == 200
    delivery = test_res.json().get("delivery", {})
    results = delivery.get("results", [])
    assert len(results) == 3
    channels = {item["channel"] for item in results}
    assert channels == {"discord", "email", "mobile"}


def test_memory_workspaces_graph_and_reminders() -> None:
    client = _client()

    mem_a = client.post(
        "/agent/memory/remember",
        json={
            "content": "Jarvis workspace should prioritize dashboard upgrades.",
            "tags": ["jarvis", "dashboard", "priority"],
            "importance": 5,
            "memory_type": "project",
            "subject": "dashboard-core",
            "pinned": True,
            "lane": "project",
            "source": "test",
        },
    )
    assert mem_a.status_code == 200
    mem_b = client.post(
        "/agent/memory/remember",
        json={
            "content": "Voice flow should stay low-latency and holographic.",
            "tags": ["jarvis", "voice", "ux"],
            "importance": 4,
            "memory_type": "workflow",
            "subject": "voice-core",
            "source": "test",
        },
    )
    assert mem_b.status_code == 200
    mem_ids = [int(mem_a.json()["id"]), int(mem_b.json()["id"])]

    created = client.post(
        "/agent/memory/workspaces",
        json={
            "name": "Jarvis Prime",
            "description": "Main workspace for Jarvis upgrades",
            "focus": "dashboard and voice",
            "memory_ids": mem_ids,
        },
    )
    assert created.status_code == 200
    workspace = created.json()["workspace"]
    workspace_id = int(workspace["id"])
    assert workspace["name"] == "Jarvis Prime"
    assert len(workspace.get("memories", [])) >= 2

    active = client.post("/agent/memory/workspaces/active", json={"workspace_id": workspace_id})
    assert active.status_code == 200
    assert int(active.json()["active_workspace_id"]) == workspace_id

    graph = client.get(f"/agent/memory/graph?workspace_id={workspace_id}&limit=30")
    assert graph.status_code == 200
    graph_body = graph.json()
    assert graph_body["counts"]["nodes"] >= 3
    assert graph_body["counts"]["edges"] >= 2

    generated = client.post(f"/agent/memory/reminders/generate?workspace_id={workspace_id}&limit=3")
    assert generated.status_code == 200
    open_items = generated.json().get("open", [])
    assert open_items
    reminder_id = int(open_items[0]["id"])

    listed = client.get(f"/agent/memory/reminders?workspace_id={workspace_id}&status=open")
    assert listed.status_code == 200
    assert any(int(item["id"]) == reminder_id for item in listed.json().get("items", []))

    closed = client.post(f"/agent/memory/reminders/{reminder_id}", json={"status": "done", "delivered": True})
    assert closed.status_code == 200
    assert closed.json()["reminder"]["status"] == "done"


def test_control_config_and_safe_preview() -> None:
    client = _client()

    cfg = client.post(
        "/agent/control/config",
        json={
            "browser_enabled": True,
            "desktop_enabled": True,
            "execute_on_host": True,
            "browser_name": "default",
            "search_engine": "duckduckgo",
        },
    )
    assert cfg.status_code == 200
    cfg_body = cfg.json()
    assert cfg_body["browser_enabled"] is True
    assert cfg_body["search_engine"] == "duckduckgo"
    assert cfg_body["execute_on_host"] is False
    assert "browser_runtime" in cfg_body

    opened = client.post("/agent/control/browser/open", json={"url": "example.com"})
    assert opened.status_code == 200
    opened_body = opened.json()["result"]
    assert opened_body["executed"] is False
    assert opened_body["kind"] == "browser_open"
    assert "xdg-open" in opened_body["preview_command"]

    searched = client.post("/agent/control/browser/search", json={"query": "jarvis ai memory graph"})
    assert searched.status_code == 200
    search_body = searched.json()["result"]
    assert search_body["executed"] is False
    assert search_body["kind"] == "browser_search"
    assert "duckduckgo.com" in search_body["url"]

    launched = client.post("/agent/control/desktop/launch", json={"app": "dashboard"})
    assert launched.status_code == 200
    launch_body = launched.json()["result"]
    assert launch_body["executed"] is False
    assert launch_body["kind"] == "desktop_launch"


def test_workspace_switching_in_chat_and_tool_routing() -> None:
    client = _client()
    mem = client.post(
        "/agent/memory/remember",
        json={
            "content": "Jarvis Prime owns dashboard and voice focus.",
            "tags": ["jarvis", "project"],
            "importance": 5,
            "memory_type": "project",
            "subject": "jarvis-prime",
            "source": "test",
        },
    )
    assert mem.status_code == 200
    created = client.post(
        "/agent/memory/workspaces",
        json={
            "name": "Jarvis Prime",
            "focus": "dashboard and voice",
            "memory_ids": [int(mem.json()["id"])],
        },
    )
    assert created.status_code == 200
    switched = client.post("/agent/chat", json={"message": "switch workspace to Jarvis Prime", "session_id": "ws-chat"})
    assert switched.status_code == 200
    assert "switched" in switched.json()["reply"].lower()

    active = client.post("/agent/chat", json={"message": "what is my active workspace", "session_id": "ws-chat"})
    assert active.status_code == 200
    assert "Jarvis Prime" in active.json()["reply"]

    goal = client.post(
        "/agent/goals/run",
        json={"goal": "switch workspace to Jarvis Prime", "session_id": "ws-goal", "auto_approve": True},
    )
    assert goal.status_code == 200
    assert goal.json()["status"] in {"done", "blocked", "failed"}


def test_due_reminder_dispatch_and_voice_feed(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(agent_api, "_dispatch_briefing_discord", lambda webhook_url, payload: {"sent": True, "target": webhook_url})
    cfg = client.post(
        "/agent/memory/briefings/delivery",
        json={
            "enabled": True,
            "discord_enabled": True,
            "discord_webhook_url": "https://discord.example/webhook",
            "email_enabled": False,
            "mobile_enabled": False,
        },
    )
    assert cfg.status_code == 200
    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Due Jarvis reminder",
            "content": "Announce this now",
            "due_at": "2001-01-01T00:00:00+00:00",
            "priority": "high",
        },
    )
    assert reminder.status_code == 200
    rid = int(reminder.json()["reminder"]["id"])

    voice_feed = client.get("/agent/memory/reminders/voice-feed?limit=5")
    assert voice_feed.status_code == 200
    items = voice_feed.json().get("items", [])
    assert any("Announce this now" in item["text"] for item in items)

    dispatched = client.post("/agent/memory/reminders/dispatch-due?limit=5")
    assert dispatched.status_code == 200
    discord_items = dispatched.json().get("discord", [])
    assert any(int(item["reminder_id"]) == rid for item in discord_items)


def test_browser_workflow_route(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        agent_api,
        "_run_browser_workflow",
        lambda start_url, steps, headless=True, storage_state=None, capture_storage_state=False: {
            "ok": True,
            "start_url": start_url or "about:blank",
            "final_url": start_url or "about:blank",
            "results": [{"index": 1, "action": steps[0]["action"], "ok": True, "text": "Jarvis Browser Task"}],
            "extracted": [{"text": "Jarvis Browser Task"}],
            "runtime": {"available": True},
        },
    )
    res = client.post(
        "/agent/control/browser/workflow",
        json={
            "start_url": "data:text/html,<html><body><h1 id='title'>Jarvis Browser Task</h1></body></html>",
            "steps": [{"action": "extract_text", "selector": "#title"}],
            "headless": True,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["result"]["results"][0]["text"] == "Jarvis Browser Task"


def test_browser_workflow_library_and_template_run(monkeypatch) -> None:
    client = _client()
    library = client.get("/agent/control/browser/templates/library")
    assert library.status_code == 200
    items = library.json().get("items", [])
    names = {item["name"] for item in items}
    assert "GitHub Issues Inbox" in names
    assert "Gmail Inbox Sweep" in names
    assert "Notion Workspace Search" in names
    assert "Figma Recent Files" in names

    monkeypatch.setattr(
        agent_api,
        "_run_browser_workflow",
        lambda start_url, steps, headless=True, storage_state=None, capture_storage_state=False: {
            "ok": True,
            "start_url": start_url or "about:blank",
            "final_url": start_url or "about:blank",
            "results": [{"index": 1, "action": steps[0]["action"], "ok": True, "text": "Template workflow active"}],
            "extracted": [{"text": "Template workflow active"}],
            "runtime": {"available": True},
        },
    )
    run = client.post(
        "/agent/control/browser/workflow",
        json={
            "template_name": "GitHub Issues Inbox",
        },
    )
    assert run.status_code == 200
    body = run.json()
    assert body["result"]["start_url"] == "https://github.com/issues"
    assert body["result"]["results"][0]["text"] == "Template workflow active"


def test_browser_templates_and_trust_report() -> None:
    client = _client()
    saved = client.post(
        "/agent/control/browser/templates",
        json={
            "name": "Elite Template",
            "description": "Test template",
            "start_url": "https://example.com",
            "steps": [{"action": "goto", "value": "https://example.com"}],
        },
    )
    assert saved.status_code == 200
    assert any(item["name"] == "Elite Template" for item in saved.json().get("items", []))

    listed = client.get("/agent/control/browser/templates")
    assert listed.status_code == 200
    assert any(item["name"] == "Elite Template" for item in listed.json().get("items", []))

    client.post("/agent/control/browser/open", json={"url": "example.com"})
    client.post("/agent/control/browser/search", json={"query": "jarvis trust layer"})
    trust = client.get("/agent/trust/report?limit=40")
    assert trust.status_code == 200
    body = trust.json()
    assert body["total_runs"] >= 2
    assert any(item["tool_name"] == "browser_open" for item in body.get("tools", []))


def test_browser_session_storage_and_reload(monkeypatch) -> None:
    client = _client()
    workspace_name = f"Session Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Browser auth", "description": "Saved login sessions"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])
    session_name = f"elite-session-{uuid4().hex[:8]}"
    seen: dict[str, object] = {}

    def fake_browser_workflow(*, start_url, steps, headless=True, storage_state=None, capture_storage_state=False):
        seen["storage_state"] = storage_state
        return {
            "ok": True,
            "start_url": start_url or "about:blank",
            "final_url": start_url or "about:blank",
            "results": [{"index": 1, "action": steps[0]["action"], "ok": True, "text": "Jarvis Browser Task"}],
            "extracted": [{"text": "Jarvis Browser Task"}],
            "runtime": {"available": True},
            "storage_state": (
                {"cookies": [{"name": "jarvis_auth", "value": "token-123", "domain": "example.com", "path": "/"}], "origins": []}
                if capture_storage_state
                else None
            ),
        }

    monkeypatch.setattr(agent_api, "_run_browser_workflow", fake_browser_workflow)

    first = client.post(
        "/agent/control/browser/workflow",
        json={
            "start_url": "https://example.com/login",
            "steps": [{"action": "extract_text", "selector": "body"}],
            "workspace_id": workspace_id,
            "session_name": session_name,
            "save_session": True,
            "session_notes": "Jarvis auth",
        },
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["saved_session"]["name"] == session_name
    assert seen["storage_state"] is None

    listed = client.get(f"/agent/control/browser/sessions?workspace_id={workspace_id}")
    assert listed.status_code == 200
    assert any(item["name"] == session_name for item in listed.json().get("items", []))

    second = client.post(
        "/agent/control/browser/workflow",
        json={
            "start_url": "https://example.com/account",
            "steps": [{"action": "extract_text", "selector": "body"}],
            "workspace_id": workspace_id,
            "session_name": session_name,
        },
    )
    assert second.status_code == 200
    stored_state = seen.get("storage_state")
    assert isinstance(stored_state, dict)
    assert stored_state["cookies"][0]["name"] == "jarvis_auth"


def test_workspace_policy_and_next_best_action() -> None:
    client = _client()
    memory = client.post(
        "/agent/memory/remember",
        json={
            "content": "Jarvis should push the authenticated browser workspace forward.",
            "tags": ["jarvis", "workspace"],
            "importance": 5,
            "memory_type": "project",
            "subject": f"elite-{uuid4().hex[:6]}",
            "source": "test",
            "session_id": "next-action-memory",
            "pinned": True,
            "lane": "project",
        },
    )
    assert memory.status_code == 200
    memory_id = int(memory.json()["id"])

    workspace_name = f"Policy Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={
            "name": workspace_name,
            "focus": "Elite ops",
            "description": "Workspace policy test",
            "memory_ids": [memory_id],
        },
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    policy = client.post(
        f"/agent/memory/workspaces/{workspace_id}/policy",
        json={
            "browser_allowed": False,
            "desktop_allowed": True,
            "shell_allowed": False,
            "repo_write_allowed": False,
            "require_confirmation": True,
        },
    )
    assert policy.status_code == 200
    assert policy.json()["policy"]["browser_allowed"] is False

    blocked = client.post("/agent/control/browser/open", json={"url": "https://example.com", "workspace_id": workspace_id})
    assert blocked.status_code == 403

    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Review elite workspace policy",
            "content": "Browser access is currently disabled for this workspace.",
            "priority": "critical",
            "workspace_id": workspace_id,
        },
    )
    assert reminder.status_code == 200

    agent_api._task_memory.log_tool_execution(
        tool_name="browser_workflow",
        status="failed",
        confidence=0.31,
        verification={"status": "failed", "checks": [{"name": "steps_ok", "ok": False}]},
        detail={"result": {"ok": False}},
        session_id=f"trust-{workspace_id}",
    )
    agent_api._task_memory.log_tool_execution(
        tool_name="browser_workflow",
        status="failed",
        confidence=0.29,
        verification={"status": "failed", "checks": [{"name": "steps_ok", "ok": False}]},
        detail={"result": {"ok": False}},
        session_id=f"trust-{workspace_id}",
    )

    next_action = client.get(f"/agent/next-action?workspace_id={workspace_id}&limit=6")
    assert next_action.status_code == 200
    body = next_action.json()
    sources = {item["source"] for item in body.get("actions", [])}
    assert "reminder" in sources
    assert "trust" in sources or "policy" in sources
    assert body["policy"]["browser_allowed"] is False


def test_auth_templates_and_next_action_execute() -> None:
    client = _client()
    auth_templates = client.get("/agent/control/browser/templates/auth")
    assert auth_templates.status_code == 200
    auth_names = {item["name"] for item in auth_templates.json().get("items", [])}
    assert "GitHub Auth Capture" in auth_names

    workspace_name = f"Execute Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Execution", "description": "Next action execution"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Close loop on Jarvis execution",
            "content": "Mark this reminder done through next-action execution.",
            "priority": "high",
            "workspace_id": workspace_id,
        },
    )
    assert reminder.status_code == 200

    action_list = client.get(f"/agent/next-action?workspace_id={workspace_id}&limit=6")
    assert action_list.status_code == 200
    actions = action_list.json().get("actions", [])
    reminder_action = next(item for item in actions if item.get("source") == "reminder")

    executed = client.post("/agent/next-action/execute", json={"action": reminder_action, "session_id": "next-action-exec"})
    assert executed.status_code == 200
    result = executed.json().get("result", {})
    assert result.get("status") == "done"


def test_autonomy_mission_run_executes_actions() -> None:
    client = _client()
    workspace_name = f"Mission Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Mission mode", "description": "Autonomy mission test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Mission reminder",
            "content": "Mission mode should close this.",
            "priority": "critical",
            "workspace_id": workspace_id,
        },
    )
    assert reminder.status_code == 200

    mission = client.post(
        "/agent/autonomy/mission/run",
        json={"workspace_id": workspace_id, "session_id": "mission-test", "limit": 2, "auto_approve": True},
    )
    assert mission.status_code == 200
    body = mission.json()
    assert body["session_id"] == "mission-test"
    assert len(body.get("executed", [])) >= 1


def test_chat_based_approval_prompt_and_resume(monkeypatch) -> None:
    client = _client()
    workspace_name = f"Approval Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Approvals", "description": "Chat approval workflow"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    policy = client.post(
        f"/agent/memory/workspaces/{workspace_id}/policy",
        json={
            "browser_allowed": True,
            "desktop_allowed": True,
            "shell_allowed": False,
            "repo_write_allowed": False,
            "require_confirmation": True,
        },
    )
    assert policy.status_code == 200

    monkeypatch.setattr(
        agent_api,
        "_run_browser_workflow",
        lambda start_url, steps, headless=True, storage_state=None, capture_storage_state=False: {
            "ok": True,
            "start_url": start_url or "about:blank",
            "final_url": start_url or "about:blank",
            "results": [{"index": 1, "action": steps[0]["action"], "ok": True, "text": "Approved run"}],
            "extracted": [{"text": "Approved run"}],
            "runtime": {"available": True},
        },
    )

    request_text = (
        '/tool browser_workflow {"start_url":"https://example.com","steps":[{"action":"extract_text","selector":"body"}],'
        f'"workspace_id":{workspace_id}}}'
    )
    first = client.post("/agent/chat", json={"message": request_text, "session_id": "approval-chat"})
    assert first.status_code == 200
    first_body = first.json()
    assert "waiting for approval" in first_body["reply"].lower()
    pending = first_body["tool_result"]["pending_approval"]
    assert pending["tool"] == "browser_workflow"

    approved = client.post("/agent/chat", json={"message": "approve", "session_id": "approval-chat"})
    assert approved.status_code == 200
    approved_body = approved.json()
    assert "approved and executed" in approved_body["reply"].lower()
    assert approved_body["tool_result"]["result"]["results"][0]["text"] == "Approved run"


def test_voice_wake_config_and_detect(monkeypatch) -> None:
    client = _client()
    saved = client.post(
        "/agent/voice/wake",
        json={"enabled": True, "wake_word": "hey jarvis", "threshold": 0.42, "chunk_ms": 800},
    )
    assert saved.status_code == 200
    assert saved.json()["wake_word"] == "hey jarvis"

    current = client.get("/agent/voice/wake")
    assert current.status_code == 200
    assert current.json()["threshold"] == 0.42

    monkeypatch.setattr(
        agent_api,
        "_wakeword_detect",
        lambda payload: {
            "ok": True,
            "available": True,
            "provider": "openwakeword",
            "wake_word": payload.wake_word or "hey jarvis",
            "detected": True,
            "score": 0.91,
            "threshold": payload.threshold or 0.42,
            "matched_model": "hey_jarvis",
            "available_models": ["hey_jarvis"],
            "sample_rate": 16000,
            "samples": 4096,
        },
    )
    detected = client.post(
        "/agent/voice/wake/detect",
        json={"pcm16_b64": "AAAAAA==", "sample_rate": 16000, "channels": 1, "wake_word": "hey jarvis"},
    )
    assert detected.status_code == 200
    assert detected.json()["detected"] is True


def test_local_voice_config_transcribe_and_speak(monkeypatch) -> None:
    client = _client()
    cfg = client.post(
        "/agent/voice/local",
        json={"enabled": True, "stt_model": "tiny", "tts_provider": "enhanced_local", "tts_style": "deep", "tts_voice": "en-us", "tts_rate": 160, "tts_pitch": 40},
    )
    assert cfg.status_code == 200
    assert cfg.json()["stt_model"] == "tiny"
    assert "enhanced_local" in cfg.json().get("available_tts_providers", [])
    assert "deep" in cfg.json().get("available_tts_styles", [])

    monkeypatch.setattr(
        agent_api,
        "_transcribe_audio_local",
        lambda data, filename="voice.wav": {
            "ok": True,
            "provider": "faster_whisper",
            "available": True,
            "text": "jarvis local voice is active",
            "language": "en",
            "model": "tiny",
        },
    )
    monkeypatch.setattr(agent_api, "_synthesize_speech_local", lambda text: b"RIFFdemo")

    transcribed = client.post(
        "/agent/voice/transcribe",
        files={"file": ("sample.wav", b"RIFF....WAVEfmt ", "audio/wav")},
    )
    assert transcribed.status_code == 200
    assert transcribed.json()["text"] == "jarvis local voice is active"

    spoken = client.post("/agent/voice/speak", json={"text": "Jarvis speaking locally"})
    assert spoken.status_code == 200
    assert spoken.content.startswith(b"RIFF")

    presets = client.get("/agent/voice/local/presets")
    assert presets.status_code == 200
    preset_items = presets.json().get("items", [])
    assert any(item["id"] == "prime" for item in preset_items)
    assert any(item["style"] == "operator" for item in preset_items)


def test_browser_session_health_and_extended_auth_templates(monkeypatch) -> None:
    client = _client()
    templates = client.get("/agent/control/browser/templates/auth")
    assert templates.status_code == 200
    auth_names = {item["name"] for item in templates.json().get("items", [])}
    assert "Slack Auth Capture" in auth_names
    assert "Microsoft Auth Capture" in auth_names
    assert "Notion Auth Capture" in auth_names
    assert "Figma Auth Capture" in auth_names

    saved = client.post(
        "/agent/control/browser/sessions",
        json={
            "name": "slack-auth",
            "workspace_id": None,
            "notes": "Slack workspace",
            "provider": "slack",
            "template_name": "Slack Auth Capture",
            "storage_state": {"cookies": [{"name": "slack", "value": "token"}], "origins": []},
        },
    )
    assert saved.status_code == 200

    monkeypatch.setattr(
        agent_api,
        "_run_browser_workflow",
        lambda start_url, steps, headless=True, storage_state=None, capture_storage_state=False: {
            "ok": True,
            "start_url": start_url or "about:blank",
            "final_url": "https://app.slack.com/client/T1/C1",
            "results": [{"index": 1, "action": "extract_text", "ok": True}],
            "extracted": [{"text": "Threads Channels Later"}],
            "runtime": {"available": True},
        },
    )
    health = client.post(
        "/agent/control/browser/sessions/health",
        json={"session_name": "slack-auth", "limit": 1},
    )
    assert health.status_code == 200
    item = health.json()["items"][0]
    assert item["health_status"] == "healthy"


def test_project_watcher_triggers_mission(monkeypatch) -> None:
    client = _client()
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": f"Watcher Workspace {uuid4().hex[:8]}", "focus": "Watcher mode", "description": "Autonomous watcher"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Watcher reminder",
            "content": "Trigger the project watcher mission.",
            "priority": "critical",
            "workspace_id": workspace_id,
        },
    )
    assert reminder.status_code == 200

    watcher = client.post(
        "/agent/autonomy/watchers",
        json={
            "workspace_id": workspace_id,
            "interval_minutes": 15,
            "session_id": "watcher-test",
            "auto_approve": True,
            "min_score": 5.0,
        },
    )
    assert watcher.status_code == 200
    job = watcher.json()["job"]
    assert job["mode"] == "watcher"

    monkeypatch.setattr(
        agent_api,
        "_run_autonomous_mission",
        lambda workspace_id=None, session_id=None, limit=3, auto_approve=False: {
            "ok": True,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "executed": [{"title": "Watcher reminder"}],
            "summary": "Watcher mission complete.",
        },
    )
    run = client.post(f"/agent/autonomy/jobs/{job['id']}/run")
    assert run.status_code == 200
    result = run.json()["result"]
    assert result["mode"] == "watcher"
    assert result["triggered"] is True


def test_chat_memory_identity_and_project_prompts() -> None:
    client = _client()
    client.post(
        "/agent/memory/remember",
        json={
            "content": "My name is Will.",
            "tags": ["profile", "identity"],
            "importance": 5,
            "memory_type": "identity",
            "subject": "user_name",
            "source": "test",
            "session_id": "mem-chat",
        },
    )
    client.post(
        "/agent/memory/remember",
        json={
            "content": "I am building Jarvis AI as my active project.",
            "tags": ["project"],
            "importance": 5,
            "memory_type": "project",
            "subject": "jarvis",
            "source": "test",
            "session_id": "mem-chat",
        },
    )
    who = client.post("/agent/chat", json={"message": "who am i", "session_id": "mem-chat"})
    assert who.status_code == 200
    assert "Will" in who.json()["reply"]

    proj = client.post("/agent/chat", json={"message": "what am i working on", "session_id": "mem-chat"})
    assert proj.status_code == 200
    assert "Jarvis AI" in proj.json()["reply"]


def test_vision_analyze_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_api,
        "_ocr_from_image",
        lambda img: {"text": "Jarvis dashboard online", "confidence": 96.4, "engine": "tesseract"},
    )
    monkeypatch.setattr(
        agent_api,
        "_multimodal_vision_reasoning",
        lambda data, ocr_text="", prompt=None: {
            "summary": "The dashboard is open and healthy, with the Jarvis panel centered.",
            "provider": "ollama",
            "model": "llava:7b",
        },
    )
    client = _client()
    img = Image.new("RGB", (120, 80), color=(20, 80, 140))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    res = client.post(
        "/agent/vision/analyze",
        files={"file": ("screen.png", buf.getvalue(), "image/png")},
        data={"source": "test_capture", "session_id": "vision-test", "prompt": "Describe the UI state"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("ok") is True
    assert "summary" in body
    assert "Jarvis dashboard online" in body["summary"]
    assert body["details"]["ocr_text"] == "Jarvis dashboard online"
    assert body["details"]["multimodal_model"] == "llava:7b"
    assert body["details"]["vision_mode"] == "multimodal+ocr"
    listed = client.get("/agent/vision/observations?limit=5")
    assert listed.status_code == 200
    assert isinstance(listed.json().get("items"), list)


def test_vision_config_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(agent_api, "ollama_list_models", lambda timeout_s=5: ["llava:7b", "llama3.2:1b"])
    client = _client()
    saved = client.post("/agent/vision/config", json={"provider": "ollama", "model": "llava:7b"})
    assert saved.status_code == 200
    body = saved.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "llava:7b"
    loaded = client.get("/agent/vision/config")
    assert loaded.status_code == 200
    assert loaded.json()["provider"] == "ollama"


def test_autonomous_job_endpoints() -> None:
    client = _client()
    created = client.post(
        "/agent/autonomy/jobs",
        json={
            "name": "Docs planner",
            "goal": "plan API docs cleanup for the dashboard",
            "mode": "multi_agent",
            "interval_minutes": 30,
            "session_id": "auto-job-test",
            "auto_approve": True,
            "enabled": True,
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body.get("ok") is True
    job_id = int(body["id"])

    listed = client.get("/agent/autonomy/jobs?limit=20")
    assert listed.status_code == 200
    items = listed.json().get("items") or []
    target = next((item for item in items if int(item["id"]) == job_id), None)
    assert target is not None
    assert target["name"] == "Docs planner"

    updated = client.post(
        f"/agent/autonomy/jobs/{job_id}",
        json={"enabled": False, "interval_minutes": 45},
    )
    assert updated.status_code == 200
    assert updated.json().get("ok") is True

    run = client.post(f"/agent/autonomy/jobs/{job_id}/run")
    assert run.status_code == 200
    result = run.json()
    assert result.get("ok") is True
    assert int(result["job_id"]) == job_id
    assert isinstance(result.get("result"), dict)


def test_llm_benchmark_dry_run() -> None:
    client = _client()
    res = client.post(
        "/agent/benchmark/llm",
        json={
            "prompts": ["Say hi", "List 2 API hardening checks"],
            "runs_per_prompt": 2,
            "mode": "fast",
            "dry_run": True,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("ok") is True
    assert body.get("dry_run") is True
    assert body.get("runs_per_prompt") == 2


def test_multi_agent_eval_endpoint() -> None:
    client = _client()
    res = client.post(
        "/agent/evals/multi-agent",
        json={"task": "Plan API hardening tasks", "runs": 1, "mode": "fast", "dry_run": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("ok") is True
    assert body.get("dry_run") is True
    assert body.get("runs") == 1


def test_autonomy_jobs_crud_and_run() -> None:
    client = _client()
    created = client.post(
        "/agent/autonomy/jobs",
        json={
            "name": "Data sweep",
            "goal": "list files under data",
            "mode": "goal",
            "interval_minutes": 30,
            "session_id": "auto-job-test",
            "auto_approve": True,
            "enabled": True,
        },
    )
    assert created.status_code == 200
    job_id = int(created.json()["id"])
    listed = client.get("/agent/autonomy/jobs?limit=20")
    assert listed.status_code == 200
    assert any(int(i["id"]) == job_id for i in listed.json().get("items", []))
    run = client.post(f"/agent/autonomy/jobs/{job_id}/run")
    assert run.status_code == 200
    body = run.json()
    assert body.get("ok") is True


def test_multi_agent_fast_synthesis_shape() -> None:
    client = _client()
    res = client.post(
        "/agent/multi-agent/run",
        json={"task": "quick plan for API docs cleanup", "fast_synthesis": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("fast_synthesis") is True
    assert isinstance(body.get("first_pass_roles"), list)
    assert "planner" in body.get("agents", {})
    assert "coder" in body.get("agents", {})
    assert "operator" in body.get("agents", {})


def test_elite_integrations_config_and_summary() -> None:
    client = _client()
    saved = client.post(
        "/agent/integrations/config",
        json={
            "github_enabled": True,
            "github_repo": "MastaTrill/JarvisAI",
            "github_token_set": True,
            "calendar_enabled": True,
            "calendar_provider": "google",
            "calendar_id": "primary",
            "email_enabled": True,
            "email_to": "jarvis@example.com",
        },
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["connections"]["github"]["connected"] is True
    assert body["connections"]["calendar"]["provider"] == "google"

    summary = client.get("/agent/integrations/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["connected_count"] >= 3
    assert "elite integrations" in payload["summary"].lower()


def test_integration_intelligence_and_desktop_awareness() -> None:
    client = _client()
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": f"Awareness {uuid4().hex[:8]}", "focus": "Deep work", "description": "Awareness context"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    client.post(
        "/agent/integrations/config",
        json={
            "github_enabled": True,
            "github_repo": "MastaTrill/JarvisAI",
            "github_token_set": False,
            "calendar_enabled": True,
            "calendar_provider": "local",
            "calendar_id": "jarvis-local",
            "email_enabled": True,
            "email_to": "jarvis@example.com",
        },
    )
    starts_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    ends_at = (datetime.now(timezone.utc) + timedelta(hours=1, minutes=30)).isoformat()
    client.post(
        "/agent/integrations/calendar/events",
        json={
            "title": "Awareness sync",
            "description": "Review elite roadmap",
            "starts_at": starts_at,
            "ends_at": ends_at,
            "location": "Mission Control",
            "workspace_id": workspace_id,
        },
    )
    saved = client.post(
        "/agent/desktop/presence",
        json={
            "workspace_id": workspace_id,
            "app_name": "VS Code",
            "window_title": "agent_api.py",
            "summary": "Implementing elite awareness",
            "details": {"mode": "coding", "focus_file": "agent_api.py"},
        },
    )
    assert saved.status_code == 200

    intelligence = client.get("/agent/integrations/intelligence")
    assert intelligence.status_code == 200
    body = intelligence.json()
    assert body["summary"]["enabled"] >= 3
    assert any(item["channel"] == "calendar" for item in body["recommendations"])

    awareness = client.get(f"/agent/desktop/awareness?workspace_id={workspace_id}")
    assert awareness.status_code == 200
    payload = awareness.json()
    assert payload["focus_mode"] == "coding"
    assert payload["signals"]["focus_file"] == "agent_api.py"
    assert any(item["title"] == "Awareness sync" for item in payload["nearby_calendar_events"])


def test_persisted_autonomy_mission_and_resume() -> None:
    client = _client()
    workspace_name = f"Mission Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Elite autonomy", "description": "Mission persistence test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    reminder = client.post(
        "/agent/memory/reminders",
        json={
            "title": "Elite mission reminder",
            "content": "Close this reminder from mission mode.",
            "priority": "critical",
            "workspace_id": workspace_id,
        },
    )
    assert reminder.status_code == 200

    started = client.post(
        "/agent/autonomy/missions/start",
        json={"workspace_id": workspace_id, "limit": 2, "auto_approve": True, "goal": "Handle top elite tasks"},
    )
    assert started.status_code == 200
    start_body = started.json()
    assert int(start_body["mission_id"]) > 0
    assert start_body["mission_record"]["goal"] == "Handle top elite tasks"

    listed = client.get(f"/agent/autonomy/missions?workspace_id={workspace_id}")
    assert listed.status_code == 200
    items = listed.json().get("items", [])
    assert any(int(item["id"]) == int(start_body["mission_id"]) for item in items)

    resumed = client.post(f"/agent/autonomy/missions/{start_body['mission_id']}/resume?approve=true")
    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert int(resumed_body["mission_id"]) == int(start_body["mission_id"])
    assert "mission_record" in resumed_body


def test_autonomy_mission_checkpoint_and_retry(monkeypatch) -> None:
    client = _client()
    workspace_name = f"Retry Mission {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Retries", "description": "Mission checkpoint test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    sequence = {"count": 0}

    def fake_next_best_actions(*, workspace_id=None, limit=3):
        return {
            "workspace_id": workspace_id,
            "summary": "Retry mission plan",
            "actions": [
                {
                    "title": "Retry chat step",
                    "action": "ask jarvis",
                    "execution": {"kind": "chat", "message": "retry me"},
                    "score": 9.1,
                }
            ],
        }

    def fake_agent_chat(payload):
        sequence["count"] += 1
        if sequence["count"] == 1:
            raise RuntimeError("temporary model outage")
        return agent_api.AgentChatResponse(
            reply="Recovered on retry",
            session_id=payload.session_id or "retry-session",
            plan=["Retry", "Recover"],
            tool_result={},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    monkeypatch.setattr(agent_api._task_memory, "next_best_actions", fake_next_best_actions)
    monkeypatch.setattr(agent_api, "agent_chat", fake_agent_chat)

    started = client.post(
        "/agent/autonomy/missions/start",
        json={"workspace_id": workspace_id, "limit": 1, "auto_approve": True, "goal": "Recover the mission", "retry_limit": 1},
    )
    assert started.status_code == 200
    body = started.json()
    assert body["ok"] is True
    assert body["executed"][0]["reply"] == "Recovered on retry"
    assert body["checkpoint"]["retry_attempts_used"] >= 1
    assert body["checkpoint"]["remaining_actions"] == []


def test_watcher_network_and_trust_receipts() -> None:
    client = _client()
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": f"Watcher Net {uuid4().hex[:8]}", "focus": "Watchers", "description": "Watcher network test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    client.post("/agent/integrations/config", json={"github_enabled": True, "github_repo": "MastaTrill/JarvisAI"})
    watcher = client.post(
        "/agent/autonomy/watchers",
        json={
            "workspace_id": workspace_id,
            "enabled": True,
            "interval_minutes": 20,
            "auto_approve": True,
            "min_score": 5,
            "session_id": "watcher-network",
        },
    )
    assert watcher.status_code == 200

    client.post("/agent/control/browser/search", json={"query": "jarvis elite trust receipts"})
    network = client.get("/agent/autonomy/watchers/network")
    assert network.status_code == 200
    net = network.json()
    assert net["summary"]["total_watchers"] >= 1
    assert "project" in net["coverage"]

    receipts = client.get("/agent/trust/receipts?limit=10")
    assert receipts.status_code == 200
    body = receipts.json()
    assert body["summary"]["tool_receipts"] >= 1
    assert any(item["type"] == "tool" for item in body["receipts"])


def test_desktop_presence_snapshot_and_context() -> None:
    client = _client()
    workspace_name = f"Presence Workspace {uuid4().hex[:8]}"
    created = client.post(
        "/agent/memory/workspaces",
        json={"name": workspace_name, "focus": "Desktop presence", "description": "Presence test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    saved = client.post(
        "/agent/desktop/presence",
        json={
            "workspace_id": workspace_id,
            "app_name": "VS Code",
            "window_title": "JarvisAI-main",
            "summary": "Editing Jarvis elite features",
            "details": {"mode": "coding", "focus_file": "agent_api.py"},
        },
    )
    assert saved.status_code == 200
    snapshot = saved.json()["snapshot"]
    assert snapshot["app_name"] == "VS Code"
    assert snapshot["details"]["focus_file"] == "agent_api.py"

    presence = client.get(f"/agent/desktop/presence?workspace_id={workspace_id}")
    assert presence.status_code == 200
    body = presence.json()
    assert body["snapshot"]["window_title"] == "JarvisAI-main"
    assert body["workspace"]["id"] == workspace_id


def test_desktop_control_preview_and_verification() -> None:
    client = _client()
    result = client.post(
        "/agent/control/desktop/action",
        json={"action": "type_text", "text": "Jarvis desktop control test"},
    )
    assert result.status_code == 200
    body = result.json()
    assert body["result"]["action"] == "type_text"
    assert body["result"]["runtime"]["xdotool_available"] in {True, False}
    assert body["verification"]["status"] in {"preview", "verified", "failed"}


def test_github_issue_action_preview_and_send(monkeypatch) -> None:
    client = _client()
    preview = client.post(
        "/agent/integrations/github/issues",
        json={"title": "Elite issue", "body": "Preview path", "repo": "MastaTrill/JarvisAI"},
    )
    assert preview.status_code == 200
    assert preview.json()["preview"] is True

    monkeypatch.setattr(
        agent_api,
        "_github_issue_create",
        lambda payload: {
            "ok": True,
            "preview": False,
            "provider": "github",
            "repo": payload.repo,
            "issue": {"html_url": "https://github.com/MastaTrill/JarvisAI/issues/1", "number": 1},
        },
    )
    sent = client.post(
        "/agent/integrations/github/issues",
        json={"title": "Elite issue", "body": "Send path", "repo": "MastaTrill/JarvisAI"},
    )
    assert sent.status_code == 200
    body = sent.json()
    assert body["preview"] is False
    assert body["issue"]["number"] == 1


def test_github_pull_review_preview() -> None:
    client = _client()
    preview = client.post(
        "/agent/integrations/github/pulls/review",
        json={"repo": "MastaTrill/JarvisAI", "pull_number": 7, "body": "Looks good overall.", "event": "COMMENT"},
    )
    assert preview.status_code == 200
    body = preview.json()
    assert body["provider"] == "github"
    assert body["pull_number"] == 7
    assert body["preview"] is True


def test_github_pull_summary_preview(monkeypatch) -> None:
    client = _client()

    def fake_api_json(url: str, *, token: str = ""):
        if url.endswith("/pulls/9"):
            return {
                "title": "Mission mode cleanup",
                "state": "open",
                "draft": False,
                "mergeable_state": "clean",
                "user": {"login": "jarvis"},
                "base": {"ref": "main"},
                "head": {"ref": "codex/mission-mode"},
                "commits": 3,
                "additions": 42,
                "deletions": 7,
                "changed_files": 2,
                "comments": 1,
                "review_comments": 2,
                "body": "Tightens the command surface and mission HUD.",
            }
        if "/files?" in url:
            return [
                {
                    "filename": "static/dashboard/index.html",
                    "status": "modified",
                    "additions": 20,
                    "deletions": 4,
                    "changes": 24,
                    "patch": "@@ -1 +1 @@\n-old\n+new",
                }
            ]
        if "/reviews?" in url:
            return [{"state": "APPROVED"}, {"state": "COMMENTED"}]
        if "/comments?" in url:
            return [{"user": {"login": "operator"}, "body": "Looks sharp.", "created_at": "2026-03-24T12:00:00Z"}]
        return {}

    monkeypatch.setattr(agent_api, "_github_api_json", fake_api_json)
    monkeypatch.setattr(agent_api, "_github_api_json_list", fake_api_json)

    summary = client.post(
        "/agent/integrations/github/pulls/summary",
        json={"repo": "MastaTrill/JarvisAI", "pull_number": 9},
    )
    assert summary.status_code == 200
    body = summary.json()
    assert body["ok"] is True
    assert body["summary"]["title"] == "Mission mode cleanup"
    assert body["reviews"]["APPROVED"] == 1
    assert body["files"][0]["filename"] == "static/dashboard/index.html"


def test_calendar_event_create_and_list() -> None:
    client = _client()
    client.post("/agent/integrations/config", json={"calendar_enabled": True, "calendar_provider": "local", "calendar_id": "jarvis-local"})
    created = client.post(
        "/agent/integrations/calendar/events",
        json={
            "title": "Elite standup",
            "description": "Ship Jarvis upgrades",
            "starts_at": "2030-01-01T09:00:00+00:00",
            "ends_at": "2030-01-01T09:30:00+00:00",
            "location": "Mission Control",
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["event"]["title"] == "Elite standup"

    listed = client.get("/agent/integrations/calendar/events?limit=10")
    assert listed.status_code == 200
    assert any(item["title"] == "Elite standup" for item in listed.json().get("items", []))


def test_email_action_preview() -> None:
    client = _client()
    client.post("/agent/integrations/config", json={"email_enabled": False, "email_to": "jarvis@example.com"})
    res = client.post(
        "/agent/integrations/email/send",
        json={"subject": "Elite summary", "body": "Jarvis preview email"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["preview"] is True
    assert body["to"] == "jarvis@example.com"


def test_typed_watchers_and_voice_config_extensions() -> None:
    client = _client()
    cfg = client.get("/agent/voice/local")
    assert cfg.status_code == 200
    body = cfg.json()
    assert "available_tts_providers" in body
    assert "available_tts_styles" in body

    created = client.post(
        "/agent/memory/workspaces",
        json={"name": f"Typed Watchers {uuid4().hex[:8]}", "focus": "Typed watchers", "description": "Watcher type test"},
    )
    assert created.status_code == 200
    workspace_id = int(created.json()["workspace"]["id"])

    calendar = client.post(
        "/agent/autonomy/watchers",
        json={"watcher_type": "calendar", "workspace_id": workspace_id, "enabled": True, "interval_minutes": 30, "auto_approve": False, "min_score": 4.0},
    )
    assert calendar.status_code == 200

    desktop = client.post(
        "/agent/autonomy/watchers",
        json={"watcher_type": "desktop", "workspace_id": workspace_id, "enabled": True, "interval_minutes": 30, "auto_approve": False, "min_score": 4.0},
    )
    assert desktop.status_code == 200

    network = client.get("/agent/autonomy/watchers/network")
    assert network.status_code == 200
    payload = network.json()
    assert payload["coverage"]["calendar"]["enabled"] >= 1
    assert payload["coverage"]["desktop"]["enabled"] >= 1
    assert "calendar" in payload["summary"].get("typed_watchers", [])
