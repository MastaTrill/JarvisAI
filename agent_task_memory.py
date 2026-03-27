from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import re
from zoneinfo import ZoneInfo


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _default_db_path() -> str:
    override = os.getenv("JARVIS_AGENT_TASK_DB", "").strip()
    if override:
        return override
    project_root = Path(__file__).resolve().parent
    return str(project_root / "data" / "agent_tasks.db")


class AgentTaskMemory:
    """
    Persistent task + settings storage backed by SQLite.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        resolved_path = db_path or _default_db_path()
        self._db_path = Path(resolved_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    note TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL,
                    verification_json TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    session_id TEXT,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    session_id TEXT,
                    interval_minutes INTEGER NOT NULL,
                    auto_approve INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT NOT NULL,
                    last_status TEXT,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quantum_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    session_id TEXT,
                    measurement_basis TEXT,
                    outcome INTEGER,
                    measurement_probability REAL,
                    entanglement_strength REAL,
                    correlation_coefficient REAL,
                    states_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quantum_decipher_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    window_hours INTEGER NOT NULL,
                    events_analyzed INTEGER NOT NULL,
                    confidence_pct REAL NOT NULL,
                    signals_json TEXT NOT NULL,
                    patterns_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quantum_ops_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    op_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quantum_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    author TEXT,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    importance INTEGER NOT NULL DEFAULT 3,
                    memory_type TEXT NOT NULL DEFAULT 'general',
                    subject TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    source TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomous_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'multi_agent',
                    session_id TEXT,
                    interval_minutes INTEGER NOT NULL,
                    auto_approve INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT NOT NULL,
                    last_status TEXT,
                    last_error TEXT,
                    last_result_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    focus TEXT,
                    color TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_memory_map (
                    workspace_id INTEGER NOT NULL,
                    memory_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, memory_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS browser_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    workspace_id INTEGER,
                    storage_state_json TEXT NOT NULL,
                    provider TEXT,
                    template_name TEXT,
                    notes TEXT,
                    health_status TEXT,
                    health_details_json TEXT,
                    last_health_check_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_policies (
                    workspace_id INTEGER PRIMARY KEY,
                    browser_allowed INTEGER NOT NULL DEFAULT 1,
                    desktop_allowed INTEGER NOT NULL DEFAULT 1,
                    shell_allowed INTEGER NOT NULL DEFAULT 0,
                    repo_write_allowed INTEGER NOT NULL DEFAULT 0,
                    require_confirmation INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    due_at TEXT,
                    channel TEXT,
                    workspace_id INTEGER,
                    memory_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    delivered_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mission_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id INTEGER,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    goal TEXT,
                    auto_approve INTEGER NOT NULL DEFAULT 0,
                    limit_count INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS desktop_presence_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id INTEGER,
                    app_name TEXT,
                    window_title TEXT,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "long_term_memories", "memory_type", "TEXT NOT NULL DEFAULT 'general'")
            self._ensure_column(conn, "long_term_memories", "subject", "TEXT")
            self._ensure_column(conn, "long_term_memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "long_term_memories", "pinned", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "long_term_memories", "archived", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "long_term_memories", "pin_order", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "long_term_memories", "lane", "TEXT NOT NULL DEFAULT 'personal'")
            self._ensure_column(conn, "long_term_memories", "project_order", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "autonomous_jobs", "schedule_type", "TEXT NOT NULL DEFAULT 'interval'")
            self._ensure_column(conn, "autonomous_jobs", "run_hour", "INTEGER")
            self._ensure_column(conn, "autonomous_jobs", "run_minute", "INTEGER")
            self._ensure_column(conn, "autonomous_jobs", "timezone", "TEXT")
            self._ensure_column(conn, "autonomous_jobs", "metadata_json", "TEXT")
            self._ensure_column(conn, "project_workspaces", "focus", "TEXT")
            self._ensure_column(conn, "project_workspaces", "color", "TEXT")
            self._ensure_column(conn, "project_workspaces", "status", "TEXT NOT NULL DEFAULT 'active'")
            self._ensure_column(conn, "proactive_reminders", "priority", "TEXT NOT NULL DEFAULT 'normal'")
            self._ensure_column(conn, "proactive_reminders", "channel", "TEXT")
            self._ensure_column(conn, "proactive_reminders", "workspace_id", "INTEGER")
            self._ensure_column(conn, "proactive_reminders", "memory_id", "INTEGER")
            self._ensure_column(conn, "proactive_reminders", "delivered_at", "TEXT")
            self._ensure_column(conn, "proactive_reminders", "discord_delivered_at", "TEXT")
            self._ensure_column(conn, "proactive_reminders", "voice_announced_at", "TEXT")
            self._ensure_column(conn, "browser_sessions", "workspace_id", "INTEGER")
            self._ensure_column(conn, "browser_sessions", "provider", "TEXT")
            self._ensure_column(conn, "browser_sessions", "template_name", "TEXT")
            self._ensure_column(conn, "browser_sessions", "notes", "TEXT")
            self._ensure_column(conn, "browser_sessions", "health_status", "TEXT")
            self._ensure_column(conn, "browser_sessions", "health_details_json", "TEXT")
            self._ensure_column(conn, "browser_sessions", "last_health_check_at", "TEXT")
            self._ensure_column(conn, "browser_sessions", "last_used_at", "TEXT")
            self._ensure_column(conn, "mission_runs", "workspace_id", "INTEGER")
            self._ensure_column(conn, "mission_runs", "session_id", "TEXT")
            self._ensure_column(conn, "mission_runs", "goal", "TEXT")
            self._ensure_column(conn, "mission_runs", "auto_approve", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "mission_runs", "limit_count", "INTEGER NOT NULL DEFAULT 3")
            self._ensure_column(conn, "mission_runs", "summary", "TEXT")
            self._ensure_column(conn, "desktop_presence_snapshots", "workspace_id", "INTEGER")
            self._ensure_column(conn, "desktop_presence_snapshots", "app_name", "TEXT")
            self._ensure_column(conn, "desktop_presence_snapshots", "window_title", "TEXT")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _memory_type_weight(self, memory_type: str) -> float:
        weights = {
            "identity": 1.55,
            "profile": 1.4,
            "preference": 1.3,
            "project": 1.2,
            "workflow": 1.15,
            "fact": 1.0,
            "general": 1.0,
            "temporary": 0.7,
        }
        return float(weights.get(str(memory_type or "general").strip().lower(), 1.0))

    def _memory_decay_multiplier(
        self,
        *,
        created_at: Optional[str],
        last_accessed_at: Optional[str],
        memory_type: str,
        access_count: int,
    ) -> float:
        now = datetime.now(timezone.utc)
        created_dt = _parse_iso(created_at) or now
        accessed_dt = _parse_iso(last_accessed_at) or created_dt
        age_days = max(0.0, (now - created_dt).total_seconds() / 86400.0)
        idle_days = max(0.0, (now - accessed_dt).total_seconds() / 86400.0)

        type_decay = {
            "identity": 0.01,
            "profile": 0.015,
            "preference": 0.02,
            "project": 0.03,
            "workflow": 0.04,
            "fact": 0.045,
            "general": 0.05,
            "temporary": 0.08,
        }
        rate = float(type_decay.get(str(memory_type or "general").strip().lower(), 0.05))
        age_penalty = min(0.65, age_days * rate * 0.02)
        idle_penalty = min(0.75, idle_days * rate * 0.035)
        access_bonus = min(0.45, max(0, int(access_count)) * 0.03)
        return max(0.35, 1.0 - age_penalty - idle_penalty + access_bonus)

    def _memory_with_scores(self, row: sqlite3.Row) -> Dict[str, Any]:
        item = {
            "id": int(row["id"]),
            "session_id": row["session_id"],
            "content": row["content"],
            "tags": json_loads(row["tags_json"], []),
            "importance": int(row["importance"]),
            "memory_type": row["memory_type"] or "general",
            "subject": row["subject"],
            "access_count": int(row["access_count"] or 0),
            "pinned": bool(int(row["pinned"] or 0)),
            "archived": bool(int(row["archived"] or 0)),
            "pin_order": int(row["pin_order"] or 0),
            "lane": str(row["lane"] or "personal"),
            "project_order": int(row["project_order"] or 0),
            "source": row["source"],
            "created_at": row["created_at"],
            "last_accessed_at": row["last_accessed_at"],
        }
        type_weight = self._memory_type_weight(str(item["memory_type"]))
        decay_multiplier = self._memory_decay_multiplier(
            created_at=item["created_at"],
            last_accessed_at=item["last_accessed_at"],
            memory_type=str(item["memory_type"]),
            access_count=int(item["access_count"]),
        )
        pinned_bonus = 2.0 if bool(item.get("pinned")) else 0.0
        item["effective_importance"] = round((float(item["importance"]) * type_weight * decay_multiplier) + pinned_bonus, 3)
        item["stability_score"] = round(type_weight * decay_multiplier, 3)
        return item

    def _memory_sort_key(self, item: Dict[str, Any]) -> tuple[Any, ...]:
        pinned = bool(item.get("pinned"))
        pin_order = int(item.get("pin_order") or 0)
        is_project = str(item.get("memory_type") or "general") == "project"
        project_order = int(item.get("project_order") or 0)
        effective = float(item.get("effective_importance") or 0.0)
        return (
            0 if pinned else 1,
            pin_order if pinned and pin_order > 0 else 999999,
            0 if is_project and project_order > 0 else 1,
            project_order if is_project and project_order > 0 else 999999,
            -effective,
            -int(item.get("id") or 0),
        )

    def _sort_memory_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items, key=self._memory_sort_key)

    def _next_pin_order(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(pin_order), 0) AS max_pin_order FROM long_term_memories").fetchone()
        return int((row["max_pin_order"] if row else 0) or 0) + 1

    def _next_project_order(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(project_order), 0) AS max_project_order FROM long_term_memories").fetchone()
        return int((row["max_project_order"] if row else 0) or 0) + 1

    def _timezone_or_utc(self, timezone_name: Optional[str]) -> timezone | ZoneInfo:
        tz_name = str(timezone_name or "").strip() or "UTC"
        try:
            return ZoneInfo(tz_name)
        except Exception:
            return timezone.utc

    def _next_autonomous_job_run(
        self,
        *,
        now_iso: str,
        interval_minutes: int,
        schedule_type: str = "interval",
        run_hour: Optional[int] = None,
        run_minute: Optional[int] = None,
        timezone_name: Optional[str] = None,
    ) -> str:
        safe_schedule = str(schedule_type or "interval").strip().lower() or "interval"
        now_dt = _parse_iso(now_iso) or datetime.now(timezone.utc)
        if safe_schedule == "daily":
            tzinfo = self._timezone_or_utc(timezone_name)
            local_now = now_dt.astimezone(tzinfo)
            target = local_now.replace(
                hour=max(0, min(23, int(run_hour if run_hour is not None else 8))),
                minute=max(0, min(59, int(run_minute if run_minute is not None else 0))),
                second=0,
                microsecond=0,
            )
            if target <= local_now:
                target = target + timedelta(days=1)
            return target.astimezone(timezone.utc).isoformat()
        return _iso_add_minutes(now_dt.isoformat(), max(1, int(interval_minutes)))

    def _normalize_memory_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _find_memory_merge_candidate(
        self,
        conn: sqlite3.Connection,
        *,
        content: str,
        memory_type: str,
        subject: Optional[str],
    ) -> Optional[sqlite3.Row]:
        norm = self._normalize_memory_text(content)
        rows = conn.execute(
            """
            SELECT id, session_id, content, tags_json, importance, memory_type, subject, access_count, pinned, archived, pin_order, lane, project_order, source, created_at, last_accessed_at
            FROM long_term_memories
            WHERE memory_type = ? AND archived = 0
            ORDER BY id DESC
            LIMIT 100
            """,
            (str(memory_type or "general").strip().lower(),),
        ).fetchall()
        subject_raw = str(subject or "").strip()
        for row in rows:
            existing_subject = str(row["subject"] or "").strip()
            same_text = self._normalize_memory_text(str(row["content"] or "")) == norm
            same_subject = bool(subject_raw and existing_subject and subject_raw == existing_subject)
            if same_text:
                return row
            if same_subject and str(memory_type or "").strip().lower() in {"identity", "profile", "project", "workflow"}:
                return row
        return None

    def get_setting(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )
            conn.commit()

    def delete_setting(self, key: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            conn.commit()
            return int(cur.rowcount) > 0

    def get_json_setting(self, key: str, default: Any = None) -> Any:
        raw = self.get_setting(key)
        if raw in {None, ""}:
            return default
        return json_loads(raw, default)

    def set_json_setting(self, key: str, value: Any) -> None:
        self.set_setting(key, json_dumps(value))

    def save_mission_run(
        self,
        *,
        workspace_id: Optional[int],
        session_id: Optional[str],
        status: str,
        goal: Optional[str],
        auto_approve: bool,
        limit_count: int,
        summary: Optional[str],
        result: Dict[str, Any],
        mission_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = _now_iso()
        payload = json_dumps(result or {})
        with self._connect() as conn:
            if mission_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO mission_runs
                        (workspace_id, session_id, status, goal, auto_approve, limit_count, created_at, updated_at, summary, result_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        session_id,
                        str(status or "unknown").strip() or "unknown",
                        str(goal).strip() if goal is not None else None,
                        1 if auto_approve else 0,
                        max(1, int(limit_count or 1)),
                        now,
                        now,
                        str(summary).strip() if summary else None,
                        payload,
                    ),
                )
                mission_id = int(cur.lastrowid)
            else:
                conn.execute(
                    """
                    UPDATE mission_runs
                    SET workspace_id = ?,
                        session_id = ?,
                        status = ?,
                        goal = ?,
                        auto_approve = ?,
                        limit_count = ?,
                        updated_at = ?,
                        summary = ?,
                        result_json = ?
                    WHERE id = ?
                    """,
                    (
                        workspace_id,
                        session_id,
                        str(status or "unknown").strip() or "unknown",
                        str(goal).strip() if goal is not None else None,
                        1 if auto_approve else 0,
                        max(1, int(limit_count or 1)),
                        now,
                        str(summary).strip() if summary else None,
                        payload,
                        int(mission_id),
                    ),
                )
            conn.commit()
        item = self.get_mission_run(int(mission_id))
        if item is None:
            raise ValueError("mission run was not persisted")
        return item

    def get_mission_run(self, mission_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM mission_runs WHERE id = ?", (int(mission_id),)).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "workspace_id": int(row["workspace_id"]) if row["workspace_id"] is not None else None,
            "session_id": row["session_id"],
            "status": row["status"],
            "goal": row["goal"],
            "auto_approve": bool(int(row["auto_approve"] or 0)),
            "limit_count": int(row["limit_count"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "summary": row["summary"],
            "result": json_loads(row["result_json"], {}),
        }

    def list_mission_runs(self, *, limit: int = 20, workspace_id: Optional[int] = None) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        query = "SELECT * FROM mission_runs"
        values: List[Any] = []
        if workspace_id is not None:
            query += " WHERE workspace_id = ?"
            values.append(int(workspace_id))
        query += " ORDER BY id DESC LIMIT ?"
        values.append(lim)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(values)).fetchall()
        items = []
        for row in rows:
            items.append(
                {
                    "id": int(row["id"]),
                    "workspace_id": int(row["workspace_id"]) if row["workspace_id"] is not None else None,
                    "session_id": row["session_id"],
                    "status": row["status"],
                    "goal": row["goal"],
                    "auto_approve": bool(int(row["auto_approve"] or 0)),
                    "limit_count": int(row["limit_count"] or 0),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "summary": row["summary"],
                    "result": json_loads(row["result_json"], {}),
                }
            )
        return items

    def save_desktop_presence_snapshot(
        self,
        *,
        summary: str,
        details: Dict[str, Any],
        workspace_id: Optional[int] = None,
        app_name: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO desktop_presence_snapshots
                    (workspace_id, app_name, window_title, summary, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    str(app_name).strip() if app_name else None,
                    str(window_title).strip() if window_title else None,
                    str(summary or "").strip() or "Desktop presence snapshot",
                    json_dumps(details or {}),
                    now,
                ),
            )
            conn.commit()
            snapshot_id = int(cur.lastrowid)
        item = self.get_desktop_presence_snapshot(snapshot_id)
        if item is None:
            raise ValueError("desktop presence snapshot was not persisted")
        return item

    def get_desktop_presence_snapshot(self, snapshot_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM desktop_presence_snapshots WHERE id = ?", (int(snapshot_id),)).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "workspace_id": int(row["workspace_id"]) if row["workspace_id"] is not None else None,
            "app_name": row["app_name"],
            "window_title": row["window_title"],
            "summary": row["summary"],
            "details": json_loads(row["details_json"], {}),
            "created_at": row["created_at"],
        }

    def latest_desktop_presence_snapshot(self, *, workspace_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM desktop_presence_snapshots"
        values: List[Any] = []
        if workspace_id is not None:
            query += " WHERE workspace_id = ?"
            values.append(int(workspace_id))
        query += " ORDER BY id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, tuple(values)).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "workspace_id": int(row["workspace_id"]) if row["workspace_id"] is not None else None,
            "app_name": row["app_name"],
            "window_title": row["window_title"],
            "summary": row["summary"],
            "details": json_loads(row["details_json"], {}),
            "created_at": row["created_at"],
        }

    def log_tool_execution(
        self,
        *,
        tool_name: str,
        status: str,
        confidence: Optional[float],
        verification: Dict[str, Any],
        detail: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO tool_execution_log
                    (session_id, tool_name, status, confidence, verification_json, detail_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(tool_name or "").strip() or "unknown",
                    str(status or "").strip() or "unknown",
                    float(confidence) if confidence is not None else None,
                    json_dumps(verification or {}),
                    json_dumps(detail or {}),
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_tool_executions(
        self,
        *,
        limit: int = 50,
        tool_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        where: List[str] = []
        values: List[Any] = []
        if tool_name:
            where.append("tool_name = ?")
            values.append(str(tool_name).strip())
        if session_id:
            where.append("session_id = ?")
            values.append(str(session_id).strip())
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, tool_name, status, confidence, verification_json, detail_json, created_at
                FROM tool_execution_log
                {clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "session_id": r["session_id"],
                    "tool_name": r["tool_name"],
                    "status": r["status"],
                    "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
                    "verification": json_loads(r["verification_json"], {}),
                    "detail": json_loads(r["detail_json"], {}),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def tool_reliability_report(self, *, limit: int = 100) -> Dict[str, Any]:
        items = self.list_tool_executions(limit=limit)
        by_tool: Dict[str, Dict[str, Any]] = {}
        for item in items:
            key = str(item.get("tool_name") or "unknown")
            bucket = by_tool.setdefault(
                key,
                {
                    "tool_name": key,
                    "runs": 0,
                    "verified": 0,
                    "failed": 0,
                    "preview": 0,
                    "confidences": [],
                },
            )
            bucket["runs"] += 1
            status = str(item.get("status") or "")
            if status == "verified":
                bucket["verified"] += 1
            elif status == "failed":
                bucket["failed"] += 1
            elif status == "preview":
                bucket["preview"] += 1
            conf = item.get("confidence")
            if isinstance(conf, (int, float)):
                bucket["confidences"].append(float(conf))
        tools: List[Dict[str, Any]] = []
        for bucket in by_tool.values():
            runs = max(1, int(bucket["runs"]))
            avg_conf = (
                sum(bucket["confidences"]) / len(bucket["confidences"])
                if bucket["confidences"]
                else None
            )
            tools.append(
                {
                    "tool_name": bucket["tool_name"],
                    "runs": runs,
                    "verified": int(bucket["verified"]),
                    "failed": int(bucket["failed"]),
                    "preview": int(bucket["preview"]),
                    "verification_rate_pct": round((int(bucket["verified"]) / runs) * 100.0, 1),
                    "avg_confidence": round(avg_conf, 3) if avg_conf is not None else None,
                }
            )
        tools.sort(key=lambda item: (-item["runs"], item["tool_name"]))
        overall_avg = [item["confidence"] for item in items if isinstance(item.get("confidence"), (int, float))]
        return {
            "total_runs": len(items),
            "verified_runs": len([i for i in items if i.get("status") == "verified"]),
            "failed_runs": len([i for i in items if i.get("status") == "failed"]),
            "preview_runs": len([i for i in items if i.get("status") == "preview"]),
            "avg_confidence": round(sum(overall_avg) / len(overall_avg), 3) if overall_avg else None,
            "tools": tools[:20],
            "recent": items[:20],
        }

    def create_task(self, task: str, session_id: Optional[str] = None) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (session_id, task, status, created_at, updated_at, note)
                VALUES (?, ?, 'open', ?, ?, NULL)
                """,
                (session_id, task, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_tasks(
        self,
        *,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        where: List[str] = []
        values: List[Any] = []
        if session_id:
            where.append("session_id = ?")
            values.append(session_id)
        if status:
            where.append("status = ?")
            values.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT id, session_id, task, status, created_at, updated_at, note
            FROM tasks
            {clause}
            ORDER BY id DESC
            LIMIT ?
        """
        values.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
            return [dict(r) for r in rows]

    def update_task_status(self, task_id: int, status: str, note: Optional[str] = None) -> bool:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status = ?, updated_at = ?, note = ?
                WHERE id = ?
                """,
                (status, now, note, task_id),
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def create_goal_run(self, *, task_id: int, session_id: Optional[str], goal: str, plan: List[str]) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO goal_runs (task_id, session_id, goal, status, created_at, updated_at, plan_json, steps_json, result_json)
                VALUES (?, ?, ?, 'running', ?, ?, ?, '[]', NULL)
                """,
                (task_id, session_id, goal, now, now, json_dumps(plan)),
            )
            conn.commit()
            return int(cur.lastrowid)

    def update_goal_run(
        self,
        run_id: int,
        *,
        status: str,
        steps: List[Dict[str, Any]],
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE goal_runs
                SET status = ?, updated_at = ?, steps_json = ?, result_json = ?
                WHERE id = ?
                """,
                (status, now, json_dumps(steps), json_dumps(result) if result is not None else None, run_id),
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def list_goal_runs(self, *, limit: int = 20, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        where = ""
        values: List[Any] = []
        if session_id:
            where = "WHERE session_id = ?"
            values.append(session_id)
        values.append(limit)
        sql = f"""
            SELECT id, task_id, session_id, goal, status, created_at, updated_at, plan_json, steps_json, result_json
            FROM goal_runs
            {where}
            ORDER BY id DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "task_id": int(r["task_id"]) if r["task_id"] is not None else None,
                    "session_id": r["session_id"],
                    "goal": r["goal"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "plan": json_loads(r["plan_json"], []),
                    "steps": json_loads(r["steps_json"], []),
                    "result": json_loads(r["result_json"], None),
                }
                for r in rows
            ]

    def create_goal_schedule(
        self,
        *,
        goal: str,
        interval_minutes: int,
        session_id: Optional[str] = None,
        auto_approve: bool = False,
        enabled: bool = True,
    ) -> int:
        interval = max(1, min(int(interval_minutes), 10080))
        now = _now_iso()
        next_run = _iso_add_minutes(now, interval)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO goal_schedules
                    (goal, session_id, interval_minutes, auto_approve, enabled, created_at, updated_at, last_run_at, next_run_at, last_status, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
                """,
                (goal, session_id, interval, 1 if auto_approve else 0, 1 if enabled else 0, now, now, next_run),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_goal_schedules(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, goal, session_id, interval_minutes, auto_approve, enabled, created_at, updated_at, last_run_at, next_run_at, last_status, last_error
                FROM goal_schedules
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "goal": r["goal"],
                    "session_id": r["session_id"],
                    "interval_minutes": int(r["interval_minutes"]),
                    "auto_approve": bool(r["auto_approve"]),
                    "enabled": bool(r["enabled"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "last_run_at": r["last_run_at"],
                    "next_run_at": r["next_run_at"],
                    "last_status": r["last_status"],
                    "last_error": r["last_error"],
                }
                for r in rows
            ]

    def update_goal_schedule(
        self,
        schedule_id: int,
        *,
        enabled: Optional[bool] = None,
        interval_minutes: Optional[int] = None,
        auto_approve: Optional[bool] = None,
        goal: Optional[str] = None,
    ) -> bool:
        current = None
        with self._connect() as conn:
            current = conn.execute(
                """
                SELECT id, interval_minutes
                FROM goal_schedules
                WHERE id = ?
                """,
                (schedule_id,),
            ).fetchone()
            if current is None:
                return False

            updates: List[str] = []
            values: List[Any] = []
            now = _now_iso()

            if enabled is not None:
                updates.append("enabled = ?")
                values.append(1 if enabled else 0)
            if interval_minutes is not None:
                interval = max(1, min(int(interval_minutes), 10080))
                updates.append("interval_minutes = ?")
                values.append(interval)
                updates.append("next_run_at = ?")
                values.append(_iso_add_minutes(now, interval))
            if auto_approve is not None:
                updates.append("auto_approve = ?")
                values.append(1 if auto_approve else 0)
            if goal is not None:
                updates.append("goal = ?")
                values.append(goal)

            updates.append("updated_at = ?")
            values.append(now)
            values.append(schedule_id)

            conn.execute(
                f"""
                UPDATE goal_schedules
                SET {', '.join(updates)}
                WHERE id = ?
                """,
                values,
            )
            conn.commit()
            return True

    def claim_due_goal_schedules(self, *, now_iso: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        now = now_iso or _now_iso()
        lim = max(1, min(int(limit), 100))
        claimed: List[Dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, goal, session_id, interval_minutes, auto_approve, enabled, next_run_at
                FROM goal_schedules
                WHERE enabled = 1 AND next_run_at <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (now, lim),
            ).fetchall()
            for r in rows:
                next_run = _iso_add_minutes(now, int(r["interval_minutes"]))
                conn.execute(
                    """
                    UPDATE goal_schedules
                    SET last_run_at = ?, next_run_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, next_run, now, int(r["id"])),
                )
                claimed.append(
                    {
                        "id": int(r["id"]),
                        "goal": r["goal"],
                        "session_id": r["session_id"],
                        "interval_minutes": int(r["interval_minutes"]),
                        "auto_approve": bool(r["auto_approve"]),
                    }
                )
            conn.commit()
        return claimed

    def mark_goal_schedule_result(self, schedule_id: int, *, ok: bool, error: Optional[str] = None) -> bool:
        now = _now_iso()
        status = "ok" if ok else "failed"
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE goal_schedules
                SET last_status = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, now, schedule_id),
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def add_quantum_event(
        self,
        *,
        event_type: str,
        session_id: Optional[str] = None,
        measurement_basis: Optional[str] = None,
        outcome: Optional[int] = None,
        measurement_probability: Optional[float] = None,
        entanglement_strength: Optional[float] = None,
        correlation_coefficient: Optional[float] = None,
        states: Optional[List[str]] = None,
    ) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO quantum_events
                    (event_type, session_id, measurement_basis, outcome, measurement_probability, entanglement_strength, correlation_coefficient, states_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    session_id,
                    measurement_basis,
                    outcome,
                    measurement_probability,
                    entanglement_strength,
                    correlation_coefficient,
                    json_dumps(states or []),
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_quantum_events(
        self,
        *,
        limit: int = 100,
        event_type: Optional[str] = None,
        since_hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        where: List[str] = []
        values: List[Any] = []
        if event_type:
            where.append("event_type = ?")
            values.append(event_type)
        if since_hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=int(since_hours))).isoformat()
            where.append("created_at >= ?")
            values.append(cutoff)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT id, event_type, session_id, measurement_basis, outcome, measurement_probability,
                   entanglement_strength, correlation_coefficient, states_json, created_at
            FROM quantum_events
            {clause}
            ORDER BY id DESC
            LIMIT ?
        """
        values.append(lim)
        with self._connect() as conn:
            rows = conn.execute(sql, values).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "event_type": r["event_type"],
                    "session_id": r["session_id"],
                    "measurement_basis": r["measurement_basis"],
                    "outcome": r["outcome"],
                    "measurement_probability": r["measurement_probability"],
                    "entanglement_strength": r["entanglement_strength"],
                    "correlation_coefficient": r["correlation_coefficient"],
                    "states": json_loads(r["states_json"], []),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def quantum_stats(self, *, hours: int = 24) -> Dict[str, Any]:
        h = max(1, min(int(hours), 24 * 365))
        events = self.list_quantum_events(limit=5000, since_hours=h)
        measurements = [e for e in events if e.get("event_type") == "measurement"]
        entangles = [e for e in events if e.get("event_type") == "entangle"]
        supers = [e for e in events if e.get("event_type") == "superposition"]

        outcome_counts = {"0": 0, "1": 0, "other": 0}
        for m in measurements:
            if m.get("outcome") == 0:
                outcome_counts["0"] += 1
            elif m.get("outcome") == 1:
                outcome_counts["1"] += 1
            else:
                outcome_counts["other"] += 1

        def _avg(values: List[Optional[float]]) -> Optional[float]:
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            if not nums:
                return None
            return sum(nums) / len(nums)

        return {
            "window_hours": h,
            "total_events": len(events),
            "measurements": len(measurements),
            "entangles": len(entangles),
            "superpositions": len(supers),
            "measurement_outcomes": outcome_counts,
            "avg_measurement_probability": _avg([m.get("measurement_probability") for m in measurements]),
            "avg_entanglement_strength": _avg([e.get("entanglement_strength") for e in entangles]),
            "avg_correlation_coefficient": _avg([e.get("correlation_coefficient") for e in entangles]),
            "avg_superposition_states": _avg([len(s.get("states") or []) for s in supers]),
        }

    def create_quantum_decipher_snapshot(self, payload: Dict[str, Any]) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO quantum_decipher_snapshots
                    (window_hours, events_analyzed, confidence_pct, signals_json, patterns_json, recommendations_json, stats_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(payload.get("window_hours") or 24),
                    int(payload.get("events_analyzed") or 0),
                    float(payload.get("confidence_pct") or 0.0),
                    json_dumps(payload.get("signals") or {}),
                    json_dumps(payload.get("patterns") or []),
                    json_dumps(payload.get("recommendations") or []),
                    json_dumps(payload.get("stats") or {}),
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_quantum_decipher_snapshots(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, window_hours, events_analyzed, confidence_pct, signals_json, patterns_json, recommendations_json, stats_json, created_at
                FROM quantum_decipher_snapshots
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "window_hours": int(r["window_hours"]),
                    "events_analyzed": int(r["events_analyzed"]),
                    "confidence_pct": float(r["confidence_pct"]),
                    "signals": json_loads(r["signals_json"], {}),
                    "patterns": json_loads(r["patterns_json"], []),
                    "recommendations": json_loads(r["recommendations_json"], []),
                    "stats": json_loads(r["stats_json"], {}),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def add_quantum_ops_audit(self, *, op_type: str, status: str, details: Dict[str, Any]) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO quantum_ops_audit
                    (op_type, status, details_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (op_type, status, json_dumps(details), now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_quantum_ops_audit(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, op_type, status, details_json, created_at
                FROM quantum_ops_audit
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "op_type": r["op_type"],
                    "status": r["status"],
                    "details": json_loads(r["details_json"], {}),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def add_quantum_annotation(
        self,
        *,
        item_type: str,
        item_id: str,
        note: str,
        author: Optional[str] = None,
    ) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO quantum_annotations
                    (item_type, item_id, author, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (item_type, item_id, author, note, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_quantum_annotations(
        self,
        *,
        item_type: Optional[str] = None,
        item_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 1000))
        where: List[str] = []
        vals: List[Any] = []
        if item_type:
            where.append("item_type = ?")
            vals.append(item_type)
        if item_id:
            where.append("item_id = ?")
            vals.append(item_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
            SELECT id, item_type, item_id, author, note, created_at
            FROM quantum_annotations
            {clause}
            ORDER BY id DESC
            LIMIT ?
        """
        vals.append(lim)
        with self._connect() as conn:
            rows = conn.execute(sql, vals).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "item_type": r["item_type"],
                    "item_id": r["item_id"],
                    "author": r["author"],
                    "note": r["note"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def create_long_term_memory(
        self,
        *,
        content: str,
        tags: Optional[List[str]] = None,
        importance: int = 3,
        memory_type: str = "general",
        subject: Optional[str] = None,
        pinned: bool = False,
        lane: Optional[str] = None,
        source: str = "user",
        session_id: Optional[str] = None,
    ) -> int:
        now = _now_iso()
        importance = max(1, min(int(importance), 5))
        safe_tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:12]
        safe_type = str(memory_type or "general").strip().lower() or "general"
        safe_subject = str(subject or "").strip() or None
        lane_value = str(lane or "").strip().lower()
        if lane_value not in {"critical", "project", "personal"}:
            lane_value = "project" if safe_type == "project" else "personal"
        with self._connect() as conn:
            existing = self._find_memory_merge_candidate(
                conn,
                content=content.strip(),
                memory_type=safe_type,
                subject=safe_subject,
            )
            if existing is not None:
                existing_tags = [str(t).strip() for t in json_loads(existing["tags_json"], []) if str(t).strip()]
                merged_tags: List[str] = []
                for tag in existing_tags + safe_tags:
                    if tag and tag not in merged_tags:
                        merged_tags.append(tag)
                merged_content = str(existing["content"] or "").strip()
                incoming = content.strip()
                if len(incoming) > len(merged_content):
                    merged_content = incoming
                pin_order = int(existing["pin_order"] or 0)
                merged_pinned = bool(existing["pinned"]) or bool(pinned)
                project_order = int(existing["project_order"] or 0)
                if merged_pinned and pin_order <= 0:
                    pin_order = self._next_pin_order(conn)
                if safe_type == "project" and project_order <= 0:
                    project_order = self._next_project_order(conn)
                conn.execute(
                    """
                    UPDATE long_term_memories
                    SET content = ?, tags_json = ?, importance = ?, source = ?, subject = ?, pinned = ?, pin_order = ?, lane = ?, project_order = ?, access_count = COALESCE(access_count, 0) + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_content,
                        json_dumps(merged_tags[:12]),
                        max(int(existing["importance"] or 1), importance),
                        source.strip() or str(existing["source"] or "user"),
                        safe_subject or existing["subject"],
                        1 if merged_pinned else 0,
                        pin_order if merged_pinned else 0,
                        lane_value or str(existing["lane"] or "personal"),
                        project_order if safe_type == "project" else int(existing["project_order"] or 0),
                        now,
                        int(existing["id"]),
                    ),
                )
                conn.commit()
                return int(existing["id"])
            pin_order = self._next_pin_order(conn) if bool(pinned) else 0
            project_order = self._next_project_order(conn) if safe_type == "project" else 0
            cur = conn.execute(
                """
                INSERT INTO long_term_memories
                    (session_id, content, tags_json, importance, memory_type, subject, access_count, pinned, pin_order, lane, project_order, source, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    content.strip(),
                    json_dumps(safe_tags),
                    importance,
                    safe_type,
                    safe_subject,
                    1 if bool(pinned) else 0,
                    pin_order,
                    lane_value,
                    project_order,
                    source.strip() or "user",
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def consolidate_long_term_memories(self, *, limit: int = 500) -> Dict[str, Any]:
        items = self.list_long_term_memories(limit=limit)
        merged = 0
        seen_keys: Dict[str, Dict[str, Any]] = {}
        for item in items:
            memory_type = str(item.get("memory_type") or "general").strip().lower()
            subject = str(item.get("subject") or "").strip().lower()
            content_key = self._normalize_memory_text(str(item.get("content") or ""))
            key = f"{memory_type}|{subject}|{content_key}"
            if key not in seen_keys:
                seen_keys[key] = item
                continue
            primary = seen_keys[key]
            duplicate_id = int(item["id"])
            primary_id = int(primary["id"])
            merged_tags: List[str] = []
            for tag in list(primary.get("tags") or []) + list(item.get("tags") or []):
                t = str(tag).strip()
                if t and t not in merged_tags:
                    merged_tags.append(t)
            merged_content = str(primary.get("content") or "")
            incoming = str(item.get("content") or "")
            if len(incoming) > len(merged_content):
                merged_content = incoming
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE long_term_memories
                    SET content = ?, tags_json = ?, importance = ?, access_count = ?, pinned = ?, pin_order = ?, lane = ?, project_order = ?, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_content,
                        json_dumps(merged_tags[:12]),
                        max(int(primary.get("importance") or 1), int(item.get("importance") or 1)),
                        int(primary.get("access_count") or 0) + int(item.get("access_count") or 0),
                        1 if (bool(primary.get("pinned")) or bool(item.get("pinned"))) else 0,
                        min(
                            [value for value in [int(primary.get("pin_order") or 0), int(item.get("pin_order") or 0)] if value > 0] or [0]
                        ),
                        str(primary.get("lane") or item.get("lane") or "personal"),
                        min(
                            [value for value in [int(primary.get("project_order") or 0), int(item.get("project_order") or 0)] if value > 0] or [0]
                        ),
                        max(str(primary.get("last_accessed_at") or ""), str(item.get("last_accessed_at") or "")) or _now_iso(),
                        primary_id,
                    ),
                )
                conn.execute("DELETE FROM long_term_memories WHERE id = ?", (duplicate_id,))
                conn.commit()
            primary["tags"] = merged_tags[:12]
            primary["importance"] = max(int(primary.get("importance") or 1), int(item.get("importance") or 1))
            primary["access_count"] = int(primary.get("access_count") or 0) + int(item.get("access_count") or 0)
            primary["content"] = merged_content
            primary["pinned"] = bool(primary.get("pinned")) or bool(item.get("pinned"))
            primary["pin_order"] = min(
                [value for value in [int(primary.get("pin_order") or 0), int(item.get("pin_order") or 0)] if value > 0] or [0]
            )
            primary["lane"] = str(primary.get("lane") or item.get("lane") or "personal")
            primary["project_order"] = min(
                [value for value in [int(primary.get("project_order") or 0), int(item.get("project_order") or 0)] if value > 0] or [0]
            )
            merged += 1
        return {"ok": True, "merged": merged, "items": self.list_long_term_memories(limit=min(limit, 100))}

    def memory_overview(self, *, limit_per_group: int = 6, since_hours: Optional[int] = None, archived: bool = False) -> Dict[str, Any]:
        items = self.list_long_term_memories(limit=400, since_hours=since_hours, archived=archived)
        archived_items = self.list_long_term_memories(limit=100, archived=True) if not archived else items
        counts_by_type: Dict[str, int] = {}
        for item in items:
            key = str(item.get("memory_type") or "general")
            counts_by_type[key] = counts_by_type.get(key, 0) + 1

        def _top(memory_types: List[str], limit: int) -> List[Dict[str, Any]]:
            filtered = [i for i in items if str(i.get("memory_type") or "general") in memory_types]
            return filtered[: max(1, limit)]

        profile = _top(["identity", "profile", "preference"], limit_per_group)
        projects = _top(["project"], limit_per_group)
        workflows = _top(["workflow"], limit_per_group)
        pinned = [i for i in items if bool(i.get("pinned"))][: max(1, limit_per_group)]
        important = items[: max(1, limit_per_group)]
        pinned_lanes = {
            "critical": [i for i in pinned if str(i.get("lane") or "personal") == "critical"][: max(1, limit_per_group)],
            "project": [i for i in pinned if str(i.get("lane") or "personal") == "project"][: max(1, limit_per_group)],
            "personal": [i for i in pinned if str(i.get("lane") or "personal") == "personal"][: max(1, limit_per_group)],
        }

        subject_map: Dict[str, Dict[str, Any]] = {}
        for item in items:
            subject = str(item.get("subject") or "").strip()
            if not subject:
                continue
            if subject not in subject_map:
                subject_map[subject] = item

        return {
            "total": len(items),
            "counts_by_type": counts_by_type,
            "profile": profile,
            "projects": projects,
            "workflows": workflows,
            "pinned": pinned,
            "pinned_lanes": pinned_lanes,
            "important": important,
            "since_hours": since_hours,
            "archived": archived,
            "archived_count": len(archived_items),
            "subjects": [
                {
                    "subject": subject,
                    "memory_type": item.get("memory_type"),
                    "content": item.get("content"),
                    "effective_importance": item.get("effective_importance"),
                }
                for subject, item in list(subject_map.items())[: max(1, limit_per_group * 2)]
            ],
        }

    def memory_briefing(self, *, period: str = "morning", recent_project_hours: Optional[int] = None) -> Dict[str, Any]:
        period_name = str(period or "morning").strip().lower()
        if period_name not in {"morning", "evening"}:
            period_name = "morning"
        recent_hours = int(recent_project_hours or (72 if period_name == "morning" else 24))
        pinned = self.list_long_term_memories(limit=8, pinned=True, archived=False)
        recent_projects = self.list_long_term_memories(
            limit=6,
            memory_type="project",
            since_hours=recent_hours,
            archived=False,
        )
        profile = self.list_long_term_memories(limit=4, memory_type="identity", archived=False)
        if not profile:
            profile = self.list_long_term_memories(limit=4, memory_type="profile", archived=False)

        lines: List[str] = []
        if period_name == "morning":
            lines.append("Good morning. Here is your Jarvis briefing.")
            if pinned:
                lines.append("Priority memory:")
                for item in pinned[:3]:
                    lines.append(str(item.get("content") or "").strip())
            if recent_projects:
                lines.append("Recent project focus:")
                for item in recent_projects[:2]:
                    lines.append(str(item.get("content") or "").strip())
            if profile:
                lines.append("Profile reminder:")
                lines.append(str(profile[0].get("content") or "").strip())
        else:
            lines.append("Good evening. Here is your Jarvis wrap-up.")
            if pinned:
                lines.append("Still-priority memory:")
                for item in pinned[:2]:
                    lines.append(str(item.get("content") or "").strip())
            if recent_projects:
                lines.append("Project activity to carry forward:")
                for item in recent_projects[:3]:
                    lines.append(str(item.get("content") or "").strip())
            else:
                lines.append("No recent project memory was active in the selected window.")

        text = " ".join([line for line in lines if line]).strip()
        return {
            "period": period_name,
            "text": text,
            "pinned": pinned[:4],
            "recent_projects": recent_projects[:4],
            "profile": profile[:2],
            "recent_project_hours": recent_hours,
        }

    def memory_context_bundle(self, *, query: str, limit: int = 6, since_hours: Optional[int] = None) -> Dict[str, Any]:
        lim = max(2, min(int(limit), 20))
        search_hits = self.search_long_term_memories(query=query, limit=lim)
        overview = self.memory_overview(limit_per_group=max(2, lim // 2), since_hours=since_hours)
        chosen: List[Dict[str, Any]] = []
        seen_ids: set[int] = set()

        def _add(items: List[Dict[str, Any]], cap: int) -> None:
            for item in items[:cap]:
                iid = int(item.get("id") or 0)
                if iid and iid not in seen_ids:
                    chosen.append(item)
                    seen_ids.add(iid)

        _add(search_hits, lim)
        _add(overview.get("pinned") or [], 3)
        _add(overview.get("profile") or [], 3)
        _add(overview.get("projects") or [], 3)
        _add(overview.get("workflows") or [], 2)
        chosen = self._sort_memory_items(chosen)
        return {
            "query": query,
            "items": chosen[:lim],
            "profile": overview.get("profile") or [],
            "projects": overview.get("projects") or [],
            "workflows": overview.get("workflows") or [],
        }

    def create_project_workspace(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        focus: Optional[str] = None,
        color: Optional[str] = None,
        status: str = "active",
        memory_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        now = _now_iso()
        safe_name = str(name or "").strip()
        if not safe_name:
            raise ValueError("workspace name is required")
        safe_status = str(status or "active").strip().lower()
        if safe_status not in {"active", "paused", "archived"}:
            safe_status = "active"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM project_workspaces WHERE LOWER(name) = LOWER(?)",
                (safe_name,),
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO project_workspaces
                        (name, description, focus, color, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        safe_name,
                        str(description or "").strip() or None,
                        str(focus or "").strip() or None,
                        str(color or "").strip() or None,
                        safe_status,
                        now,
                        now,
                    ),
                )
                workspace_id = int(cur.lastrowid)
            else:
                workspace_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE project_workspaces
                    SET description = ?, focus = ?, color = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        str(description or "").strip() or None,
                        str(focus or "").strip() or None,
                        str(color or "").strip() or None,
                        safe_status,
                        now,
                        workspace_id,
                    ),
                )
            if memory_ids is not None:
                self._set_workspace_memories_conn(conn, workspace_id, memory_ids)
            conn.commit()
        return self.get_project_workspace(workspace_id) or {}

    def _set_workspace_memories_conn(self, conn: sqlite3.Connection, workspace_id: int, memory_ids: List[int]) -> None:
        safe_ids = [int(i) for i in memory_ids if int(i) > 0][:200]
        conn.execute("DELETE FROM workspace_memory_map WHERE workspace_id = ?", (int(workspace_id),))
        now = _now_iso()
        for memory_id in safe_ids:
            conn.execute(
                """
                INSERT OR IGNORE INTO workspace_memory_map (workspace_id, memory_id, created_at)
                VALUES (?, ?, ?)
                """,
                (int(workspace_id), memory_id, now),
            )

    def set_project_workspace_memories(self, workspace_id: int, memory_ids: List[int]) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM project_workspaces WHERE id = ?", (int(workspace_id),)).fetchone()
            if row is None:
                raise ValueError("workspace not found")
            self._set_workspace_memories_conn(conn, int(workspace_id), memory_ids)
            conn.execute(
                "UPDATE project_workspaces SET updated_at = ? WHERE id = ?",
                (_now_iso(), int(workspace_id)),
            )
            conn.commit()
        return self.get_project_workspace(int(workspace_id)) or {}

    def list_project_workspaces(self, *, limit: int = 50, include_archived: bool = False) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        where = "" if include_archived else "WHERE status != 'archived'"
        active_workspace_id = self.get_active_workspace_id()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT pw.id, pw.name, pw.description, pw.focus, pw.color, pw.status, pw.created_at, pw.updated_at,
                       COUNT(wmm.memory_id) AS memory_count
                FROM project_workspaces pw
                LEFT JOIN workspace_memory_map wmm ON wmm.workspace_id = pw.id
                {where}
                GROUP BY pw.id
                ORDER BY CASE WHEN pw.status = 'active' THEN 0 WHEN pw.status = 'paused' THEN 1 ELSE 2 END,
                         pw.updated_at DESC, pw.id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            items = []
            for row in rows:
                workspace_id = int(row["id"])
                items.append(
                    {
                        "id": workspace_id,
                        "name": row["name"],
                        "description": row["description"],
                        "focus": row["focus"],
                        "color": row["color"] or "#2ee6d6",
                        "status": row["status"] or "active",
                        "memory_count": int(row["memory_count"] or 0),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "active": workspace_id == active_workspace_id,
                    }
                )
            return items

    def get_project_workspace(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, description, focus, color, status, created_at, updated_at
                FROM project_workspaces
                WHERE id = ?
                """,
                (int(workspace_id),),
            ).fetchone()
            if row is None:
                return None
            memory_rows = conn.execute(
                """
                SELECT ltm.id, ltm.session_id, ltm.content, ltm.tags_json, ltm.importance, ltm.memory_type, ltm.subject,
                       ltm.access_count, ltm.pinned, ltm.archived, ltm.pin_order, ltm.lane, ltm.project_order,
                       ltm.source, ltm.created_at, ltm.last_accessed_at
                FROM workspace_memory_map wmm
                JOIN long_term_memories ltm ON ltm.id = wmm.memory_id
                WHERE wmm.workspace_id = ? AND ltm.archived = 0
                ORDER BY CASE WHEN ltm.memory_type = 'project' THEN 0 ELSE 1 END,
                         CASE WHEN ltm.project_order > 0 THEN ltm.project_order ELSE 999999 END ASC,
                         CASE WHEN ltm.pin_order > 0 THEN ltm.pin_order ELSE 999999 END ASC,
                         ltm.id DESC
                """,
                (int(workspace_id),),
            ).fetchall()
            memories = [self._memory_with_scores(r) for r in memory_rows]
            active_workspace_id = self.get_active_workspace_id()
            return {
                "id": int(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "focus": row["focus"],
                "color": row["color"] or "#2ee6d6",
                "status": row["status"] or "active",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "active": int(row["id"]) == active_workspace_id,
                "memory_count": len(memories),
                "memories": memories[:24],
            }

    def update_project_workspace(
        self,
        workspace_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        focus: Optional[str] = None,
        color: Optional[str] = None,
        status: Optional[str] = None,
        memory_ids: Optional[List[int]] = None,
    ) -> bool:
        updates: List[str] = []
        values: List[Any] = []
        if name is not None:
            updates.append("name = ?")
            values.append(str(name).strip())
        if description is not None:
            updates.append("description = ?")
            values.append(str(description).strip() or None)
        if focus is not None:
            updates.append("focus = ?")
            values.append(str(focus).strip() or None)
        if color is not None:
            updates.append("color = ?")
            values.append(str(color).strip() or None)
        if status is not None:
            safe_status = str(status).strip().lower()
            if safe_status not in {"active", "paused", "archived"}:
                safe_status = "active"
            updates.append("status = ?")
            values.append(safe_status)
        if not updates and memory_ids is None:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM project_workspaces WHERE id = ?", (int(workspace_id),)).fetchone()
            if row is None:
                return False
            if updates:
                updates.append("updated_at = ?")
                values.append(_now_iso())
                values.append(int(workspace_id))
                conn.execute(f"UPDATE project_workspaces SET {', '.join(updates)} WHERE id = ?", values)
            if memory_ids is not None:
                self._set_workspace_memories_conn(conn, int(workspace_id), memory_ids)
            conn.commit()
            return True

    def set_active_workspace(self, workspace_id: Optional[int]) -> Dict[str, Any]:
        wid = int(workspace_id) if workspace_id is not None else 0
        if wid > 0 and self.get_project_workspace(wid) is None:
            raise ValueError("workspace not found")
        self.set_setting("active_workspace_id", str(wid or ""))
        return {
            "active_workspace_id": wid or None,
            "workspace": self.get_project_workspace(wid) if wid > 0 else None,
        }

    def get_active_workspace_id(self) -> Optional[int]:
        raw = str(self.get_setting("active_workspace_id") or "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except Exception:
            return None
        return value if value > 0 else None

    def get_workspace_policy(self, workspace_id: Optional[int]) -> Dict[str, Any]:
        default_policy = {
            "workspace_id": int(workspace_id) if workspace_id else None,
            "browser_allowed": True,
            "desktop_allowed": True,
            "shell_allowed": False,
            "repo_write_allowed": False,
            "require_confirmation": False,
        }
        if not workspace_id:
            return default_policy
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT workspace_id, browser_allowed, desktop_allowed, shell_allowed, repo_write_allowed, require_confirmation, updated_at
                FROM workspace_policies
                WHERE workspace_id = ?
                """,
                (int(workspace_id),),
            ).fetchone()
            if row is None:
                return default_policy
            return {
                "workspace_id": int(row["workspace_id"]),
                "browser_allowed": bool(row["browser_allowed"]),
                "desktop_allowed": bool(row["desktop_allowed"]),
                "shell_allowed": bool(row["shell_allowed"]),
                "repo_write_allowed": bool(row["repo_write_allowed"]),
                "require_confirmation": bool(row["require_confirmation"]),
                "updated_at": row["updated_at"],
            }

    def set_workspace_policy(
        self,
        workspace_id: int,
        *,
        browser_allowed: bool,
        desktop_allowed: bool,
        shell_allowed: bool,
        repo_write_allowed: bool,
        require_confirmation: bool,
    ) -> Dict[str, Any]:
        if self.get_project_workspace(int(workspace_id)) is None:
            raise ValueError("workspace not found")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workspace_policies
                    (workspace_id, browser_allowed, desktop_allowed, shell_allowed, repo_write_allowed, require_confirmation, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    browser_allowed = excluded.browser_allowed,
                    desktop_allowed = excluded.desktop_allowed,
                    shell_allowed = excluded.shell_allowed,
                    repo_write_allowed = excluded.repo_write_allowed,
                    require_confirmation = excluded.require_confirmation,
                    updated_at = excluded.updated_at
                """,
                (
                    int(workspace_id),
                    1 if browser_allowed else 0,
                    1 if desktop_allowed else 0,
                    1 if shell_allowed else 0,
                    1 if repo_write_allowed else 0,
                    1 if require_confirmation else 0,
                    now,
                ),
            )
            conn.commit()
        return self.get_workspace_policy(int(workspace_id))

    def save_browser_session(
        self,
        *,
        name: str,
        storage_state: Dict[str, Any],
        workspace_id: Optional[int] = None,
        notes: Optional[str] = None,
        provider: Optional[str] = None,
        template_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_name = str(name or "").strip()
        if not safe_name:
            raise ValueError("session name is required")
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO browser_sessions
                    (name, workspace_id, storage_state_json, provider, template_name, notes, health_status, health_details_json, last_health_check_at, created_at, updated_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    storage_state_json = excluded.storage_state_json,
                    provider = COALESCE(excluded.provider, browser_sessions.provider),
                    template_name = COALESCE(excluded.template_name, browser_sessions.template_name),
                    notes = excluded.notes,
                    health_status = COALESCE(browser_sessions.health_status, excluded.health_status),
                    health_details_json = COALESCE(browser_sessions.health_details_json, excluded.health_details_json),
                    last_health_check_at = COALESCE(browser_sessions.last_health_check_at, excluded.last_health_check_at),
                    updated_at = excluded.updated_at,
                    last_used_at = excluded.last_used_at
                """,
                (
                    safe_name,
                    int(workspace_id) if workspace_id else None,
                    json_dumps(storage_state or {}),
                    str(provider or "").strip() or None,
                    str(template_name or "").strip() or None,
                    str(notes or "").strip() or None,
                    "unknown",
                    json_dumps({}),
                    None,
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
        return self.get_browser_session_by_name(safe_name) or {}

    def get_browser_session_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, workspace_id, storage_state_json, provider, template_name, notes, health_status, health_details_json, last_health_check_at, created_at, updated_at, last_used_at
                FROM browser_sessions
                WHERE LOWER(name) = LOWER(?)
                """,
                (str(name or "").strip(),),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": int(row["id"]),
                "name": row["name"],
                "workspace_id": row["workspace_id"],
                "storage_state": json_loads(row["storage_state_json"], {}),
                "provider": row["provider"],
                "template_name": row["template_name"],
                "notes": row["notes"],
                "health_status": row["health_status"] or "unknown",
                "health_details": json_loads(row["health_details_json"], {}),
                "last_health_check_at": row["last_health_check_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_used_at": row["last_used_at"],
            }

    def get_browser_session(self, value: Any) -> Optional[Dict[str, Any]]:
        raw = str(value or "").strip()
        if not raw:
            return None
        with self._connect() as conn:
            row = None
            if raw.isdigit():
                row = conn.execute(
                    """
                    SELECT id, name, workspace_id, storage_state_json, provider, template_name, notes, health_status, health_details_json, last_health_check_at, created_at, updated_at, last_used_at
                    FROM browser_sessions
                    WHERE id = ?
                    """,
                    (int(raw),),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    """
                    SELECT id, name, workspace_id, storage_state_json, provider, template_name, notes, health_status, health_details_json, last_health_check_at, created_at, updated_at, last_used_at
                    FROM browser_sessions
                    WHERE LOWER(name) = LOWER(?)
                    """,
                    (raw,),
                ).fetchone()
            if row is None:
                return None
            return {
                "id": int(row["id"]),
                "name": row["name"],
                "workspace_id": row["workspace_id"],
                "storage_state": json_loads(row["storage_state_json"], {}),
                "provider": row["provider"],
                "template_name": row["template_name"],
                "notes": row["notes"],
                "health_status": row["health_status"] or "unknown",
                "health_details": json_loads(row["health_details_json"], {}),
                "last_health_check_at": row["last_health_check_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_used_at": row["last_used_at"],
            }

    def list_browser_sessions(self, *, limit: int = 50, workspace_id: Optional[int] = None) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        where = ""
        values: List[Any] = []
        if workspace_id:
            where = "WHERE workspace_id = ?"
            values.append(int(workspace_id))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, name, workspace_id, storage_state_json, provider, template_name, notes, health_status, health_details_json, last_health_check_at, created_at, updated_at, last_used_at
                FROM browser_sessions
                {where}
                ORDER BY COALESCE(last_used_at, updated_at) DESC, id DESC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "workspace_id": row["workspace_id"],
                    "provider": row["provider"],
                    "template_name": row["template_name"],
                    "notes": row["notes"],
                    "health_status": row["health_status"] or "unknown",
                    "health_details": json_loads(row["health_details_json"], {}),
                    "last_health_check_at": row["last_health_check_at"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "last_used_at": row["last_used_at"],
                }
                for row in rows
            ]

    def update_browser_session_health(
        self,
        value: Any,
        *,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        target = self.get_browser_session(value)
        if target is None:
            return None
        safe_status = str(status or "unknown").strip().lower()
        if safe_status not in {"healthy", "login_required", "expired", "warning", "error", "unknown"}:
            safe_status = "unknown"
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE browser_sessions
                SET health_status = ?, health_details_json = ?, last_health_check_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    safe_status,
                    json_dumps(details or {}),
                    now,
                    now,
                    int(target["id"]),
                ),
            )
            conn.commit()
        return self.get_browser_session(int(target["id"]))

    def touch_browser_session(self, name: str) -> bool:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE browser_sessions SET last_used_at = ?, updated_at = ? WHERE LOWER(name) = LOWER(?)",
                (now, now, str(name or "").strip()),
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def delete_browser_session(self, value: Any) -> bool:
        raw = str(value or "").strip()
        if not raw:
            return False
        with self._connect() as conn:
            if raw.isdigit():
                cur = conn.execute("DELETE FROM browser_sessions WHERE id = ?", (int(raw),))
            else:
                cur = conn.execute("DELETE FROM browser_sessions WHERE LOWER(name) = LOWER(?)", (raw,))
            conn.commit()
            return int(cur.rowcount) > 0

    def next_best_actions(self, *, workspace_id: Optional[int] = None, limit: int = 5) -> Dict[str, Any]:
        lim = max(1, min(int(limit), 12))
        resolved_workspace_id = int(workspace_id) if workspace_id else self.get_active_workspace_id()
        workspace = self.get_project_workspace(resolved_workspace_id) if resolved_workspace_id else None
        policy = self.get_workspace_policy(resolved_workspace_id)
        overview = self.memory_overview(limit_per_group=8)
        reminders = self.list_proactive_reminders(
            limit=10,
            status="open",
            workspace_id=resolved_workspace_id,
            due_within_hours=72,
        )
        trust = self.tool_reliability_report(limit=120)
        items: List[Dict[str, Any]] = []

        def _append(
            *,
            title: str,
            reason: str,
            priority: str,
            source: str,
            score: int,
            recommended_tool: Optional[str] = None,
            action: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            execution: Optional[Dict[str, Any]] = None,
            cta_label: Optional[str] = None,
        ) -> None:
            items.append(
                {
                    "title": title,
                    "reason": reason,
                    "priority": priority,
                    "source": source,
                    "score": int(score),
                    "recommended_tool": recommended_tool,
                    "action": action,
                    "metadata": metadata or {},
                    "execution": execution or {},
                    "cta_label": cta_label or "Run",
                }
            )

        if workspace is None:
            workspaces = self.list_project_workspaces(limit=6, include_archived=False)
            if workspaces:
                candidate = workspaces[0]
                _append(
                    title=f"Activate workspace {candidate.get('name')}",
                    reason="Jarvis can route memory, reminders, and permissions more accurately with an active workspace.",
                    priority="high",
                    source="workspace",
                    score=97,
                    action=f"switch workspace to {candidate.get('name')}",
                    metadata={"workspace_id": candidate.get("id")},
                    execution={"kind": "activate_workspace", "workspace_id": candidate.get("id")},
                    cta_label="Activate",
                )
        else:
            _append(
                title=f"Stay focused on {workspace.get('name')}",
                reason=(f"Current focus: {workspace.get('focus')}." if workspace.get("focus") else "Active workspace context is loaded and ready."),
                priority="normal",
                source="workspace",
                score=40,
                action=f"what is my active workspace",
                metadata={"workspace_id": workspace.get("id")},
                execution={"kind": "chat", "message": "what is my active workspace"},
                cta_label="Show",
            )

        priority_scores = {"critical": 100, "high": 88, "normal": 70, "low": 55}
        for reminder in reminders[:4]:
            priority = str(reminder.get("priority") or "normal").lower()
            due_at = _parse_iso(reminder.get("due_at"))
            due_label = "soon"
            if due_at is not None:
                delta = due_at - datetime.now(timezone.utc)
                hours_left = int(delta.total_seconds() // 3600)
                if hours_left <= 0:
                    due_label = "now"
                elif hours_left < 24:
                    due_label = f"in {hours_left}h"
                else:
                    due_label = f"in {max(1, hours_left // 24)}d"
            _append(
                title=str(reminder.get("title") or "Open reminder"),
                reason=f"{str(reminder.get('content') or '').strip()} Due {due_label}.",
                priority=priority,
                source="reminder",
                score=priority_scores.get(priority, 60),
                action=str(reminder.get("title") or "").strip(),
                metadata={"reminder_id": reminder.get("id"), "workspace_id": reminder.get("workspace_id")},
                execution={"kind": "reminder_done", "reminder_id": reminder.get("id")},
                cta_label="Mark done",
            )

        pinned = list(overview.get("pinned") or [])
        if workspace is not None and workspace.get("memories"):
            project_memories = [
                item
                for item in list(workspace.get("memories") or [])
                if str(item.get("memory_type") or "").strip().lower() == "project"
            ]
        else:
            project_memories = list(overview.get("projects") or [])
        for item in (pinned[:2] + project_memories[:2]):
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            importance = int(item.get("importance") or 3)
            _append(
                title=f"Revisit memory: {str(item.get('subject') or item.get('memory_type') or 'context')}",
                reason=content[:220],
                priority="high" if bool(item.get("pinned")) or importance >= 4 else "normal",
                source="memory",
                score=76 + min(importance, 5),
                action=content[:120],
                metadata={"memory_id": item.get("id"), "memory_type": item.get("memory_type")},
                execution={"kind": "chat", "message": f"Help me act on this memory: {content[:180]}"},
                cta_label="Use",
            )

        weak_tools = [
            tool
            for tool in list(trust.get("tools") or [])
            if int(tool.get("runs") or 0) >= 2 and float(tool.get("verification_rate_pct") or 0.0) < 75.0
        ]
        for tool in weak_tools[:2]:
            _append(
                title=f"Verify {tool.get('tool_name')} before relying on it",
                reason=(
                    f"{tool.get('tool_name')} is only verifying {tool.get('verification_rate_pct')}% of the time "
                    f"across {tool.get('runs')} recent runs."
                ),
                priority="high",
                source="trust",
                score=82,
                recommended_tool=str(tool.get("tool_name") or ""),
                action=f"Run a controlled test for {tool.get('tool_name')}",
                metadata=tool,
                execution={"kind": "chat", "message": f"Run a controlled verification for {tool.get('tool_name')}"},
                cta_label="Verify",
            )

        if policy.get("require_confirmation"):
            locked = [
                label
                for label, allowed in (
                    ("browser", policy.get("browser_allowed")),
                    ("desktop", policy.get("desktop_allowed")),
                    ("shell", policy.get("shell_allowed")),
                    ("repo write", policy.get("repo_write_allowed")),
                )
                if not bool(allowed)
            ]
            reason = "Workspace approvals are enabled."
            if locked:
                reason += " Disabled capabilities: " + ", ".join(locked[:4]) + "."
            _append(
                title="Review workspace approval policy",
                reason=reason,
                priority="normal",
                source="policy",
                score=65,
                action="Review workspace permissions",
                metadata={"workspace_id": resolved_workspace_id, "policy": policy},
                execution={"kind": "workspace_policy", "workspace_id": resolved_workspace_id},
                cta_label="Review",
            )

        deduped: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in sorted(items, key=lambda entry: (-int(entry.get("score") or 0), str(entry.get("title") or ""))):
            key = f"{str(item.get('source') or '')}:{str(item.get('title') or '').strip().lower()}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)

        chosen = deduped[:lim]
        summary = "No strong next action found yet."
        if chosen:
            summary = "Top priorities: " + "; ".join(str(item.get("title") or "").strip() for item in chosen[:3]) + "."
        return {
            "workspace_id": resolved_workspace_id,
            "workspace": workspace,
            "policy": policy,
            "summary": summary,
            "actions": chosen,
            "counts": {
                "open_reminders": len(reminders),
                "pinned_memories": len(pinned),
                "weak_tools": len(weak_tools),
            },
        }

    def workspace_memory_graph(self, *, workspace_id: Optional[int] = None, limit: int = 80) -> Dict[str, Any]:
        lim = max(2, min(int(limit), 200))
        workspace = self.get_project_workspace(int(workspace_id)) if workspace_id else None
        if workspace is not None:
            memory_items = list(workspace.get("memories") or [])[:lim]
        else:
            memory_items = self.list_long_term_memories(limit=lim, archived=False)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_edge_keys: set[str] = set()
        if workspace is not None:
            workspace_node_id = f"workspace:{workspace['id']}"
            nodes.append(
                {
                    "id": workspace_node_id,
                    "kind": "workspace",
                    "label": workspace.get("name"),
                    "status": workspace.get("status"),
                    "color": workspace.get("color") or "#2ee6d6",
                }
            )
        else:
            workspace_node_id = None
        for item in memory_items:
            node_id = f"memory:{int(item['id'])}"
            nodes.append(
                {
                    "id": node_id,
                    "kind": "memory",
                    "memory_id": int(item["id"]),
                    "label": item.get("subject") or item.get("memory_type") or f"memory-{item['id']}",
                    "memory_type": item.get("memory_type"),
                    "lane": item.get("lane"),
                    "importance": item.get("effective_importance"),
                    "pinned": bool(item.get("pinned")),
                }
            )
            if workspace_node_id:
                edges.append(
                    {
                        "source": workspace_node_id,
                        "target": node_id,
                        "type": "contains",
                        "weight": 1.0,
                    }
                )
        for idx, left in enumerate(memory_items):
            left_tags = {str(t).strip().lower() for t in list(left.get("tags") or []) if str(t).strip()}
            left_subject = str(left.get("subject") or "").strip().lower()
            for right in memory_items[idx + 1 :]:
                shared_tags = left_tags.intersection(
                    {str(t).strip().lower() for t in list(right.get("tags") or []) if str(t).strip()}
                )
                right_subject = str(right.get("subject") or "").strip().lower()
                relation = None
                weight = 0.0
                if left_subject and right_subject and left_subject == right_subject:
                    relation = "subject"
                    weight = 0.95
                elif shared_tags:
                    relation = "tag"
                    weight = min(0.9, 0.4 + 0.15 * len(shared_tags))
                elif left.get("memory_type") == "project" and right.get("memory_type") == "project":
                    relation = "project"
                    weight = 0.35
                if relation:
                    edge_key = f"{int(left['id'])}:{int(right['id'])}:{relation}"
                    if edge_key in seen_edge_keys:
                        continue
                    seen_edge_keys.add(edge_key)
                    edges.append(
                        {
                            "source": f"memory:{int(left['id'])}",
                            "target": f"memory:{int(right['id'])}",
                            "type": relation,
                            "weight": round(weight, 3),
                            "shared_tags": sorted(shared_tags)[:6],
                        }
                    )
        return {
            "workspace_id": int(workspace["id"]) if workspace else None,
            "workspace": workspace,
            "nodes": nodes,
            "edges": edges,
            "counts": {"nodes": len(nodes), "edges": len(edges)},
        }

    def create_proactive_reminder(
        self,
        *,
        title: str,
        content: str,
        due_at: Optional[str] = None,
        priority: str = "normal",
        channel: Optional[str] = None,
        workspace_id: Optional[int] = None,
        memory_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        safe_title = str(title or "").strip()
        safe_content = str(content or "").strip()
        if not safe_title or not safe_content:
            raise ValueError("reminder title and content are required")
        safe_priority = str(priority or "normal").strip().lower()
        if safe_priority not in {"low", "normal", "high", "critical"}:
            safe_priority = "normal"
        now = _now_iso()
        if due_at is None:
            due_at = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO proactive_reminders
                    (title, content, status, priority, due_at, channel, workspace_id, memory_id, created_at, updated_at, delivered_at, discord_delivered_at, voice_announced_at)
                VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    safe_title,
                    safe_content,
                    safe_priority,
                    due_at,
                    str(channel or "").strip() or None,
                    int(workspace_id) if workspace_id else None,
                    int(memory_id) if memory_id else None,
                    now,
                    now,
                ),
            )
            conn.commit()
            reminder_id = int(cur.lastrowid)
        return self.get_proactive_reminder(reminder_id) or {}

    def get_proactive_reminder(self, reminder_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT pr.id, pr.title, pr.content, pr.status, pr.priority, pr.due_at, pr.channel,
                       pr.workspace_id, pw.name AS workspace_name, pr.memory_id, pr.created_at,
                       pr.updated_at, pr.delivered_at, pr.discord_delivered_at, pr.voice_announced_at
                FROM proactive_reminders pr
                LEFT JOIN project_workspaces pw ON pw.id = pr.workspace_id
                WHERE pr.id = ?
                """,
                (int(reminder_id),),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": int(row["id"]),
                "title": row["title"],
                "content": row["content"],
                "status": row["status"],
                "priority": row["priority"],
                "due_at": row["due_at"],
                "channel": row["channel"],
                "workspace_id": row["workspace_id"],
                "workspace_name": row["workspace_name"],
                "memory_id": row["memory_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "delivered_at": row["delivered_at"],
                "discord_delivered_at": row["discord_delivered_at"],
                "voice_announced_at": row["voice_announced_at"],
            }

    def list_proactive_reminders(
        self,
        *,
        limit: int = 100,
        status: Optional[str] = None,
        workspace_id: Optional[int] = None,
        due_within_hours: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 300))
        where: List[str] = []
        values: List[Any] = []
        if status:
            where.append("pr.status = ?")
            values.append(str(status).strip().lower())
        if workspace_id:
            where.append("pr.workspace_id = ?")
            values.append(int(workspace_id))
        if due_within_hours is not None:
            cutoff = (datetime.now(timezone.utc) + timedelta(hours=max(1, int(due_within_hours)))).isoformat()
            where.append("(pr.due_at IS NOT NULL AND pr.due_at <= ?)")
            values.append(cutoff)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT pr.id, pr.title, pr.content, pr.status, pr.priority, pr.due_at, pr.channel,
                       pr.workspace_id, pw.name AS workspace_name, pr.memory_id, pr.created_at,
                       pr.updated_at, pr.delivered_at, pr.discord_delivered_at, pr.voice_announced_at
                FROM proactive_reminders pr
                LEFT JOIN project_workspaces pw ON pw.id = pr.workspace_id
                {clause}
                ORDER BY CASE pr.priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                         COALESCE(pr.due_at, pr.created_at) ASC, pr.id DESC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "content": row["content"],
                    "status": row["status"],
                    "priority": row["priority"],
                    "due_at": row["due_at"],
                    "channel": row["channel"],
                    "workspace_id": row["workspace_id"],
                    "workspace_name": row["workspace_name"],
                    "memory_id": row["memory_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "delivered_at": row["delivered_at"],
                    "discord_delivered_at": row["discord_delivered_at"],
                    "voice_announced_at": row["voice_announced_at"],
                }
                for row in rows
            ]

    def update_proactive_reminder(
        self,
        reminder_id: int,
        *,
        status: Optional[str] = None,
        due_at: Optional[str] = None,
        delivered: Optional[bool] = None,
        discord_delivered: Optional[bool] = None,
        voice_announced: Optional[bool] = None,
    ) -> bool:
        updates: List[str] = []
        values: List[Any] = []
        if status is not None:
            safe_status = str(status).strip().lower()
            if safe_status not in {"open", "done", "dismissed"}:
                safe_status = "open"
            updates.append("status = ?")
            values.append(safe_status)
        if due_at is not None:
            updates.append("due_at = ?")
            values.append(due_at)
        if delivered is not None:
            updates.append("delivered_at = ?")
            values.append(_now_iso() if delivered else None)
        if discord_delivered is not None:
            updates.append("discord_delivered_at = ?")
            values.append(_now_iso() if discord_delivered else None)
        if voice_announced is not None:
            updates.append("voice_announced_at = ?")
            values.append(_now_iso() if voice_announced else None)
        if not updates:
            return False
        updates.append("updated_at = ?")
        values.append(_now_iso())
        values.append(int(reminder_id))
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE proactive_reminders SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()
            return int(cur.rowcount) > 0

    def list_due_proactive_reminders(
        self,
        *,
        limit: int = 20,
        for_discord: bool = False,
        for_voice: bool = False,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 100))
        now = _now_iso()
        where = ["pr.status = 'open'", "pr.due_at IS NOT NULL", "pr.due_at <= ?"]
        values: List[Any] = [now]
        if for_discord:
            where.append("pr.discord_delivered_at IS NULL")
        if for_voice:
            where.append("pr.voice_announced_at IS NULL")
        clause = " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT pr.id, pr.title, pr.content, pr.status, pr.priority, pr.due_at, pr.channel,
                       pr.workspace_id, pw.name AS workspace_name, pr.memory_id, pr.created_at,
                       pr.updated_at, pr.delivered_at, pr.discord_delivered_at, pr.voice_announced_at
                FROM proactive_reminders pr
                LEFT JOIN project_workspaces pw ON pw.id = pr.workspace_id
                WHERE {clause}
                ORDER BY CASE pr.priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
                         pr.due_at ASC, pr.id ASC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "content": row["content"],
                    "status": row["status"],
                    "priority": row["priority"],
                    "due_at": row["due_at"],
                    "channel": row["channel"],
                    "workspace_id": row["workspace_id"],
                    "workspace_name": row["workspace_name"],
                    "memory_id": row["memory_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "delivered_at": row["delivered_at"],
                    "discord_delivered_at": row["discord_delivered_at"],
                    "voice_announced_at": row["voice_announced_at"],
                }
                for row in rows
            ]

    def generate_proactive_reminders(
        self,
        *,
        workspace_id: Optional[int] = None,
        limit: int = 4,
    ) -> Dict[str, Any]:
        lim = max(1, min(int(limit), 10))
        workspace = self.get_project_workspace(int(workspace_id)) if workspace_id else None
        active_workspace_id = workspace_id or self.get_active_workspace_id()
        candidates = (workspace.get("memories") if workspace else None) or self.list_long_term_memories(limit=20)
        created: List[Dict[str, Any]] = []
        open_existing = self.list_proactive_reminders(limit=200, status="open", workspace_id=active_workspace_id)
        existing_memory_ids = {int(item["memory_id"]) for item in open_existing if item.get("memory_id")}
        for item in candidates:
            memory_id = int(item["id"])
            if memory_id in existing_memory_ids:
                continue
            title = f"Follow up: {item.get('subject') or item.get('memory_type') or 'memory'}"
            content = str(item.get("content") or "").strip()
            due_at = (datetime.now(timezone.utc) + timedelta(hours=2 if bool(item.get("pinned")) else 8)).isoformat()
            priority = "high" if bool(item.get("pinned")) else ("normal" if item.get("memory_type") == "project" else "low")
            created.append(
                self.create_proactive_reminder(
                    title=title[:120],
                    content=content[:500],
                    due_at=due_at,
                    priority=priority,
                    channel="dashboard",
                    workspace_id=active_workspace_id,
                    memory_id=memory_id,
                )
            )
            if len(created) >= lim:
                break
        return {
            "ok": True,
            "workspace_id": active_workspace_id,
            "created": created,
            "open": self.list_proactive_reminders(limit=20, status="open", workspace_id=active_workspace_id),
        }

    def list_long_term_memories(
        self,
        *,
        limit: int = 100,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        pinned: Optional[bool] = None,
        since_hours: Optional[int] = None,
        archived: bool = False,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        lane: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        where: List[str] = []
        values: List[Any] = []
        if memory_type:
            where.append("memory_type = ?")
            values.append(str(memory_type).strip().lower())
        if subject:
            where.append("subject = ?")
            values.append(str(subject).strip())
        if pinned is not None:
            where.append("pinned = ?")
            values.append(1 if bool(pinned) else 0)
        if lane:
            where.append("lane = ?")
            values.append(str(lane).strip().lower())
        where.append("archived = ?")
        values.append(1 if bool(archived) else 0)
        if query:
            q = f"%{str(query).strip().lower()}%"
            where.append("(LOWER(content) LIKE ? OR LOWER(COALESCE(subject, '')) LIKE ? OR LOWER(tags_json) LIKE ?)")
            values.extend([q, q, q])
        if tag:
            t = str(tag).strip().lower()
            if t:
                where.append("LOWER(tags_json) LIKE ?")
                values.append(f'%"{t}"%')
        if since_hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))).isoformat()
            where.append("(created_at >= ? OR last_accessed_at >= ?)")
            values.extend([cutoff, cutoff])
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, session_id, content, tags_json, importance, memory_type, subject, access_count, pinned, archived, pin_order, source, created_at, last_accessed_at
                       , lane, project_order
                FROM long_term_memories
                {clause}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            items = [self._memory_with_scores(r) for r in rows]
            return self._sort_memory_items(items)

    def search_long_term_memories(self, *, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        lim = max(1, min(int(limit), 50))
        memories = self.list_long_term_memories(limit=400)
        if not q:
            return memories[:lim]

        words = {w for w in q.replace("\n", " ").split(" ") if w}
        scored: List[tuple[float, Dict[str, Any]]] = []
        for item in memories:
            hay = (
                str(item.get("content") or "")
                + " "
                + " ".join(item.get("tags") or [])
                + " "
                + str(item.get("memory_type") or "")
                + " "
                + str(item.get("subject") or "")
            ).lower()
            overlap = sum(1 for w in words if w in hay)
            if overlap <= 0:
                continue
            score = (overlap * 10.0) + float(item.get("effective_importance") or item.get("importance") or 1)
            if q in hay:
                score += 3.0
            if words and all(w in hay for w in words):
                score += 5.0
            scored.append((score, item))
        scored.sort(key=lambda x: (x[0], int(x[1].get("id") or 0)), reverse=True)
        results = [item for _, item in scored[:lim]]
        if results:
            now = _now_iso()
            with self._connect() as conn:
                for item in results:
                    conn.execute(
                        "UPDATE long_term_memories SET last_accessed_at = ?, access_count = COALESCE(access_count, 0) + 1 WHERE id = ?",
                        (now, int(item["id"])),
                    )
                conn.commit()
        return results

    def update_long_term_memory(
        self,
        memory_id: int,
        *,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[int] = None,
        memory_type: Optional[str] = None,
        subject: Optional[str] = None,
        pinned: Optional[bool] = None,
        archived: Optional[bool] = None,
        lane: Optional[str] = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, pinned, pin_order, memory_type, project_order, lane FROM long_term_memories WHERE id = ?",
                (int(memory_id),),
            ).fetchone()
            if row is None:
                return False
            current_pinned = bool(int(row["pinned"] or 0)) if "pinned" in row.keys() else False
            current_pin_order = int(row["pin_order"] or 0) if "pin_order" in row.keys() else 0
            current_memory_type = str(row["memory_type"] or "general").strip().lower()
            current_project_order = int(row["project_order"] or 0) if "project_order" in row.keys() else 0
            updates: List[str] = []
            values: List[Any] = []
            now = _now_iso()
            if content is not None:
                updates.append("content = ?")
                values.append(str(content).strip())
            if tags is not None:
                safe_tags = [str(t).strip() for t in tags if str(t).strip()][:12]
                updates.append("tags_json = ?")
                values.append(json_dumps(safe_tags))
            if importance is not None:
                updates.append("importance = ?")
                values.append(max(1, min(int(importance), 5)))
            if memory_type is not None:
                updates.append("memory_type = ?")
                values.append(str(memory_type).strip().lower() or "general")
                if str(memory_type).strip().lower() == "project" and current_project_order <= 0:
                    updates.append("project_order = ?")
                    values.append(self._next_project_order(conn))
            if subject is not None:
                updates.append("subject = ?")
                values.append(str(subject).strip() or None)
            if pinned is not None:
                updates.append("pinned = ?")
                values.append(1 if bool(pinned) else 0)
                updates.append("pin_order = ?")
                if bool(pinned):
                    values.append(current_pin_order if current_pinned and current_pin_order > 0 else self._next_pin_order(conn))
                else:
                    values.append(0)
            if lane is not None:
                lane_value = str(lane).strip().lower()
                if lane_value not in {"critical", "project", "personal"}:
                    lane_value = "personal"
                updates.append("lane = ?")
                values.append(lane_value)
            if archived is not None:
                updates.append("archived = ?")
                values.append(1 if bool(archived) else 0)
            if current_memory_type == "project" and current_project_order <= 0 and memory_type is None:
                updates.append("project_order = ?")
                values.append(self._next_project_order(conn))
            updates.append("last_accessed_at = ?")
            values.append(now)
            values.append(int(memory_id))
            conn.execute(f"UPDATE long_term_memories SET {', '.join(updates)} WHERE id = ?", values)
            conn.commit()
            return True

    def archive_long_term_memory(self, memory_id: int, *, archived: bool = True) -> bool:
        return self.update_long_term_memory(int(memory_id), archived=archived)

    def bulk_update_long_term_memories(self, ids: List[int], *, action: str) -> Dict[str, Any]:
        safe_ids = [int(i) for i in ids if int(i) > 0][:200]
        action_name = str(action or "").strip().lower()
        updated = 0
        for memory_id in safe_ids:
            ok = False
            if action_name == "pin":
                ok = self.update_long_term_memory(memory_id, pinned=True)
            elif action_name == "unpin":
                ok = self.update_long_term_memory(memory_id, pinned=False)
            elif action_name == "archive":
                ok = self.archive_long_term_memory(memory_id, archived=True)
            elif action_name == "restore":
                ok = self.archive_long_term_memory(memory_id, archived=False)
            if ok:
                updated += 1
        return {"ok": True, "action": action_name, "updated": updated, "ids": safe_ids}

    def reorder_pinned_memories(self, ordered_ids: List[int]) -> Dict[str, Any]:
        safe_ids = [int(i) for i in ordered_ids if int(i) > 0][:200]
        with self._connect() as conn:
            existing_rows = conn.execute(
                """
                SELECT id, pin_order
                FROM long_term_memories
                WHERE pinned = 1 AND archived = 0
                ORDER BY CASE WHEN pin_order > 0 THEN pin_order ELSE 999999 END ASC, id DESC
                """
            ).fetchall()
            existing_ids = [int(row["id"]) for row in existing_rows]
            ordered = [mid for mid in safe_ids if mid in existing_ids]
            for mid in existing_ids:
                if mid not in ordered:
                    ordered.append(mid)
            for idx, memory_id in enumerate(ordered, start=1):
                conn.execute(
                    "UPDATE long_term_memories SET pin_order = ?, last_accessed_at = ? WHERE id = ?",
                    (idx, _now_iso(), memory_id),
                )
            conn.commit()
        return {"ok": True, "ordered_ids": ordered}

    def reorder_project_memories(self, ordered_ids: List[int]) -> Dict[str, Any]:
        safe_ids = [int(i) for i in ordered_ids if int(i) > 0][:200]
        with self._connect() as conn:
            existing_rows = conn.execute(
                """
                SELECT id, project_order
                FROM long_term_memories
                WHERE memory_type = 'project' AND archived = 0
                ORDER BY CASE WHEN project_order > 0 THEN project_order ELSE 999999 END ASC, id DESC
                """
            ).fetchall()
            existing_ids = [int(row["id"]) for row in existing_rows]
            ordered = [mid for mid in safe_ids if mid in existing_ids]
            for mid in existing_ids:
                if mid not in ordered:
                    ordered.append(mid)
            for idx, memory_id in enumerate(ordered, start=1):
                conn.execute(
                    "UPDATE long_term_memories SET project_order = ?, last_accessed_at = ? WHERE id = ?",
                    (idx, _now_iso(), memory_id),
                    )
            conn.commit()
        return {"ok": True, "ordered_ids": ordered}

    def add_vision_observation(
        self,
        *,
        summary: str,
        details: Dict[str, Any],
        source: str = "upload",
        session_id: Optional[str] = None,
    ) -> int:
        now = _now_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO vision_observations
                    (session_id, source, summary, details_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, source, summary, json_dumps(details), now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_vision_observations(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, source, summary, details_json, created_at
                FROM vision_observations
                ORDER BY id DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "session_id": r["session_id"],
                    "source": r["source"],
                    "summary": r["summary"],
                    "details": json_loads(r["details_json"], {}),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def create_autonomous_job(
        self,
        *,
        name: str,
        goal: str,
        mode: str = "multi_agent",
        interval_minutes: int,
        session_id: Optional[str] = None,
        auto_approve: bool = False,
        enabled: bool = True,
        schedule_type: str = "interval",
        run_hour: Optional[int] = None,
        run_minute: Optional[int] = None,
        timezone_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        interval = max(1, min(int(interval_minutes), 10080))
        now = _now_iso()
        next_run = self._next_autonomous_job_run(
            now_iso=now,
            interval_minutes=interval,
            schedule_type=schedule_type,
            run_hour=run_hour,
            run_minute=run_minute,
            timezone_name=timezone_name,
        )
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO autonomous_jobs
                    (name, goal, mode, session_id, interval_minutes, auto_approve, enabled, created_at, updated_at, last_run_at, next_run_at, last_status, last_error, last_result_json, schedule_type, run_hour, run_minute, timezone, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    name.strip(),
                    goal.strip(),
                    mode.strip() or "multi_agent",
                    session_id,
                    interval,
                    1 if auto_approve else 0,
                    1 if enabled else 0,
                    now,
                    now,
                    next_run,
                    str(schedule_type or "interval").strip().lower() or "interval",
                    max(0, min(23, int(run_hour))) if run_hour is not None else None,
                    max(0, min(59, int(run_minute))) if run_minute is not None else None,
                    str(timezone_name or "").strip() or None,
                    json_dumps(metadata or {}),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_autonomous_jobs(self, *, limit: int = 100, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        where = ""
        values: List[Any] = []
        if mode:
            where = "WHERE mode = ?"
            values.append(str(mode).strip())
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, name, goal, mode, session_id, interval_minutes, auto_approve, enabled, created_at, updated_at, last_run_at, next_run_at, last_status, last_error, last_result_json, schedule_type, run_hour, run_minute, timezone, metadata_json
                FROM autonomous_jobs
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(values + [lim]),
            ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "name": r["name"],
                    "goal": r["goal"],
                    "mode": r["mode"],
                    "session_id": r["session_id"],
                    "interval_minutes": int(r["interval_minutes"]),
                    "auto_approve": bool(r["auto_approve"]),
                    "enabled": bool(r["enabled"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "last_run_at": r["last_run_at"],
                    "next_run_at": r["next_run_at"],
                    "last_status": r["last_status"],
                    "last_error": r["last_error"],
                    "last_result": json_loads(r["last_result_json"], None),
                    "schedule_type": r["schedule_type"] or "interval",
                    "run_hour": r["run_hour"],
                    "run_minute": r["run_minute"],
                    "timezone": r["timezone"],
                    "metadata": json_loads(r["metadata_json"], {}),
                }
                for r in rows
            ]

    def update_autonomous_job(
        self,
        job_id: int,
        *,
        enabled: Optional[bool] = None,
        interval_minutes: Optional[int] = None,
        auto_approve: Optional[bool] = None,
        goal: Optional[str] = None,
        name: Optional[str] = None,
        mode: Optional[str] = None,
        schedule_type: Optional[str] = None,
        run_hour: Optional[int] = None,
        run_minute: Optional[int] = None,
        timezone_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, interval_minutes, schedule_type, run_hour, run_minute, timezone, metadata_json FROM autonomous_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            updates: List[str] = []
            values: List[Any] = []
            now = _now_iso()
            next_schedule_type = str(schedule_type or row["schedule_type"] or "interval").strip().lower() or "interval"
            next_interval = int(interval_minutes if interval_minutes is not None else row["interval_minutes"])
            next_run_hour = int(run_hour if run_hour is not None else (row["run_hour"] if row["run_hour"] is not None else 8))
            next_run_minute = int(run_minute if run_minute is not None else (row["run_minute"] if row["run_minute"] is not None else 0))
            next_timezone = str(timezone_name if timezone_name is not None else (row["timezone"] or "")).strip() or None
            if enabled is not None:
                updates.append("enabled = ?")
                values.append(1 if enabled else 0)
            if interval_minutes is not None:
                interval = max(1, min(int(interval_minutes), 10080))
                updates.append("interval_minutes = ?")
                values.append(interval)
            if auto_approve is not None:
                updates.append("auto_approve = ?")
                values.append(1 if auto_approve else 0)
            if goal is not None:
                updates.append("goal = ?")
                values.append(goal)
            if name is not None:
                updates.append("name = ?")
                values.append(name)
            if mode is not None:
                updates.append("mode = ?")
                values.append(mode)
            if schedule_type is not None:
                updates.append("schedule_type = ?")
                values.append(next_schedule_type)
            if run_hour is not None:
                updates.append("run_hour = ?")
                values.append(max(0, min(23, int(run_hour))))
            if run_minute is not None:
                updates.append("run_minute = ?")
                values.append(max(0, min(59, int(run_minute))))
            if timezone_name is not None:
                updates.append("timezone = ?")
                values.append(next_timezone)
            if metadata is not None:
                updates.append("metadata_json = ?")
                values.append(json_dumps(metadata))
            if (
                interval_minutes is not None
                or schedule_type is not None
                or run_hour is not None
                or run_minute is not None
                or timezone_name is not None
            ):
                updates.append("next_run_at = ?")
                values.append(
                    self._next_autonomous_job_run(
                        now_iso=now,
                        interval_minutes=next_interval,
                        schedule_type=next_schedule_type,
                        run_hour=next_run_hour,
                        run_minute=next_run_minute,
                        timezone_name=next_timezone,
                    )
                )
            updates.append("updated_at = ?")
            values.append(now)
            values.append(job_id)
            conn.execute(
                f"UPDATE autonomous_jobs SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            conn.commit()
            return True

    def claim_due_autonomous_jobs(self, *, now_iso: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        now = now_iso or _now_iso()
        lim = max(1, min(int(limit), 100))
        claimed: List[Dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, goal, mode, session_id, interval_minutes, auto_approve, schedule_type, run_hour, run_minute, timezone, metadata_json
                FROM autonomous_jobs
                WHERE enabled = 1 AND next_run_at <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (now, lim),
            ).fetchall()
            for r in rows:
                next_run = self._next_autonomous_job_run(
                    now_iso=now,
                    interval_minutes=int(r["interval_minutes"]),
                    schedule_type=str(r["schedule_type"] or "interval"),
                    run_hour=r["run_hour"],
                    run_minute=r["run_minute"],
                    timezone_name=r["timezone"],
                )
                conn.execute(
                    """
                    UPDATE autonomous_jobs
                    SET last_run_at = ?, next_run_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, next_run, now, int(r["id"])),
                )
                claimed.append(
                    {
                        "id": int(r["id"]),
                        "name": r["name"],
                        "goal": r["goal"],
                        "mode": r["mode"],
                        "session_id": r["session_id"],
                        "interval_minutes": int(r["interval_minutes"]),
                        "auto_approve": bool(r["auto_approve"]),
                        "schedule_type": r["schedule_type"] or "interval",
                        "run_hour": r["run_hour"],
                        "run_minute": r["run_minute"],
                        "timezone": r["timezone"],
                        "metadata": json_loads(r["metadata_json"], {}),
                    }
                )
            conn.commit()
        return claimed

    def ensure_daily_briefing_automations(
        self,
        *,
        session_id: Optional[str] = None,
        timezone_name: str = "America/Chicago",
        morning_hour: int = 8,
        evening_hour: int = 18,
    ) -> List[Dict[str, Any]]:
        targets = [
            {
                "name": "Jarvis Morning Briefing",
                "goal": "Generate the morning Jarvis briefing from pinned and recent project memory.",
                "period": "morning",
                "hour": max(0, min(23, int(morning_hour))),
                "recent_project_hours": 72,
            },
            {
                "name": "Jarvis Evening Briefing",
                "goal": "Generate the evening Jarvis briefing from pinned and recent project memory.",
                "period": "evening",
                "hour": max(0, min(23, int(evening_hour))),
                "recent_project_hours": 24,
            },
        ]
        existing = {str(job.get("name") or ""): job for job in self.list_autonomous_jobs(limit=200, mode="briefing")}
        job_ids: List[int] = []
        for target in targets:
            metadata = {
                "period": target["period"],
                "recent_project_hours": target["recent_project_hours"],
                "kind": "memory_briefing",
            }
            current = existing.get(target["name"])
            if current:
                self.update_autonomous_job(
                    int(current["id"]),
                    name=target["name"],
                    goal=target["goal"],
                    mode="briefing",
                    enabled=True,
                    interval_minutes=1440,
                    schedule_type="daily",
                    run_hour=target["hour"],
                    run_minute=0,
                    timezone_name=timezone_name,
                    metadata=metadata,
                )
                job_ids.append(int(current["id"]))
            else:
                job_ids.append(
                    self.create_autonomous_job(
                        name=target["name"],
                        goal=target["goal"],
                        mode="briefing",
                        interval_minutes=1440,
                        session_id=session_id,
                        auto_approve=True,
                        enabled=True,
                        schedule_type="daily",
                        run_hour=target["hour"],
                        run_minute=0,
                        timezone_name=timezone_name,
                        metadata=metadata,
                    )
                )
        return [job for job in self.list_autonomous_jobs(limit=200, mode="briefing") if int(job["id"]) in job_ids]

    def mark_autonomous_job_result(self, job_id: int, *, ok: bool, error: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> bool:
        now = _now_iso()
        status = "ok" if ok else "failed"
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE autonomous_jobs
                SET last_status = ?, last_error = ?, last_result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, json_dumps(result) if result is not None else None, now, job_id),
            )
            conn.commit()
            return int(cur.rowcount) > 0


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True)


def json_loads(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    import json

    try:
        return json.loads(raw)
    except Exception:
        return default


def _iso_add_minutes(iso_value: str, minutes: int) -> str:
    dt = datetime.fromisoformat(iso_value)
    return (dt + timedelta(minutes=minutes)).isoformat()
