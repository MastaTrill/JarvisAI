"""
Jarvis "brain" API.

This is intentionally minimal: it provides a single chat endpoint plus a small
tool registry so the UI can drive the system without requiring an external LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
import shlex
import subprocess
import shutil
import sys
import re
import math
import time
import asyncio
import ast
import tempfile
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from uuid import uuid4
import zipfile
from urllib import request as urlrequest
from urllib.parse import quote_plus
import smtplib
from email.mime.text import MIMEText

import os
import json
import base64

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageStat
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency at runtime
    np = None
try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency at runtime
    WhisperModel = None
try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency at runtime
    pytesseract = None
try:
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenWakeWordModel = None
try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional dependency at runtime
    sync_playwright = None
    PlaywrightTimeoutError = Exception

from agent_memory import AgentMemory, StoredMessage
from agent_task_memory import AgentTaskMemory
from llm_openai import (
    chat_with_tools,
    is_openai_configured,
    _extract_first_function_call,
    _extract_output_text,
    vision_analyze as openai_vision_analyze,
)
from llm_ollama import (
    is_ollama_configured,
    ollama_chat,
    ollama_chat_stream,
    ollama_chat_with_images,
    ollama_list_models,
    ollama_model,
    parse_tool_directive,
    strip_final_answer,
)


router = APIRouter(prefix="/agent", tags=["Agent"])
_memory = AgentMemory()
_task_memory = AgentTaskMemory()


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None


class AgentLlmConfigRequest(BaseModel):
    mode: str = Field(default="quality", pattern="^(fast|quality)$")
    model: Optional[str] = Field(default=None, max_length=200)


class AgentVisionConfigRequest(BaseModel):
    provider: str = Field(default="auto", pattern="^(auto|heuristic|ollama|openai)$")
    model: Optional[str] = Field(default=None, max_length=200)


class AgentPolicyConfigRequest(BaseModel):
    strict_confirm: bool = True


class MultiAgentRunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    fast_synthesis: bool = True


class LongTermMemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    tags: List[str] = Field(default_factory=list, max_length=12)
    importance: int = Field(default=3, ge=1, le=5)
    memory_type: str = Field(default="general", max_length=40)
    subject: Optional[str] = Field(default=None, max_length=120)
    pinned: bool = False
    lane: Optional[str] = Field(default=None, pattern="^(critical|project|personal)?$")
    source: str = Field(default="user", max_length=100)
    session_id: Optional[str] = None


class LongTermMemoryUpdateRequest(BaseModel):
    content: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    tags: Optional[List[str]] = Field(default=None, max_length=12)
    importance: Optional[int] = Field(default=None, ge=1, le=5)
    memory_type: Optional[str] = Field(default=None, max_length=40)
    subject: Optional[str] = Field(default=None, max_length=120)
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    lane: Optional[str] = Field(default=None, pattern="^(critical|project|personal)$")


class LongTermMemoryBulkActionRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=200)
    action: str = Field(..., pattern="^(pin|unpin|archive|restore)$")


class LongTermMemoryPinOrderRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=200)


class LongTermMemoryProjectOrderRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=200)


class MemorySavedFilterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    query: str = Field(default="", max_length=160)
    tag: str = Field(default="", max_length=60)
    memory_type: str = Field(default="", max_length=40)


class MemoryBriefingDeliveryConfigRequest(BaseModel):
    enabled: bool = True
    discord_enabled: bool = False
    discord_webhook_url: str = Field(default="", max_length=2000)
    email_enabled: bool = False
    email_to: str = Field(default="", max_length=320)
    mobile_enabled: bool = False
    mobile_push_url: str = Field(default="", max_length=2000)
    mobile_channel: str = Field(default="ntfy", pattern="^(ntfy|generic)$")


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    focus: str = Field(default="", max_length=240)
    color: str = Field(default="#2ee6d6", max_length=32)
    status: str = Field(default="active", pattern="^(active|paused|archived)$")
    memory_ids: List[int] = Field(default_factory=list, max_length=200)


class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    focus: Optional[str] = Field(default=None, max_length=240)
    color: Optional[str] = Field(default=None, max_length=32)
    status: Optional[str] = Field(default=None, pattern="^(active|paused|archived)$")
    memory_ids: Optional[List[int]] = Field(default=None, max_length=200)


class WorkspaceActivationRequest(BaseModel):
    workspace_id: Optional[int] = Field(default=None, ge=1)


class ReminderCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    content: str = Field(..., min_length=1, max_length=1000)
    due_at: Optional[str] = Field(default=None, max_length=80)
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")
    channel: Optional[str] = Field(default="dashboard", max_length=40)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    memory_id: Optional[int] = Field(default=None, ge=1)


class ReminderUpdateRequest(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(open|done|dismissed)$")
    due_at: Optional[str] = Field(default=None, max_length=80)
    delivered: Optional[bool] = None


class AgentControlConfigRequest(BaseModel):
    browser_enabled: bool = True
    desktop_enabled: bool = True
    execute_on_host: bool = False
    browser_name: str = Field(default="default", max_length=40)
    search_engine: str = Field(default="google", pattern="^(google|duckduckgo|bing)$")


class BrowserOpenRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    confirm: bool = False


class BrowserSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    confirm: bool = False


class DesktopLaunchRequest(BaseModel):
    app: str = Field(..., min_length=1, max_length=120)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    confirm: bool = False


class DesktopControlRequest(BaseModel):
    action: str = Field(..., pattern="^(launch|open_url|type_text|hotkey)$")
    target: Optional[str] = Field(default=None, max_length=400)
    text: Optional[str] = Field(default=None, max_length=4000)
    keys: List[str] = Field(default_factory=list, max_length=8)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    confirm: bool = False


class BrowserWorkflowStep(BaseModel):
    action: str = Field(..., pattern="^(goto|click|fill|press|wait_for|extract_text|extract_html|screenshot)$")
    selector: Optional[str] = Field(default=None, max_length=500)
    value: Optional[str] = Field(default=None, max_length=4000)
    timeout_ms: Optional[int] = Field(default=None, ge=100, le=120000)


class BrowserWorkflowRequest(BaseModel):
    start_url: Optional[str] = Field(default=None, max_length=2000)
    steps: List[BrowserWorkflowStep] = Field(default_factory=list, max_length=40)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    headless: bool = True
    session_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    save_session: bool = False
    session_notes: Optional[str] = Field(default=None, max_length=240)
    template_name: Optional[str] = Field(default=None, max_length=80)
    confirm: bool = False


class BrowserSessionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    notes: Optional[str] = Field(default=None, max_length=240)
    provider: Optional[str] = Field(default=None, max_length=80)
    template_name: Optional[str] = Field(default=None, max_length=80)
    storage_state: Dict[str, Any] = Field(default_factory=dict)


class WorkspacePolicyRequest(BaseModel):
    browser_allowed: bool = True
    desktop_allowed: bool = True
    shell_allowed: bool = False
    repo_write_allowed: bool = False
    require_confirmation: bool = True


class BrowserWorkflowTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    start_url: Optional[str] = Field(default=None, max_length=2000)
    steps: List[BrowserWorkflowStep] = Field(..., min_length=1, max_length=40)
    category: str = Field(default="custom", max_length=40)
    auth_template: bool = False
    recommended_session_name: Optional[str] = Field(default=None, max_length=80)
    provider: Optional[str] = Field(default=None, max_length=80)
    healthcheck_url: Optional[str] = Field(default=None, max_length=2000)
    healthcheck_selector: Optional[str] = Field(default=None, max_length=240)
    logged_out_markers: List[str] = Field(default_factory=list, max_length=12)
    healthy_markers: List[str] = Field(default_factory=list, max_length=12)


class NextActionExecuteRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None
    approve: bool = False


class AutonomousMissionRequest(BaseModel):
    workspace_id: Optional[int] = Field(default=None, ge=1)
    session_id: Optional[str] = None
    limit: int = Field(default=3, ge=1, le=8)
    auto_approve: bool = False
    retry_limit: int = Field(default=1, ge=0, le=5)


class AutonomousJobCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    goal: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field(default="multi_agent", pattern="^(goal|multi_agent|briefing|watcher)$")
    interval_minutes: int = Field(..., ge=1, le=10080)
    session_id: Optional[str] = None
    auto_approve: bool = False
    enabled: bool = True
    schedule_type: str = Field(default="interval", pattern="^(interval|daily)$")
    run_hour: Optional[int] = Field(default=None, ge=0, le=23)
    run_minute: Optional[int] = Field(default=None, ge=0, le=59)
    timezone_name: Optional[str] = Field(default=None, max_length=120)
    metadata: Optional[Dict[str, Any]] = None


class AutonomousJobUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    goal: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    mode: Optional[str] = Field(default=None, pattern="^(goal|multi_agent|briefing|watcher)$")
    interval_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    auto_approve: Optional[bool] = None
    enabled: Optional[bool] = None
    schedule_type: Optional[str] = Field(default=None, pattern="^(interval|daily)$")
    run_hour: Optional[int] = Field(default=None, ge=0, le=23)
    run_minute: Optional[int] = Field(default=None, ge=0, le=59)
    timezone_name: Optional[str] = Field(default=None, max_length=120)
    metadata: Optional[Dict[str, Any]] = None


class MemoryBriefingAutomationRequest(BaseModel):
    session_id: Optional[str] = None
    timezone_name: str = Field(default="America/Chicago", max_length=120)
    morning_hour: int = Field(default=8, ge=0, le=23)
    evening_hour: int = Field(default=18, ge=0, le=23)


class VoiceWakeConfigRequest(BaseModel):
    enabled: bool = False
    wake_word: str = Field(default="hey jarvis", max_length=80)
    threshold: float = Field(default=0.45, ge=0.05, le=0.99)
    chunk_ms: int = Field(default=960, ge=160, le=4000)


class VoiceWakeDetectRequest(BaseModel):
    pcm16_b64: str = Field(..., min_length=8)
    sample_rate: int = Field(default=16000, ge=8000, le=96000)
    channels: int = Field(default=1, ge=1, le=2)
    wake_word: Optional[str] = Field(default=None, max_length=80)
    threshold: Optional[float] = Field(default=None, ge=0.05, le=0.99)


class LocalVoiceConfigRequest(BaseModel):
    enabled: bool = True
    stt_model: str = Field(default="base", max_length=80)
    stt_device: str = Field(default="cpu", max_length=40)
    tts_provider: str = Field(default="enhanced_local", max_length=80)
    tts_voice: str = Field(default="mb-en1", max_length=80)
    tts_rate: int = Field(default=155, ge=80, le=320)
    tts_pitch: int = Field(default=45, ge=0, le=99)
    tts_style: str = Field(default="assistant", max_length=80)


class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class IntegrationConfigRequest(BaseModel):
    github_enabled: bool = False
    github_repo: str = Field(default="", max_length=200)
    github_token_set: Optional[bool] = None
    calendar_enabled: bool = False
    calendar_provider: str = Field(default="local", max_length=40)
    calendar_id: str = Field(default="", max_length=200)
    email_enabled: bool = False
    email_to: str = Field(default="", max_length=320)


class MissionStartRequest(BaseModel):
    workspace_id: Optional[int] = Field(default=None, ge=1)
    session_id: Optional[str] = None
    limit: int = Field(default=3, ge=1, le=8)
    auto_approve: bool = False
    retry_limit: int = Field(default=1, ge=0, le=5)
    goal: Optional[str] = Field(default=None, max_length=500)


class DesktopPresenceSnapshotRequest(BaseModel):
    workspace_id: Optional[int] = Field(default=None, ge=1)
    app_name: Optional[str] = Field(default=None, max_length=120)
    window_title: Optional[str] = Field(default=None, max_length=200)
    summary: Optional[str] = Field(default=None, max_length=400)
    details: Dict[str, Any] = Field(default_factory=dict)


class GitHubIssueCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(default="", max_length=8000)
    labels: List[str] = Field(default_factory=list, max_length=20)
    repo: Optional[str] = Field(default=None, max_length=200)


class GitHubPullReviewRequest(BaseModel):
    repo: Optional[str] = Field(default=None, max_length=200)
    pull_number: int = Field(..., ge=1)
    body: str = Field(default="", max_length=8000)
    event: str = Field(default="COMMENT", pattern="^(COMMENT|APPROVE|REQUEST_CHANGES)$")


class GitHubPullSummaryRequest(BaseModel):
    repo: Optional[str] = Field(default=None, max_length=200)
    pull_number: int = Field(..., ge=1)


class CalendarEventCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    starts_at: str = Field(..., min_length=8, max_length=80)
    ends_at: Optional[str] = Field(default=None, max_length=80)
    location: str = Field(default="", max_length=240)
    workspace_id: Optional[int] = Field(default=None, ge=1)


class EmailSendRequest(BaseModel):
    to: Optional[str] = Field(default=None, max_length=320)
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=12000)


class BrowserSessionHealthCheckRequest(BaseModel):
    session_name: Optional[str] = Field(default=None, max_length=120)
    session_id: Optional[int] = Field(default=None, ge=1)
    workspace_id: Optional[int] = Field(default=None, ge=1)
    limit: int = Field(default=10, ge=1, le=50)


class ProjectWatcherRequest(BaseModel):
    workspace_id: Optional[int] = Field(default=None, ge=1)
    watcher_type: str = Field(default="project", pattern="^(project|github|calendar|email|desktop)$")
    interval_minutes: int = Field(default=20, ge=5, le=10080)
    session_id: Optional[str] = None
    auto_approve: bool = False
    min_score: float = Field(default=6.0, ge=0.0, le=10.0)
    enabled: bool = True


class AgentLlmBenchmarkRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=20)
    runs_per_prompt: int = Field(default=1, ge=1, le=10)
    mode: str = Field(default="fast", pattern="^(fast|quality)$")
    model: Optional[str] = Field(default=None, max_length=200)
    timeout_s: int = Field(default=30, ge=5, le=240)
    dry_run: bool = False


class AgentEvalRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    runs: int = Field(default=3, ge=1, le=20)
    mode: str = Field(default="fast", pattern="^(fast|quality)$")
    model: Optional[str] = Field(default=None, max_length=200)
    dry_run: bool = False


class AgentChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_result: Optional[Dict[str, Any]] = None
    timestamp: str
    plan: Optional[List[str]] = None


class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: Dict[str, Any]


class AgentProfileResponse(BaseModel):
    profile: str
    dangerous_full_access: bool


class AgentProfileUpdateRequest(BaseModel):
    profile: str = Field(..., pattern="^(safe|dev|full)$")


class AgentTaskCreateRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class AgentTaskStatusRequest(BaseModel):
    status: str = Field(..., pattern="^(open|in_progress|done|blocked|failed)$")
    note: Optional[str] = Field(default=None, max_length=2000)


class GoalRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    auto_approve: bool = False


class GoalHistoryResponse(BaseModel):
    runs: List[Dict[str, Any]]


class GoalScheduleCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=4000)
    interval_minutes: int = Field(..., ge=1, le=10080)
    session_id: Optional[str] = None
    auto_approve: bool = False
    enabled: bool = True


class GoalScheduleUpdateRequest(BaseModel):
    goal: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    interval_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    auto_approve: Optional[bool] = None
    enabled: Optional[bool] = None


class QuantumSuperpositionRequest(BaseModel):
    states: List[str] = Field(..., min_length=2, max_length=64)


class QuantumEntangleRequest(BaseModel):
    system_a: str = Field(..., min_length=1, max_length=200)
    system_b: str = Field(..., min_length=1, max_length=200)


class QuantumMeasureRequest(BaseModel):
    measurement_basis: str = Field(default="computational", min_length=1, max_length=100)


class QuantumAlertConfigRequest(BaseModel):
    enabled: bool = True
    window_hours: int = Field(default=24, ge=1, le=24 * 30)
    min_measurements: int = Field(default=5, ge=1, le=5000)
    min_entangles: int = Field(default=3, ge=1, le=5000)
    outcome_one_min_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    outcome_one_max_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    entanglement_strength_min: float = Field(default=0.90, ge=0.0, le=1.0)


class QuantumExperimentRequest(BaseModel):
    preset: str = Field(default="quick", pattern="^(quick|balanced|deep)$")
    measure_count: Optional[int] = Field(default=None, ge=1, le=100)
    entangle_count: Optional[int] = Field(default=None, ge=0, le=100)


class QuantumRemediationConfigRequest(BaseModel):
    enabled: bool = True
    bias_threshold_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    entanglement_min: float = Field(default=0.8, ge=0.0, le=1.0)
    measure_iterations: int = Field(default=3, ge=1, le=20)
    entangle_iterations: int = Field(default=2, ge=0, le=20)
    preferred_basis: str = Field(default="computational", min_length=1, max_length=100)
    auto_rollback: bool = True
    rollback_measure_iterations: int = Field(default=2, ge=0, le=20)


class QuantumNotificationConfigRequest(BaseModel):
    enabled: bool = False
    channel: str = Field(default="generic", pattern="^(generic|slack|discord)$")
    webhook_url: str = Field(default="", max_length=2000)
    webhook_url_warning: str = Field(default="", max_length=2000)
    webhook_url_critical: str = Field(default="", max_length=2000)
    min_severity: str = Field(default="warning", pattern="^(info|warning|critical)$")


class QuantumAnnotationCreateRequest(BaseModel):
    item_type: str = Field(..., min_length=1, max_length=64)
    item_id: str = Field(..., min_length=1, max_length=128)
    note: str = Field(..., min_length=1, max_length=2000)
    author: Optional[str] = Field(default="operator", max_length=200)


class QuantumPlaybookRunRequest(BaseModel):
    incident_id: Optional[str] = Field(default=None, max_length=128)
    approve: bool = False
    dry_run: bool = True


class QuantumRbacConfigRequest(BaseModel):
    role: str = Field(..., pattern="^(viewer|operator|admin)$")


class QuantumSandboxRunRequest(BaseModel):
    name: str = Field(default="simulated_outage", min_length=1, max_length=200)
    hours: int = Field(default=24, ge=1, le=24 * 365)
    inject_alert_code: Optional[str] = Field(default="measurement_outcome_bias", max_length=128)
    inject_severity: str = Field(default="warning", pattern="^(info|warning|critical)$")
    drift_pct: float = Field(default=15.0, ge=0.0, le=100.0)


@dataclass(frozen=True)
class _Tool:
    name: str
    description: str
    args_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_get_time(_args: Dict[str, Any]) -> Dict[str, Any]:
    return {"utc": _now_iso()}


def _tool_echo(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"echo": args.get("text", "")}


def _tool_db_ping(_args: Dict[str, Any]) -> Dict[str, Any]:
    # Use the same engine the app config uses.
    from sqlalchemy import text

    from db_config import engine

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    return {"ok": True}


_APP_ROOT = Path("/home/jarvisuser/app")
_DATA_ROOT = _APP_ROOT / "data"
_MODELS_ROOT = _APP_ROOT / "models"
_DEFAULT_QUANTUM_CREATOR_KEY = "AETHERON_QUANTUM_CREATOR_KEY_2025"
_quantum_processor: Any = None
_quantum_error: Optional[str] = None


def _safe_path(user_path: str, *, allowed_roots: List[Path]) -> Path:
    if not isinstance(user_path, str) or not user_path.strip():
        raise HTTPException(status_code=400, detail="path is required")

    p = Path(user_path)
    if not p.is_absolute():
        p = (_APP_ROOT / p).resolve()
    else:
        p = p.resolve()

    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            p.relative_to(root_resolved)
            return p
        except ValueError:
            continue

    raise HTTPException(status_code=403, detail="path is outside allowed roots")


def _tool_list_files(args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path", ""))
    recursive = bool(args.get("recursive", False))
    max_entries = int(args.get("max_entries", 200))
    max_entries = max(1, min(max_entries, 2000))

    target = _safe_path(path, allowed_roots=[_APP_ROOT, _DATA_ROOT, _MODELS_ROOT])
    if not target.exists():
        return {
            "path": str(target),
            "entries": [],
            "truncated": False,
            "missing": True,
            "message": "path does not exist",
        }

    if target.is_file():
        st = target.stat()
        return {
            "path": str(target),
            "entries": [{"path": str(target), "type": "file", "size": st.st_size}],
            "truncated": False,
        }

    entries: List[Dict[str, Any]] = []
    it = target.rglob("*") if recursive else target.iterdir()
    for i, child in enumerate(it):
        if i >= max_entries:
            break
        try:
            st = child.stat()
        except OSError:
            continue
        entries.append(
            {
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "size": st.st_size if child.is_file() else None,
            }
        )

    return {
        "path": str(target),
        "recursive": recursive,
        "entries": entries,
        "truncated": len(entries) >= max_entries,
    }


def _tool_read_file(args: Dict[str, Any]) -> Dict[str, Any]:
    path = str(args.get("path", ""))
    max_bytes = int(args.get("max_bytes", 20000))
    max_bytes = max(256, min(max_bytes, 200000))

    target = _safe_path(path, allowed_roots=[_APP_ROOT, _DATA_ROOT, _MODELS_ROOT])
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file does not exist")

    data = target.read_bytes()
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    return {"path": str(target), "text": text, "truncated": truncated}


def _tool_write_file(args: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("JARVIS_ALLOW_WRITE", "").strip().lower() not in {"1", "true", "yes"}:
        raise HTTPException(
            status_code=403, detail="write tools are disabled (set JARVIS_ALLOW_WRITE=true)"
        )

    path = str(args.get("path", ""))
    content = args.get("content", "")
    append = bool(args.get("append", False))

    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content must be a string")

    target = _safe_path(path, allowed_roots=[_DATA_ROOT])
    target.parent.mkdir(parents=True, exist_ok=True)

    mode = "ab" if append else "wb"
    with open(target, mode) as f:
        f.write(content.encode("utf-8"))

    return {
        "path": str(target),
        "bytes_written": len(content.encode("utf-8")),
        "append": append,
    }


def _tool_system_info(_args: Dict[str, Any]) -> Dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "time_utc": _now_iso(),
        "llm_provider": (os.getenv("LLM_PROVIDER") or "").strip() or "auto",
        "ollama_model": (os.getenv("OLLAMA_MODEL") or "").strip(),
        "profile": _current_profile(),
        "dangerous_full_access": _dangerous_full_access_enabled(),
    }


def _is_truthy(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _dangerous_full_access_enabled() -> bool:
    return _is_truthy(os.getenv("JARVIS_DANGEROUS_FULL_ACCESS", ""))


def _default_profile() -> str:
    configured = (os.getenv("JARVIS_AGENT_PROFILE", "") or "").strip().lower()
    if configured in {"safe", "dev", "full"}:
        return configured
    return "full" if _dangerous_full_access_enabled() else "dev"


def _current_profile() -> str:
    stored = _task_memory.get_setting("agent_profile")
    if stored in {"safe", "dev", "full"}:
        return str(stored)
    value = _default_profile()
    _task_memory.set_setting("agent_profile", value)
    return value


def _set_profile(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in {"safe", "dev", "full"}:
        raise HTTPException(status_code=400, detail="Invalid profile")
    if v == "full" and not _dangerous_full_access_enabled():
        raise HTTPException(
            status_code=403,
            detail="full profile requires JARVIS_DANGEROUS_FULL_ACCESS=true",
        )
    _task_memory.set_setting("agent_profile", v)
    return v


def _llm_mode_default() -> str:
    return "quality"


def _llm_mode() -> str:
    mode = (_task_memory.get_setting("agent_llm_mode") or "").strip().lower()
    if mode in {"fast", "quality"}:
        return mode
    mode = _llm_mode_default()
    _task_memory.set_setting("agent_llm_mode", mode)
    return mode


def _set_llm_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in {"fast", "quality"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    _task_memory.set_setting("agent_llm_mode", m)
    return m


def _llm_model_override() -> str:
    return (_task_memory.get_setting("agent_llm_model") or "").strip()


def _set_llm_model_override(model_name: Optional[str]) -> str:
    model = str(model_name or "").strip()
    _task_memory.set_setting("agent_llm_model", model)
    return model


def _effective_ollama_runtime() -> Dict[str, Any]:
    mode = _llm_mode()
    override = _llm_model_override()
    available = ollama_list_models(timeout_s=5)
    chosen = override or ollama_model()
    if mode == "fast":
        if "llama3.2:1b" in available:
            chosen = "llama3.2:1b"
        elif "phi3:mini" in available:
            chosen = "phi3:mini"
    temperature = 0.2 if mode == "quality" else 0.5
    return {
        "mode": mode,
        "model": chosen,
        "temperature": temperature,
        "available_models": available,
    }


def _vision_provider_default() -> str:
    raw = (os.getenv("VISION_PROVIDER", "").strip().lower() or "auto")
    return raw if raw in {"auto", "heuristic", "ollama", "openai"} else "auto"


def _vision_provider() -> str:
    provider = (_task_memory.get_setting("agent_vision_provider") or "").strip().lower()
    if provider in {"auto", "heuristic", "ollama", "openai"}:
        return provider
    provider = _vision_provider_default()
    _task_memory.set_setting("agent_vision_provider", provider)
    return provider


def _set_vision_provider(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p not in {"auto", "heuristic", "ollama", "openai"}:
        raise HTTPException(status_code=400, detail="invalid vision provider")
    _task_memory.set_setting("agent_vision_provider", p)
    return p


def _vision_model_override() -> str:
    return (_task_memory.get_setting("agent_vision_model") or "").strip()


def _set_vision_model_override(model_name: Optional[str]) -> str:
    model = str(model_name or "").strip()
    _task_memory.set_setting("agent_vision_model", model)
    return model


def _effective_vision_runtime() -> Dict[str, Any]:
    provider = _vision_provider()
    override = _vision_model_override()
    available = ollama_list_models(timeout_s=5)
    chosen_model = override or os.getenv("OLLAMA_VISION_MODEL", "").strip() or ""
    if provider == "openai":
        chosen_model = override or os.getenv("OPENAI_VISION_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "gpt-4.1").strip()
    elif provider in {"auto", "ollama"}:
        chosen_model = chosen_model or (_pick_ollama_vision_model() or "")
    return {
        "provider": provider,
        "model": chosen_model,
        "available_models": available,
        "ocr_available": pytesseract is not None,
        "openai_configured": is_openai_configured(),
    }


def _agent_policy_default() -> Dict[str, Any]:
    return {"strict_confirm": True}


def _get_agent_policy() -> Dict[str, Any]:
    raw = _task_memory.get_setting("agent_policy")
    cfg = _agent_policy_default()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                cfg.update(parsed)
        except Exception:
            pass
    cfg["strict_confirm"] = bool(cfg.get("strict_confirm", True))
    return cfg


def _set_agent_policy(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _agent_policy_default()
    if isinstance(data, dict):
        cfg.update(data)
    cfg["strict_confirm"] = bool(cfg.get("strict_confirm", True))
    _task_memory.set_setting("agent_policy", json.dumps(cfg))
    return cfg


def _strict_confirm_enabled() -> bool:
    return bool(_get_agent_policy().get("strict_confirm"))


def _control_config_default() -> Dict[str, Any]:
    return {
        "browser_enabled": True,
        "desktop_enabled": True,
        "execute_on_host": False,
        "browser_name": "default",
        "search_engine": "google",
        "host_control_available": _is_truthy(os.getenv("JARVIS_HOST_CONTROL", "")),
    }


def _get_control_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("agent_control_config")
    cfg = _control_config_default()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                cfg.update(parsed)
        except Exception:
            pass
    cfg["host_control_available"] = _is_truthy(os.getenv("JARVIS_HOST_CONTROL", ""))
    cfg["execute_on_host"] = bool(cfg.get("execute_on_host")) and bool(cfg["host_control_available"])
    return cfg


def _set_control_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _control_config_default()
    if isinstance(data, dict):
        cfg.update(data)
    cfg["browser_enabled"] = bool(cfg.get("browser_enabled", True))
    cfg["desktop_enabled"] = bool(cfg.get("desktop_enabled", True))
    cfg["execute_on_host"] = bool(cfg.get("execute_on_host", False)) and _is_truthy(os.getenv("JARVIS_HOST_CONTROL", ""))
    browser_name = str(cfg.get("browser_name") or "default").strip().lower()
    cfg["browser_name"] = browser_name or "default"
    search_engine = str(cfg.get("search_engine") or "google").strip().lower()
    if search_engine not in {"google", "duckduckgo", "bing"}:
        search_engine = "google"
    cfg["search_engine"] = search_engine
    cfg["host_control_available"] = _is_truthy(os.getenv("JARVIS_HOST_CONTROL", ""))
    _task_memory.set_setting("agent_control_config", json.dumps(cfg))
    return cfg


def _host_control_command_result(*, kind: str, target: str, command: List[str], enabled: bool) -> Dict[str, Any]:
    preview = " ".join(command)
    if not enabled:
        return {
            "ok": True,
            "executed": False,
            "kind": kind,
            "target": target,
            "preview_command": preview,
            "reason": "Host control is disabled; preview only.",
        }
    proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
    return {
        "ok": proc.returncode == 0,
        "executed": True,
        "kind": kind,
        "target": target,
        "preview_command": preview,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[:2000],
        "stderr": (proc.stderr or "")[:2000],
    }


def _browser_search_url(query: str, engine: str) -> str:
    q = quote_plus(query)
    if engine == "duckduckgo":
        return f"https://duckduckgo.com/?q={q}"
    if engine == "bing":
        return f"https://www.bing.com/search?q={q}"
    return f"https://www.google.com/search?q={q}"


def _resolve_workspace_context(workspace_id: Optional[int] = None) -> Tuple[Optional[int], Optional[Dict[str, Any]], Dict[str, Any]]:
    resolved_workspace_id = int(workspace_id) if workspace_id else _task_memory.get_active_workspace_id()
    workspace = _task_memory.get_project_workspace(resolved_workspace_id) if resolved_workspace_id else None
    policy = _task_memory.get_workspace_policy(resolved_workspace_id)
    return resolved_workspace_id, workspace, policy


def _enforce_workspace_capability(capability: str, *, workspace_id: Optional[int] = None) -> Tuple[Optional[int], Optional[Dict[str, Any]], Dict[str, Any]]:
    resolved_workspace_id, workspace, policy = _resolve_workspace_context(workspace_id)
    flag_map = {
        "browser": "browser_allowed",
        "desktop": "desktop_allowed",
        "shell": "shell_allowed",
        "repo_write": "repo_write_allowed",
    }
    flag = flag_map.get(capability)
    if flag and not bool(policy.get(flag, True)):
        workspace_label = f" for workspace {workspace.get('name')}" if workspace else ""
        raise HTTPException(status_code=403, detail=f"{capability} actions are disabled{workspace_label}")
    return resolved_workspace_id, workspace, policy


def _policy_confirmation_required(policy: Dict[str, Any], *, confirm: bool) -> bool:
    return bool(policy.get("require_confirmation")) and not bool(confirm)


def _browser_open(url: str, *, workspace_id: Optional[int] = None, confirm: bool = False) -> Dict[str, Any]:
    cfg = _get_control_config()
    if not cfg.get("browser_enabled", True):
        raise HTTPException(status_code=403, detail="browser control is disabled")
    resolved_workspace_id, workspace, policy = _enforce_workspace_capability("browser", workspace_id=workspace_id)
    target = str(url or "").strip()
    if not re.match(r"^https?://", target, flags=re.IGNORECASE):
        target = "https://" + target
    approval_required = _policy_confirmation_required(policy, confirm=confirm) and bool(cfg.get("execute_on_host"))
    result = _host_control_command_result(
        kind="browser_open",
        target=target,
        command=["xdg-open", target],
        enabled=bool(cfg.get("execute_on_host")) and not approval_required,
    )
    result["workspace_id"] = resolved_workspace_id
    result["workspace_name"] = workspace.get("name") if workspace else None
    result["workspace_policy"] = policy
    if approval_required:
        result["approval_required"] = True
        result["reason"] = "Workspace policy requires confirm=true before host browser actions execute"
    return result


def _browser_search(query: str, *, workspace_id: Optional[int] = None, confirm: bool = False) -> Dict[str, Any]:
    cfg = _get_control_config()
    if not cfg.get("browser_enabled", True):
        raise HTTPException(status_code=403, detail="browser control is disabled")
    resolved_workspace_id, workspace, policy = _enforce_workspace_capability("browser", workspace_id=workspace_id)
    target_url = _browser_search_url(str(query or "").strip(), str(cfg.get("search_engine") or "google"))
    approval_required = _policy_confirmation_required(policy, confirm=confirm) and bool(cfg.get("execute_on_host"))
    result = _host_control_command_result(
        kind="browser_search",
        target=str(query or "").strip(),
        command=["xdg-open", target_url],
        enabled=bool(cfg.get("execute_on_host")) and not approval_required,
    )
    result["url"] = target_url
    result["workspace_id"] = resolved_workspace_id
    result["workspace_name"] = workspace.get("name") if workspace else None
    result["workspace_policy"] = policy
    if approval_required:
        result["approval_required"] = True
        result["reason"] = "Workspace policy requires confirm=true before host browser actions execute"
    return result


def _desktop_launch(app_name: str, *, workspace_id: Optional[int] = None, confirm: bool = False) -> Dict[str, Any]:
    cfg = _get_control_config()
    if not cfg.get("desktop_enabled", True):
        raise HTTPException(status_code=403, detail="desktop control is disabled")
    resolved_workspace_id, workspace, policy = _enforce_workspace_capability("desktop", workspace_id=workspace_id)
    safe_name = str(app_name or "").strip().lower()
    presets = {
        "files": ["xdg-open", str(_APP_ROOT)],
        "data": ["xdg-open", str(_DATA_ROOT)],
        "models": ["xdg-open", str(_MODELS_ROOT)],
        "dashboard": ["xdg-open", "http://localhost:8000/static/dashboard/index.html"],
        "terminal": ["x-terminal-emulator"],
        "browser": ["xdg-open", "http://localhost:8000/static/dashboard/index.html"],
    }
    command = presets.get(safe_name)
    if command is None:
        command = ["xdg-open", safe_name]
    approval_required = _policy_confirmation_required(policy, confirm=confirm) and bool(cfg.get("execute_on_host"))
    result = _host_control_command_result(
        kind="desktop_launch",
        target=safe_name,
        command=command,
        enabled=bool(cfg.get("execute_on_host")) and not approval_required,
    )
    result["workspace_id"] = resolved_workspace_id
    result["workspace_name"] = workspace.get("name") if workspace else None
    result["workspace_policy"] = policy
    if approval_required:
        result["approval_required"] = True
        result["reason"] = "Workspace policy requires confirm=true before host desktop actions execute"
    result["runtime"] = _desktop_runtime()
    return result


def _desktop_runtime() -> Dict[str, Any]:
    cfg = _get_control_config()
    return {
        "host_control_available": bool(cfg.get("host_control_available")),
        "execute_on_host": bool(cfg.get("execute_on_host")),
        "xdotool_available": bool(shutil.which("xdotool")),
        "open_command": "xdg-open",
    }


def _desktop_control(
    *,
    action: str,
    target: Optional[str] = None,
    text: Optional[str] = None,
    keys: Optional[List[str]] = None,
    workspace_id: Optional[int] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    cfg = _get_control_config()
    if not cfg.get("desktop_enabled", True):
        raise HTTPException(status_code=403, detail="desktop control is disabled")
    resolved_workspace_id, workspace, policy = _enforce_workspace_capability("desktop", workspace_id=workspace_id)
    safe_action = str(action or "").strip().lower()
    if safe_action == "launch":
        result = _desktop_launch(str(target or text or ""), workspace_id=resolved_workspace_id, confirm=confirm)
        result["action"] = safe_action
        return result

    runtime = _desktop_runtime()
    approval_required = _policy_confirmation_required(policy, confirm=confirm) and bool(cfg.get("execute_on_host"))
    enabled = bool(cfg.get("execute_on_host")) and not approval_required
    safe_target = str(target or "").strip()

    if safe_action == "open_url":
        if not re.match(r"^https?://", safe_target, flags=re.IGNORECASE):
            safe_target = "https://" + safe_target.lstrip("/")
        command = ["xdg-open", safe_target]
    elif safe_action == "type_text":
        typed = str(text or target or "").strip()
        if not typed:
            raise HTTPException(status_code=400, detail="text is required for type_text")
        if not runtime.get("xdotool_available"):
            enabled = False
        command = ["xdotool", "type", "--delay", "12", typed]
        safe_target = typed[:120]
    elif safe_action == "hotkey":
        combo = [str(item).strip() for item in list(keys or []) if str(item).strip()][:8]
        if not combo:
            raise HTTPException(status_code=400, detail="keys are required for hotkey")
        if not runtime.get("xdotool_available"):
            enabled = False
        command = ["xdotool", "key", "+".join(combo)]
        safe_target = "+".join(combo)
    else:
        raise HTTPException(status_code=400, detail="unsupported desktop action")

    result = _host_control_command_result(kind="desktop_control", target=safe_target, command=command, enabled=enabled)
    result["action"] = safe_action
    result["workspace_id"] = resolved_workspace_id
    result["workspace_name"] = workspace.get("name") if workspace else None
    result["workspace_policy"] = policy
    result["runtime"] = runtime
    if approval_required:
        result["approval_required"] = True
        result["reason"] = "Workspace policy requires confirm=true before host desktop actions execute"
    elif safe_action in {"type_text", "hotkey"} and not runtime.get("xdotool_available"):
        result["reason"] = "Desktop automation runtime is missing xdotool; preview only."
    return result


def _browser_workflow_runtime() -> Dict[str, Any]:
    explicit_path = Path(str(os.getenv("JARVIS_CHROMIUM_PATH") or "").strip()).expanduser() if str(os.getenv("JARVIS_CHROMIUM_PATH") or "").strip() else None
    candidates: List[Tuple[str, Path]] = []
    if explicit_path is not None:
        candidates.append(("env", explicit_path))
    if sync_playwright is not None:
        try:
            with sync_playwright() as p:  # type: ignore[misc]
                managed_path = Path(str(p.chromium.executable_path)).expanduser()
                candidates.append(("playwright", managed_path))
        except Exception:
            pass
    if os.name == "nt":
        local_app_data_raw = str(os.getenv("LOCALAPPDATA") or "").strip()
        if local_app_data_raw:
            local_app_data = Path(local_app_data_raw).expanduser()
            ms_playwright = local_app_data / "ms-playwright"
            for pattern in ("chromium-*\\chrome-win\\chrome.exe", "chromium-*\\chrome-win64\\chrome.exe"):
                for match in sorted(ms_playwright.glob(pattern), reverse=True):
                    candidates.append(("ms-playwright", match))
        for env_name, relative_path in (
            ("PROGRAMFILES", Path("Google/Chrome/Application/chrome.exe")),
            ("PROGRAMFILES(X86)", Path("Google/Chrome/Application/chrome.exe")),
            ("PROGRAMFILES", Path("Microsoft/Edge/Application/msedge.exe")),
            ("PROGRAMFILES(X86)", Path("Microsoft/Edge/Application/msedge.exe")),
        ):
            base_dir_raw = str(os.getenv(env_name) or "").strip()
            if base_dir_raw:
                base_dir = Path(base_dir_raw).expanduser()
                candidates.append((env_name.lower(), base_dir / relative_path))
    else:
        for command_name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "microsoft-edge"):
            command_path = shutil.which(command_name)
            if command_path:
                candidates.append(("path", Path(command_path)))
    checked: set[str] = set()
    selected_source: Optional[str] = None
    selected_path: Optional[Path] = None
    for source, candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in checked:
            continue
        checked.add(candidate_str)
        if candidate.exists():
            selected_source = source
            selected_path = candidate
            break
    install_hint = (
        f"{sys.executable} -m playwright install chromium"
        if sync_playwright is not None
        else "Install the playwright Python package and browser binaries."
    )
    reason = None if selected_path else f"Install browser binaries with `{install_hint}` or set JARVIS_CHROMIUM_PATH."
    return {
        "available": selected_path is not None,
        "chromium_path": str(selected_path) if selected_path else "",
        "headless_default": True,
        "source": selected_source,
        "reason": reason,
        "install_hint": install_hint,
    }


def _run_browser_workflow(
    *,
    start_url: Optional[str],
    steps: List[Dict[str, Any]],
    headless: bool = True,
    storage_state: Optional[Dict[str, Any]] = None,
    capture_storage_state: bool = False,
) -> Dict[str, Any]:
    runtime = _browser_workflow_runtime()
    if not runtime["available"]:
        detail = str(runtime.get("reason") or "Playwright browser runtime is unavailable")
        raise HTTPException(status_code=503, detail=detail)
    results: List[Dict[str, Any]] = []
    screenshot_b64: Optional[str] = None
    saved_storage_state: Optional[Dict[str, Any]] = None
    final_url = str(start_url or "").strip() or "about:blank"
    with sync_playwright() as p:  # type: ignore[misc]
        launch_kwargs: Dict[str, Any] = {
            "headless": bool(headless),
            "args": ["--no-sandbox", "--disable-dev-shm-usage"],
        }
        chromium_path = str(runtime.get("chromium_path") or "").strip()
        if chromium_path:
            launch_kwargs["executable_path"] = chromium_path
        try:
            browser = p.chromium.launch(**launch_kwargs)
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise HTTPException(status_code=503, detail=f"Playwright browser launch failed: {message}") from exc
        context_kwargs: Dict[str, Any] = {"viewport": {"width": 1440, "height": 900}}
        if isinstance(storage_state, dict) and storage_state:
            context_kwargs["storage_state"] = storage_state
        context = browser.new_context(**context_kwargs)
        page = context.new_page()
        page.goto(final_url, wait_until="domcontentloaded", timeout=30000)
        for idx, step in enumerate(steps, start=1):
            action = str(step.get("action") or "").strip().lower()
            selector = str(step.get("selector") or "").strip() or None
            value = str(step.get("value") or "")
            timeout_ms = int(step.get("timeout_ms") or 15000)
            entry: Dict[str, Any] = {"index": idx, "action": action, "selector": selector}
            try:
                if action == "goto":
                    target = value.strip() or selector or ""
                    if not re.match(r"^https?://", target, flags=re.IGNORECASE) and not target.startswith("data:") and target != "about:blank":
                        target = "https://" + target
                    page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)
                    entry["url"] = page.url
                elif action == "click":
                    if not selector:
                        raise ValueError("selector is required for click")
                    page.locator(selector).first.click(timeout=timeout_ms)
                elif action == "fill":
                    if not selector:
                        raise ValueError("selector is required for fill")
                    page.locator(selector).first.fill(value, timeout=timeout_ms)
                elif action == "press":
                    if not selector:
                        raise ValueError("selector is required for press")
                    page.locator(selector).first.press(value or "Enter", timeout=timeout_ms)
                elif action == "wait_for":
                    if selector:
                        page.locator(selector).first.wait_for(state="visible", timeout=timeout_ms)
                    else:
                        page.wait_for_timeout(timeout_ms)
                elif action == "extract_text":
                    if not selector:
                        raise ValueError("selector is required for extract_text")
                    entry["text"] = page.locator(selector).first.inner_text(timeout=timeout_ms)
                elif action == "extract_html":
                    if not selector:
                        raise ValueError("selector is required for extract_html")
                    entry["html"] = page.locator(selector).first.inner_html(timeout=timeout_ms)[:4000]
                elif action == "screenshot":
                    raw = page.screenshot(full_page=True, type="png", timeout=timeout_ms)
                    screenshot_b64 = raw.hex()
                    entry["bytes"] = len(raw)
                else:
                    raise ValueError(f"unsupported action: {action}")
                entry["ok"] = True
            except PlaywrightTimeoutError as exc:
                entry["ok"] = False
                entry["error"] = f"timeout: {exc}"
            except Exception as exc:
                entry["ok"] = False
                entry["error"] = str(exc)
            entry["url"] = page.url
            results.append(entry)
        final_url = page.url
        if capture_storage_state:
            try:
                saved_storage_state = context.storage_state()
            except Exception:
                saved_storage_state = None
        context.close()
        browser.close()
    extracted = [r for r in results if any(k in r for k in ("text", "html"))]
    return {
        "ok": all(bool(r.get("ok")) for r in results),
        "start_url": start_url or "about:blank",
        "final_url": final_url,
        "results": results,
        "extracted": extracted,
        "screenshot_hex": screenshot_b64,
        "storage_state": saved_storage_state,
        "runtime": runtime,
    }


def _due_reminder_text(reminder: Dict[str, Any]) -> str:
    workspace_name = str(reminder.get("workspace_name") or "").strip()
    prefix = f"Workspace {workspace_name}. " if workspace_name else ""
    return prefix + f"Reminder: {str(reminder.get('title') or '').strip()}. {str(reminder.get('content') or '').strip()}"


def _dispatch_due_reminders(limit: int = 10, *, include_discord: bool = True, include_voice: bool = True) -> Dict[str, Any]:
    cfg = _get_briefing_delivery_config()
    due_for_discord = _task_memory.list_due_proactive_reminders(limit=limit, for_discord=True) if include_discord else []
    due_for_voice = _task_memory.list_due_proactive_reminders(limit=limit, for_voice=True) if include_voice else []
    sent: List[Dict[str, Any]] = []
    if bool(cfg.get("enabled")) and bool(cfg.get("discord_enabled")) and str(cfg.get("discord_webhook_url") or "").strip():
        for reminder in due_for_discord:
            payload = {
                "text": _due_reminder_text(reminder),
                "period": "reminder",
                "workspace": reminder.get("workspace_name"),
                "priority": reminder.get("priority"),
            }
            result = _dispatch_briefing_discord(str(cfg.get("discord_webhook_url")), payload)
            _task_memory.update_proactive_reminder(int(reminder["id"]), discord_delivered=True, delivered=True)
            sent.append({"channel": "discord", "reminder_id": int(reminder["id"]), **result})
    queued_voice: List[Dict[str, Any]] = []
    for reminder in due_for_voice:
        _task_memory.update_proactive_reminder(int(reminder["id"]), voice_announced=True, delivered=True)
        queued_voice.append(
            {
                "id": int(reminder["id"]),
                "text": _due_reminder_text(reminder),
                "priority": reminder.get("priority"),
                "workspace_name": reminder.get("workspace_name"),
                "due_at": reminder.get("due_at"),
            }
        )
    return {"ok": True, "discord": sent, "voice": queued_voice}


_WAKEWORD_MODEL: Any = None
_WAKEWORD_MODEL_NAMES: List[str] = []
_WHISPER_MODELS: Dict[str, Any] = {}


def _wakeword_config_default() -> Dict[str, Any]:
    return {
        "enabled": False,
        "provider": "openwakeword",
        "wake_word": "hey jarvis",
        "threshold": 0.45,
        "chunk_ms": 960,
    }


def _local_voice_config_default() -> Dict[str, Any]:
    return {
        "enabled": True,
        "stt_provider": "faster_whisper",
        "stt_model": "base",
        "stt_device": "cpu",
        "tts_provider": "enhanced_local",
        "tts_voice": "mb-en1",
        "tts_rate": 145,
        "tts_pitch": 34,
        "tts_style": "assistant",
    }


def _local_tts_provider_capabilities() -> Dict[str, bool]:
    return {
        "espeak_ng": bool(shutil.which("espeak-ng")),
        "enhanced_local": bool(shutil.which("espeak-ng") and shutil.which("ffmpeg")),
    }


def _local_voice_presets() -> List[Dict[str, Any]]:
    return [
        {
            "id": "prime",
            "name": "Prime",
            "voice": "mb-en1",
            "rate": 138,
            "pitch": 32,
            "style": "cinematic",
            "description": "Steady cinematic assistant voice for the main Jarvis channel.",
        },
        {
            "id": "ops",
            "name": "Ops",
            "voice": "mb-en1",
            "rate": 148,
            "pitch": 28,
            "style": "operator",
            "description": "Sharper operations voice for alerts, commands, and mission traffic.",
        },
        {
            "id": "arc",
            "name": "Arc",
            "voice": "en-us",
            "rate": 154,
            "pitch": 42,
            "style": "crisp",
            "description": "Lighter futuristic voice for fast replies and lighter interactions.",
        },
    ]


def _get_local_voice_config() -> Dict[str, Any]:
    cfg = _local_voice_config_default()
    raw = _task_memory.get_setting("local_voice_config")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                cfg.update(parsed)
        except Exception:
            pass
    caps = _local_tts_provider_capabilities()
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["stt_model"] = str(cfg.get("stt_model") or "base").strip() or "base"
    cfg["stt_device"] = str(cfg.get("stt_device") or "cpu").strip() or "cpu"
    cfg["tts_provider"] = str(cfg.get("tts_provider") or "enhanced_local").strip().lower() or "enhanced_local"
    if cfg["tts_provider"] == "enhanced_local" and not caps.get("enhanced_local"):
        cfg["tts_provider"] = "espeak_ng"
    cfg["tts_voice"] = str(cfg.get("tts_voice") or "mb-en1").strip() or "mb-en1"
    cfg["tts_rate"] = max(80, min(int(cfg.get("tts_rate", 145) or 145), 320))
    cfg["tts_pitch"] = max(0, min(int(cfg.get("tts_pitch", 34) or 34), 99))
    cfg["tts_style"] = str(cfg.get("tts_style") or "assistant").strip().lower() or "assistant"
    cfg["stt_available"] = bool(WhisperModel is not None)
    cfg["tts_available"] = bool(caps.get("espeak_ng"))
    cfg["tts_enhanced_available"] = bool(caps.get("enhanced_local"))
    cfg["available_tts_providers"] = list(caps.keys())
    cfg["available_tts_styles"] = ["assistant", "deep", "crisp", "cinematic", "operator", "broadcast"]
    cfg["available_stt_models"] = ["tiny", "base", "small", "medium"]
    cfg["available_voice_presets"] = _local_voice_presets()
    return cfg


def _set_local_voice_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _local_voice_config_default()
    if isinstance(data, dict):
        cfg.update(data)
    cfg["enabled"] = bool(cfg.get("enabled", True))
    cfg["stt_model"] = str(cfg.get("stt_model") or "base").strip() or "base"
    cfg["stt_device"] = str(cfg.get("stt_device") or "cpu").strip() or "cpu"
    cfg["tts_provider"] = str(cfg.get("tts_provider") or "enhanced_local").strip().lower() or "enhanced_local"
    cfg["tts_voice"] = str(cfg.get("tts_voice") or "mb-en1").strip() or "mb-en1"
    cfg["tts_rate"] = max(80, min(int(cfg.get("tts_rate", 145) or 145), 320))
    cfg["tts_pitch"] = max(0, min(int(cfg.get("tts_pitch", 34) or 34), 99))
    cfg["tts_style"] = str(cfg.get("tts_style") or "assistant").strip().lower() or "assistant"
    _task_memory.set_setting("local_voice_config", json.dumps(cfg))
    return _get_local_voice_config()


def _integration_config_default() -> Dict[str, Any]:
    return {
        "github_enabled": False,
        "github_repo": "",
        "github_token_set": False,
        "calendar_enabled": False,
        "calendar_provider": "local",
        "calendar_id": "",
        "email_enabled": False,
        "email_to": "",
    }


def _get_integration_config() -> Dict[str, Any]:
    cfg = _integration_config_default()
    raw = _task_memory.get_json_setting("elite_integration_config", {})
    if isinstance(raw, dict):
        cfg.update(raw)
    cfg["github_enabled"] = bool(cfg.get("github_enabled", False))
    cfg["github_repo"] = str(cfg.get("github_repo") or "").strip()
    cfg["github_token_set"] = bool(cfg.get("github_token_set", False) or os.getenv("GITHUB_TOKEN"))
    cfg["calendar_enabled"] = bool(cfg.get("calendar_enabled", False))
    cfg["calendar_provider"] = str(cfg.get("calendar_provider") or "local").strip() or "local"
    cfg["calendar_id"] = str(cfg.get("calendar_id") or "").strip()
    cfg["email_enabled"] = bool(cfg.get("email_enabled", False))
    cfg["email_to"] = str(cfg.get("email_to") or "").strip()
    cfg["connections"] = {
        "github": {
            "enabled": cfg["github_enabled"],
            "connected": bool(cfg["github_token_set"]),
            "repo": cfg["github_repo"],
        },
        "calendar": {
            "enabled": cfg["calendar_enabled"],
            "connected": bool(cfg["calendar_id"]),
            "provider": cfg["calendar_provider"],
            "calendar_id": cfg["calendar_id"],
        },
        "email": {
            "enabled": cfg["email_enabled"],
            "connected": bool(cfg["email_to"]),
            "email_to": cfg["email_to"],
        },
    }
    return cfg


def _set_integration_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _integration_config_default()
    if isinstance(data, dict):
        cfg.update(data)
    cfg["github_enabled"] = bool(cfg.get("github_enabled", False))
    cfg["github_repo"] = str(cfg.get("github_repo") or "").strip()
    cfg["github_token_set"] = bool(cfg.get("github_token_set", False) or os.getenv("GITHUB_TOKEN"))
    cfg["calendar_enabled"] = bool(cfg.get("calendar_enabled", False))
    cfg["calendar_provider"] = str(cfg.get("calendar_provider") or "local").strip() or "local"
    cfg["calendar_id"] = str(cfg.get("calendar_id") or "").strip()
    cfg["email_enabled"] = bool(cfg.get("email_enabled", False))
    cfg["email_to"] = str(cfg.get("email_to") or "").strip()
    _task_memory.set_json_setting(
        "elite_integration_config",
        {
            "github_enabled": cfg["github_enabled"],
            "github_repo": cfg["github_repo"],
            "github_token_set": cfg["github_token_set"],
            "calendar_enabled": cfg["calendar_enabled"],
            "calendar_provider": cfg["calendar_provider"],
            "calendar_id": cfg["calendar_id"],
            "email_enabled": cfg["email_enabled"],
            "email_to": cfg["email_to"],
        },
    )
    return _get_integration_config()


def _desktop_presence_payload(*, workspace_id: Optional[int] = None) -> Dict[str, Any]:
    active_workspace = None
    if isinstance(workspace_id, int):
        active_workspace = _task_memory.get_project_workspace(workspace_id)
    if active_workspace is None:
        active_workspace_id = _task_memory.get_active_workspace_id()
        if active_workspace_id:
            active_workspace = _task_memory.get_project_workspace(active_workspace_id)
    snapshot = _task_memory.latest_desktop_presence_snapshot(workspace_id=workspace_id if isinstance(workspace_id, int) else None)
    reminders = _task_memory.list_proactive_reminders(limit=5, status="open", workspace_id=workspace_id if isinstance(workspace_id, int) else None)
    next_actions = _task_memory.next_best_actions(workspace_id=workspace_id if isinstance(workspace_id, int) else None, limit=3)
    recent_vision = _task_memory.list_vision_observations(limit=1)
    return {
        "workspace": active_workspace,
        "snapshot": snapshot,
        "reminders": reminders,
        "next_actions": next_actions,
        "recent_vision": recent_vision[:1],
        "host_control_available": bool(_get_control_config().get("host_control_available")),
    }


def _run_and_store_mission(
    *,
    workspace_id: Optional[int],
    session_id: Optional[str],
    limit: int,
    auto_approve: bool,
    retry_limit: int = 1,
    goal: Optional[str] = None,
    mission_id: Optional[int] = None,
) -> Dict[str, Any]:
    prior_result = None
    if mission_id is not None:
        existing = _task_memory.get_mission_run(int(mission_id))
        if existing is not None and isinstance(existing.get("result"), dict):
            prior_result = existing.get("result")
    mission = _run_autonomous_mission(
        workspace_id=workspace_id,
        session_id=session_id,
        limit=limit,
        auto_approve=auto_approve,
        retry_limit=retry_limit,
        prior_result=prior_result,
    )
    status = "blocked" if mission.get("blocked") else ("completed" if mission.get("ok") else "partial")
    saved = _task_memory.save_mission_run(
        mission_id=mission_id,
        workspace_id=workspace_id,
        session_id=mission.get("session_id"),
        status=status,
        goal=goal or mission.get("summary"),
        auto_approve=auto_approve,
        limit_count=limit,
        summary=mission.get("summary"),
        result=mission | {"retry_limit": int(retry_limit)},
    )
    mission["mission_record"] = saved
    mission["mission_id"] = saved["id"]
    return mission


def _github_headers(token: str) -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "JarvisAI-Elite/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def _github_headers_optional(token: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "JarvisAI-Elite/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    safe_token = str(token or "").strip()
    if safe_token:
        headers["Authorization"] = f"Bearer {safe_token}"
    return headers


def _github_api_json(url: str, *, token: str = "") -> Dict[str, Any]:
    req = urlrequest.Request(url=url, headers=_github_headers_optional(token))
    with urlrequest.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def _github_api_json_list(url: str, *, token: str = "") -> List[Dict[str, Any]]:
    payload = _github_api_json(url, token=token)
    return payload if isinstance(payload, list) else []


def _github_issue_create(payload: GitHubIssueCreateRequest) -> Dict[str, Any]:
    cfg = _get_integration_config()
    repo = str(payload.repo or cfg.get("github_repo") or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="github repo is required")
    issue_payload = {
        "title": payload.title.strip(),
        "body": payload.body.strip(),
        "labels": [str(label).strip() for label in payload.labels if str(label).strip()][:20],
    }
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    if not token:
        return {
            "ok": True,
            "preview": True,
            "provider": "github",
            "repo": repo,
            "request": issue_payload,
            "reason": "GitHub integration is not fully connected; returning preview only.",
        }
    req = urlrequest.Request(
        url=f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps(issue_payload).encode("utf-8"),
        method="POST",
        headers=_github_headers(token),
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return {"ok": True, "preview": False, "provider": "github", "repo": repo, "issue": data}
    except Exception as exc:
        return {
            "ok": False,
            "preview": True,
            "provider": "github",
            "repo": repo,
            "request": issue_payload,
            "error": str(exc),
        }


def _github_pull_review(payload: GitHubPullReviewRequest) -> Dict[str, Any]:
    cfg = _get_integration_config()
    repo = str(payload.repo or cfg.get("github_repo") or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="github repo is required")
    review_payload = {
        "body": payload.body.strip(),
        "event": str(payload.event or "COMMENT").strip().upper() or "COMMENT",
    }
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    if not token:
        return {
            "ok": True,
            "preview": True,
            "provider": "github",
            "repo": repo,
            "pull_number": int(payload.pull_number),
            "request": review_payload,
            "reason": "GitHub integration is not fully connected; returning preview only.",
        }
    req = urlrequest.Request(
        url=f"https://api.github.com/repos/{repo}/pulls/{int(payload.pull_number)}/reviews",
        data=json.dumps(review_payload).encode("utf-8"),
        method="POST",
        headers=_github_headers(token),
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return {
            "ok": True,
            "preview": False,
            "provider": "github",
            "repo": repo,
            "pull_number": int(payload.pull_number),
            "review": data,
        }
    except Exception as exc:
        return {
            "ok": False,
            "preview": True,
            "provider": "github",
            "repo": repo,
            "pull_number": int(payload.pull_number),
            "request": review_payload,
            "error": str(exc),
        }


def _github_pull_summary(payload: GitHubPullSummaryRequest) -> Dict[str, Any]:
    cfg = _get_integration_config()
    repo = str(payload.repo or cfg.get("github_repo") or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="github repo is required")

    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    number = int(payload.pull_number)
    pull_url = f"https://api.github.com/repos/{repo}/pulls/{number}"
    files_url = f"https://api.github.com/repos/{repo}/pulls/{number}/files?per_page=20"
    reviews_url = f"https://api.github.com/repos/{repo}/pulls/{number}/reviews?per_page=10"
    issue_comments_url = f"https://api.github.com/repos/{repo}/issues/{number}/comments?per_page=10"
    try:
        pr = _github_api_json(pull_url, token=token)
        files = _github_api_json_list(files_url, token=token)
        reviews = _github_api_json_list(reviews_url, token=token)
        comments = _github_api_json_list(issue_comments_url, token=token)
    except Exception as exc:
        return {
            "ok": False,
            "preview": True,
            "provider": "github",
            "repo": repo,
            "pull_number": number,
            "error": str(exc),
            "reason": "GitHub PR summary could not be fetched live.",
        }

    changed_files: List[Dict[str, Any]] = []
    for item in files[:8]:
        if not isinstance(item, dict):
            continue
        patch_text = str(item.get("patch") or "").strip()
        patch_lines = [line for line in patch_text.splitlines()[:8] if line.strip()]
        changed_files.append(
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "additions": int(item.get("additions") or 0),
                "deletions": int(item.get("deletions") or 0),
                "changes": int(item.get("changes") or 0),
                "patch_excerpt": patch_lines,
            }
        )

    review_rollup: Dict[str, int] = {}
    for item in reviews:
        state = str(item.get("state") or "COMMENTED").strip().upper()
        review_rollup[state] = int(review_rollup.get(state, 0)) + 1

    return {
        "ok": True,
        "preview": not bool(token),
        "provider": "github",
        "repo": repo,
        "pull_number": number,
        "summary": {
            "title": pr.get("title"),
            "state": pr.get("state"),
            "draft": bool(pr.get("draft")),
            "mergeable_state": pr.get("mergeable_state"),
            "author": ((pr.get("user") or {}) if isinstance(pr.get("user"), dict) else {}).get("login"),
            "base_ref": ((pr.get("base") or {}) if isinstance(pr.get("base"), dict) else {}).get("ref"),
            "head_ref": ((pr.get("head") or {}) if isinstance(pr.get("head"), dict) else {}).get("ref"),
            "commits": int(pr.get("commits") or 0),
            "additions": int(pr.get("additions") or 0),
            "deletions": int(pr.get("deletions") or 0),
            "changed_files": int(pr.get("changed_files") or 0),
            "comments": int(pr.get("comments") or 0),
            "review_comments": int(pr.get("review_comments") or 0),
            "body_preview": str(pr.get("body") or "").strip()[:1200],
        },
        "reviews": review_rollup,
        "recent_comments": [
            {
                "author": ((item.get("user") or {}) if isinstance(item.get("user"), dict) else {}).get("login"),
                "body_preview": str(item.get("body") or "").strip()[:240],
                "created_at": item.get("created_at"),
            }
            for item in comments[:5]
            if isinstance(item, dict)
        ],
        "files": changed_files,
    }


def _calendar_events_get() -> List[Dict[str, Any]]:
    items = _task_memory.get_json_setting("elite_calendar_events", [])
    return items if isinstance(items, list) else []


def _calendar_events_set(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for item in items[:200]:
        if isinstance(item, dict):
            clean.append(item)
    _task_memory.set_json_setting("elite_calendar_events", clean)
    return clean


def _calendar_event_create(payload: CalendarEventCreateRequest) -> Dict[str, Any]:
    cfg = _get_integration_config()
    event = {
        "id": f"evt-{uuid4().hex[:12]}",
        "title": payload.title.strip(),
        "description": payload.description.strip(),
        "starts_at": payload.starts_at,
        "ends_at": payload.ends_at,
        "location": payload.location.strip(),
        "workspace_id": payload.workspace_id,
        "provider": cfg.get("calendar_provider") or "local",
    }
    provider = str(cfg.get("calendar_provider") or "local").strip().lower()
    token = str(os.getenv("GOOGLE_CALENDAR_ACCESS_TOKEN") or "").strip()
    calendar_id = str(cfg.get("calendar_id") or "primary").strip() or "primary"
    if bool(cfg.get("calendar_enabled")) and provider == "google" and token:
        body = {
            "summary": event["title"],
            "description": event["description"],
            "location": event["location"],
            "start": {"dateTime": payload.starts_at},
            "end": {"dateTime": payload.ends_at or payload.starts_at},
        }
        req = urlrequest.Request(
            url=f"https://www.googleapis.com/calendar/v3/calendars/{quote_plus(calendar_id)}/events",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "JarvisAI-Elite/1.0",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8") or "{}")
            return {"ok": True, "preview": False, "provider": "google", "event": data}
        except Exception as exc:
            event["error"] = str(exc)
    items = _calendar_events_get()
    items.insert(0, event)
    _calendar_events_set(items)
    return {"ok": True, "preview": provider != "local", "provider": provider or "local", "event": event, "items": _calendar_events_get()[:20]}


def _email_action_send(payload: EmailSendRequest) -> Dict[str, Any]:
    cfg = _get_integration_config()
    email_to = str(payload.to or cfg.get("email_to") or "").strip()
    if not email_to:
        raise HTTPException(status_code=400, detail="email target is required")
    if bool(cfg.get("email_enabled")) and str(os.getenv("SMTP_HOST") or "").strip():
        return _dispatch_briefing_email(email_to, payload.subject.strip(), payload.body)
    return {
        "ok": True,
        "preview": True,
        "provider": "email",
        "to": email_to,
        "subject": payload.subject.strip(),
        "body_preview": payload.body[:300],
        "reason": "SMTP is not configured for direct delivery; returning preview only.",
    }


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _integration_intelligence() -> Dict[str, Any]:
    cfg = _get_integration_config()
    now = datetime.now(timezone.utc)
    upcoming = []
    for event in _calendar_events_get():
        if not isinstance(event, dict):
            continue
        starts = _parse_iso_dt(event.get("starts_at"))
        if starts is None:
            continue
        if starts.tzinfo is None:
            starts = starts.replace(tzinfo=timezone.utc)
        if starts >= now:
            enriched = dict(event)
            enriched["_starts_dt"] = starts
            upcoming.append(enriched)
    upcoming.sort(key=lambda item: item["_starts_dt"])
    upcoming = upcoming[:5]

    desktop = _desktop_presence_payload()
    active_workspace = desktop.get("workspace") or {}
    snapshot = desktop.get("snapshot") or {}
    focus_file = ""
    if isinstance(snapshot.get("details"), dict):
        focus_file = str(snapshot["details"].get("focus_file") or "").strip()

    recommendations: List[Dict[str, Any]] = []
    if bool(cfg.get("connections", {}).get("github", {}).get("enabled")):
        recommendations.append(
            {
                "channel": "github",
                "title": "Review repo activity",
                "reason": f"GitHub is enabled for {cfg['connections']['github'].get('repo') or 'the current repo'}.",
            }
        )
    if upcoming:
        recommendations.append(
            {
                "channel": "calendar",
                "title": "Prepare for next calendar event",
                "reason": f"Upcoming event: {upcoming[0].get('title') or 'Untitled event'}.",
            }
        )
    if bool(cfg.get("connections", {}).get("email", {}).get("enabled")):
        recommendations.append(
            {
                "channel": "email",
                "title": "Queue a briefing email",
                "reason": f"Email delivery target is {cfg['connections']['email'].get('email_to') or 'configured'}.",
            }
        )
    if focus_file:
        recommendations.append(
            {
                "channel": "desktop",
                "title": "Stay anchored to current focus",
                "reason": f"Desktop focus is currently {focus_file}.",
            }
        )

    return {
        "connections": cfg.get("connections", {}),
        "workspace": active_workspace,
        "desktop_focus": {
            "app_name": snapshot.get("app_name"),
            "window_title": snapshot.get("window_title"),
            "focus_file": focus_file or None,
            "summary": snapshot.get("summary"),
        },
        "calendar_upcoming": [
            {k: v for k, v in item.items() if k != "_starts_dt"} for item in upcoming
        ],
        "recommendations": recommendations[:6],
        "summary": {
            "connected": sum(1 for item in cfg.get("connections", {}).values() if bool(item.get("connected"))),
            "enabled": sum(1 for item in cfg.get("connections", {}).values() if bool(item.get("enabled"))),
            "upcoming_events": len(upcoming),
            "active_workspace": active_workspace.get("name"),
        },
    }


def _desktop_awareness_payload(*, workspace_id: Optional[int] = None) -> Dict[str, Any]:
    presence = _desktop_presence_payload(workspace_id=workspace_id)
    snapshot = presence.get("snapshot") or {}
    details = snapshot.get("details") if isinstance(snapshot.get("details"), dict) else {}
    app_name = str(snapshot.get("app_name") or details.get("app_name") or "").strip()
    window_title = str(snapshot.get("window_title") or details.get("window_title") or "").strip()
    focus_file = str(details.get("focus_file") or "").strip()
    summary_text = "Desktop idle"
    mode = "standby"
    if focus_file or "code" in str(details.get("mode") or "").lower() or "code" in app_name.lower():
        mode = "coding"
        summary_text = f"Focused on {focus_file or app_name or 'development work'}."
    elif "mail" in app_name.lower() or "outlook" in app_name.lower():
        mode = "communication"
        summary_text = f"Handling communications in {app_name}."
    elif "calendar" in app_name.lower() or "meet" in window_title.lower():
        mode = "meeting"
        summary_text = f"Calendar or meeting context detected in {window_title or app_name}."

    now = datetime.now(timezone.utc)
    next_events = []
    for event in _calendar_events_get():
        starts = _parse_iso_dt(event.get("starts_at") if isinstance(event, dict) else None)
        if starts is None:
            continue
        if starts.tzinfo is None:
            starts = starts.replace(tzinfo=timezone.utc)
        delta_hours = (starts - now).total_seconds() / 3600.0
        if -1.0 <= delta_hours <= 24.0:
            next_events.append(event)
    next_events = next_events[:3]

    return {
        "presence": presence,
        "focus_mode": mode,
        "focus_summary": summary_text,
        "signals": {
            "app_name": app_name or None,
            "window_title": window_title or None,
            "focus_file": focus_file or None,
            "recent_vision_count": len(presence.get("recent_vision") or []),
            "open_reminders": len(presence.get("reminders") or []),
            "next_actions": len((presence.get("next_actions") or {}).get("actions") or []),
        },
        "nearby_calendar_events": next_events,
    }


def _watcher_network_payload() -> Dict[str, Any]:
    jobs = _task_memory.list_autonomous_jobs(limit=200, mode="watcher")
    cfg = _get_integration_config()
    latest_presence = _task_memory.latest_desktop_presence_snapshot()
    coverage: Dict[str, Dict[str, Any]] = {}
    for watcher_type in ["project", "github", "calendar", "email", "desktop"]:
        matching = [
            job for job in jobs
            if str((job.get("metadata") or {}).get("watcher_type") or "project").strip().lower() == watcher_type
        ]
        coverage[watcher_type] = {
            "enabled": sum(1 for item in matching if bool(item.get("enabled"))),
            "items": matching[:8],
        }

    recommendations: List[Dict[str, Any]] = []
    if not coverage["project"]["enabled"]:
        recommendations.append({"watcher_type": "project", "reason": "No project watcher is active yet."})
    if bool(cfg.get("connections", {}).get("github", {}).get("enabled")) and not coverage["github"]["enabled"]:
        recommendations.append({"watcher_type": "github", "reason": "GitHub is connected but no repo watcher is active."})
    if bool(cfg.get("connections", {}).get("calendar", {}).get("enabled")) and not coverage["calendar"]["enabled"]:
        recommendations.append({"watcher_type": "calendar", "reason": "Calendar is enabled but no schedule watcher is active."})
    if bool(cfg.get("connections", {}).get("email", {}).get("enabled")) and not coverage["email"]["enabled"]:
        recommendations.append({"watcher_type": "email", "reason": "Email delivery is ready but no inbox watcher is active."})
    if latest_presence and not coverage["desktop"]["enabled"]:
        recommendations.append({"watcher_type": "desktop", "reason": "Desktop presence is available but no desktop watcher is active."})

    return {
        "watchers": jobs,
        "coverage": coverage,
        "recommendations": recommendations,
        "summary": {
            "total_watchers": len(jobs),
            "enabled_watchers": sum(1 for item in jobs if bool(item.get("enabled"))),
            "connected_integrations": sum(1 for item in cfg.get("connections", {}).values() if bool(item.get("connected"))),
            "desktop_presence_available": bool(latest_presence),
            "typed_watchers": sorted([key for key, value in coverage.items() if int(value.get("enabled") or 0) > 0]),
        },
    }


def _trust_receipts_payload(*, limit: int = 20, session_id: Optional[str] = None) -> Dict[str, Any]:
    tool_runs = _task_memory.list_tool_executions(limit=limit, session_id=session_id)
    missions = _task_memory.list_mission_runs(limit=max(5, min(limit, 20)))
    receipts = []
    rollback_receipts = []
    for item in tool_runs:
        verification = item.get("verification") if isinstance(item.get("verification"), dict) else {}
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        args = detail.get("args") if isinstance(detail.get("args"), dict) else {}
        tool_name = str(item.get("tool_name") or "")
        rollback_receipt = None
        if tool_name in {"repo_write_file", "write_file"}:
            target_path = str((detail.get("result") or {}).get("path") or args.get("path") or "").strip()
            append_mode = bool(args.get("append"))
            rollback_receipt = {
                "kind": "file_write",
                "target": target_path or None,
                "strategy": "manual review" if append_mode else "restore previous contents or remove the file if it was newly created",
                "ready": bool(target_path),
                "command_preview": f"git diff -- {target_path}" if target_path else "inspect changed file",
            }
        elif tool_name == "shell_run":
            command = str((detail.get("result") or {}).get("command") or args.get("command") or "").strip()
            lowered = command.lower()
            if any(token in lowered for token in ["git", "rm ", "mv ", "cp "]):
                rollback_receipt = {
                    "kind": "repo_shell",
                    "target": command or None,
                    "strategy": "review command side effects before rollback",
                    "ready": bool(command),
                    "command_preview": "git status && git diff",
                }
        receipts.append(
            {
                "type": "tool",
                "id": item.get("id"),
                "tool_name": tool_name,
                "status": item.get("status"),
                "confidence": item.get("confidence"),
                "created_at": item.get("created_at"),
                "verified": str(item.get("status")) == "verified",
                "summary": verification.get("summary") or verification.get("status") or "tool run recorded",
                "rollback_hint": (rollback_receipt or {}).get("strategy") or ("manual review" if tool_name.startswith(("repo_", "shell_")) else "re-run safe"),
                "rollback_receipt": rollback_receipt,
                "detail": detail,
            }
        )
        if rollback_receipt:
            rollback_receipts.append(
                {
                    "tool_name": tool_name,
                    "created_at": item.get("created_at"),
                    "status": item.get("status"),
                    **rollback_receipt,
                }
            )
    mission_receipts = []
    for item in missions[:5]:
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        executed = result.get("executed") if isinstance(result.get("executed"), list) else []
        verified_steps = sum(
            1
            for step in executed
            if isinstance(step, dict) and isinstance(step.get("verification"), dict) and str(step["verification"].get("status") or "") == "verified"
        )
        mission_receipts.append(
            {
                "type": "mission",
                "id": item.get("id"),
                "status": item.get("status"),
                "goal": item.get("goal"),
                "created_at": item.get("created_at"),
                "summary": item.get("summary"),
                "verified_steps": verified_steps,
                "rollback_ready": bool(result.get("blocked")) or verified_steps > 0,
            }
        )
    return {
        "session_id": session_id,
        "receipts": receipts,
        "rollback_receipts": rollback_receipts,
        "missions": mission_receipts,
        "summary": {
            "tool_receipts": len(receipts),
            "mission_receipts": len(mission_receipts),
            "verified_receipts": sum(1 for item in receipts if item.get("verified")),
            "rollback_receipts": len(rollback_receipts),
            "rollback_ready": any(item.get("rollback_ready") for item in mission_receipts),
        },
    }


def _get_wakeword_config() -> Dict[str, Any]:
    cfg = _wakeword_config_default()
    raw = _task_memory.get_setting("voice_wakeword_config")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                cfg.update(parsed)
        except Exception:
            pass
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["threshold"] = max(0.05, min(float(cfg.get("threshold", 0.45) or 0.45), 0.99))
    cfg["chunk_ms"] = max(160, min(int(cfg.get("chunk_ms", 960) or 960), 4000))
    cfg["available"] = bool(OpenWakeWordModel is not None and np is not None)
    cfg["available_models"] = _list_wakeword_models()
    return cfg


def _set_wakeword_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _wakeword_config_default()
    if isinstance(data, dict):
        cfg.update(data)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["wake_word"] = str(cfg.get("wake_word") or "hey jarvis").strip() or "hey jarvis"
    cfg["threshold"] = max(0.05, min(float(cfg.get("threshold", 0.45) or 0.45), 0.99))
    cfg["chunk_ms"] = max(160, min(int(cfg.get("chunk_ms", 960) or 960), 4000))
    _task_memory.set_setting("voice_wakeword_config", json.dumps(cfg))
    return _get_wakeword_config()


def _extract_wakeword_model_names(model: Any) -> List[str]:
    names: List[str] = []
    for attr in ("models", "prediction_buffer", "model_outputs"):
        value = getattr(model, attr, None)
        if isinstance(value, dict):
            for key in value.keys():
                name = str(key or "").strip()
                if name and name not in names:
                    names.append(name)
    return names


def _list_wakeword_models() -> List[str]:
    global _WAKEWORD_MODEL_NAMES
    if _WAKEWORD_MODEL_NAMES:
        return list(_WAKEWORD_MODEL_NAMES)
    if OpenWakeWordModel is None or np is None:
        return []
    try:
        model = OpenWakeWordModel()
        _WAKEWORD_MODEL_NAMES = _extract_wakeword_model_names(model)
    except Exception:
        _WAKEWORD_MODEL_NAMES = []
    return list(_WAKEWORD_MODEL_NAMES)


def _get_wakeword_model() -> Tuple[Optional[Any], List[str]]:
    global _WAKEWORD_MODEL, _WAKEWORD_MODEL_NAMES
    if OpenWakeWordModel is None or np is None:
        return None, []
    if _WAKEWORD_MODEL is None:
        try:
            _WAKEWORD_MODEL = OpenWakeWordModel()
            _WAKEWORD_MODEL_NAMES = _extract_wakeword_model_names(_WAKEWORD_MODEL)
        except Exception:
            _WAKEWORD_MODEL = None
            _WAKEWORD_MODEL_NAMES = []
    return _WAKEWORD_MODEL, list(_WAKEWORD_MODEL_NAMES)


def _get_whisper_model(model_name: Optional[str] = None, device: Optional[str] = None) -> Optional[Any]:
    if WhisperModel is None:
        return None
    safe_model = str(model_name or _get_local_voice_config().get("stt_model") or "base").strip() or "base"
    safe_device = str(device or _get_local_voice_config().get("stt_device") or "cpu").strip() or "cpu"
    key = f"{safe_model}:{safe_device}"
    cached = _WHISPER_MODELS.get(key)
    if cached is not None:
        return cached
    compute_type = "int8" if safe_device == "cpu" else "float16"
    try:
        model = WhisperModel(safe_model, device=safe_device, compute_type=compute_type)
    except Exception:
        model = WhisperModel(safe_model, device="cpu", compute_type="int8")
    _WHISPER_MODELS[key] = model
    return model


def _transcribe_audio_local(data: bytes, *, filename: str = "voice.wav") -> Dict[str, Any]:
    cfg = _get_local_voice_config()
    model = _get_whisper_model(cfg.get("stt_model"), cfg.get("stt_device"))
    if model is None:
        return {
            "ok": False,
            "provider": "faster_whisper",
            "available": False,
            "text": "",
            "error": "faster-whisper runtime unavailable",
        }
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        temp_path = tmp.name
    try:
        segments, info = model.transcribe(temp_path, vad_filter=True, beam_size=1)
        text = " ".join(str(segment.text or "").strip() for segment in segments).strip()
        language = getattr(info, "language", None)
        duration = getattr(info, "duration", None)
        return {
            "ok": True,
            "provider": "faster_whisper",
            "available": True,
            "text": text,
            "language": language,
            "duration": duration,
            "model": cfg.get("stt_model"),
        }
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _enhance_speech_wav(audio: bytes, *, style: str = "assistant") -> bytes:
    if not audio or not shutil.which("ffmpeg"):
        return audio
    style_name = str(style or "assistant").strip().lower()
    filters = {
        "assistant": "highpass=f=80,lowpass=f=7000,compand=attacks=0:points=-80/-900|-45/-18|-20/-8|0/-3,loudnorm=I=-16:TP=-1.5:LRA=7",
        "deep": "highpass=f=70,lowpass=f=5200,acompressor=threshold=0.08:ratio=2.5:attack=5:release=60,bass=g=4:f=120:w=0.8,loudnorm=I=-18:TP=-2:LRA=6",
        "crisp": "highpass=f=95,lowpass=f=7600,treble=g=2.5:f=3400:w=0.7,acompressor=threshold=0.08:ratio=2.0:attack=4:release=40,loudnorm=I=-15:TP=-1.5:LRA=8",
        "cinematic": "highpass=f=75,lowpass=f=6400,acompressor=threshold=0.09:ratio=2.2:attack=6:release=80,bass=g=2.5:f=140:w=0.7,aecho=0.8:0.6:22:0.12,loudnorm=I=-18:TP=-2:LRA=7",
        "operator": "highpass=f=95,lowpass=f=6900,acompressor=threshold=0.06:ratio=2.8:attack=3:release=45,treble=g=1.8:f=2800:w=0.5,loudnorm=I=-15:TP=-1.5:LRA=5",
        "broadcast": "highpass=f=90,lowpass=f=7200,acompressor=threshold=0.05:ratio=3.2:attack=2:release=35,deesser=i=0.4:m=0.5:f=0.5,loudnorm=I=-14:TP=-1.0:LRA=4",
    }
    af = filters.get(style_name) or filters["assistant"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as src, tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
        src.write(audio)
        src_path = src.name
        dst_path = dst.name
    try:
        proc = subprocess.run([
            "ffmpeg", "-y", "-i", src_path, "-af", af, dst_path
        ], capture_output=True, timeout=30)
        if proc.returncode != 0:
            return audio
        return Path(dst_path).read_bytes() if Path(dst_path).exists() else audio
    finally:
        for candidate in [src_path, dst_path]:
            try:
                os.unlink(candidate)
            except Exception:
                pass


def _synthesize_speech_local(text: str) -> bytes:
    cfg = _get_local_voice_config()
    preset = next(
        (
            item
            for item in _local_voice_presets()
            if str(item.get("style") or "") == str(cfg.get("tts_style") or "")
            or str(item.get("id") or "") == str(cfg.get("tts_style") or "")
        ),
        None,
    )
    voice_name = str(cfg.get("tts_voice") or (preset or {}).get("voice") or "mb-en1")
    rate_value = int(cfg.get("tts_rate") or (preset or {}).get("rate") or 145)
    pitch_value = int(cfg.get("tts_pitch") or (preset or {}).get("pitch") or 34)
    command = [
        "espeak-ng",
        "--stdout",
        "-v",
        voice_name,
        "-s",
        str(rate_value),
        "-p",
        str(pitch_value),
        str(text or "").strip()[:4000],
    ]
    proc = subprocess.run(command, capture_output=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(status_code=503, detail=(proc.stderr or b"speech synthesis failed").decode(errors="ignore")[:400])
    provider = str(cfg.get("tts_provider") or "enhanced_local").strip().lower()
    if provider in {"enhanced_local", "auto"}:
        return _enhance_speech_wav(proc.stdout, style=str(cfg.get("tts_style") or "assistant"))
    return proc.stdout


def _wakeword_aliases(wake_word: str) -> List[str]:
    requested = str(wake_word or "hey jarvis").strip().lower()
    aliases = [requested]
    if requested in {"jarvis", "hey jarvis"}:
        aliases.extend(["hey_jarvis", "jarvis", "hey jarvis"])
    return list(dict.fromkeys(aliases))


def _pcm16_to_16k_mono(data: bytes, *, sample_rate: int, channels: int) -> Any:
    if np is None:
        raise RuntimeError("numpy unavailable")
    audio = np.frombuffer(data, dtype=np.int16)
    if audio.size == 0:
        return audio
    if channels > 1:
        frames = audio[: (audio.size // channels) * channels].reshape(-1, channels)
        audio = frames.mean(axis=1).astype(np.int16)
    if int(sample_rate) == 16000:
        return audio.astype(np.int16)
    target_len = max(1, int(round(audio.size * (16000.0 / float(sample_rate or 16000)))))
    src_x = np.linspace(0, 1, num=audio.size, endpoint=False)
    dst_x = np.linspace(0, 1, num=target_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, audio.astype(np.float32))
    return resampled.astype(np.int16)


def _match_wakeword_score(scores: Dict[str, Any], wake_word: str) -> Tuple[str, float]:
    aliases = _wakeword_aliases(wake_word)
    best_name = ""
    best_score = 0.0
    for key, value in (scores or {}).items():
        name = str(key or "")
        score = float(value or 0.0)
        normalized = name.replace("_", " ").strip().lower()
        if any(alias in normalized or normalized in alias for alias in aliases):
            if score > best_score:
                best_name = name
                best_score = score
    if not best_name:
        for key, value in (scores or {}).items():
            score = float(value or 0.0)
            if score > best_score:
                best_name = str(key or "")
                best_score = score
    return best_name, round(best_score, 4)


def _wakeword_detect(payload: VoiceWakeDetectRequest) -> Dict[str, Any]:
    cfg = _get_wakeword_config()
    model, available_models = _get_wakeword_model()
    if model is None or np is None:
        return {
            "ok": False,
            "available": False,
            "provider": "openwakeword",
            "available_models": available_models,
            "error": "openwakeword runtime unavailable",
            "detected": False,
        }
    try:
        raw = base64.b64decode(payload.pcm16_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid pcm16 payload: {exc}") from exc
    audio = _pcm16_to_16k_mono(raw, sample_rate=payload.sample_rate, channels=payload.channels)
    if int(getattr(audio, "size", 0) or 0) == 0:
        return {"ok": False, "available": True, "detected": False, "error": "empty audio buffer", "available_models": available_models}
    if hasattr(model, "reset"):
        try:
            model.reset()
        except Exception:
            pass
    chunk = 1280
    max_scores: Dict[str, float] = {}
    for idx in range(0, int(audio.size), chunk):
        segment = audio[idx: idx + chunk]
        if int(segment.size) < chunk:
            pad = np.zeros(chunk, dtype=np.int16)
            pad[: int(segment.size)] = segment
            segment = pad
        scores = model.predict(segment)
        if isinstance(scores, dict):
            for key, value in scores.items():
                score = float(value or 0.0)
                if score > float(max_scores.get(str(key), 0.0)):
                    max_scores[str(key)] = score
    wake_word = payload.wake_word or str(cfg.get("wake_word") or "hey jarvis")
    matched_name, matched_score = _match_wakeword_score(max_scores, wake_word)
    threshold = float(payload.threshold if payload.threshold is not None else cfg.get("threshold", 0.45))
    detected = matched_score >= threshold
    return {
        "ok": True,
        "available": True,
        "provider": "openwakeword",
        "wake_word": wake_word,
        "detected": detected,
        "score": matched_score,
        "threshold": threshold,
        "matched_model": matched_name or None,
        "available_models": available_models,
        "sample_rate": 16000,
        "samples": int(audio.size),
    }


def _browser_templates_default() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Extract Title",
            "description": "Open a page and extract the H1 title.",
            "start_url": "data:text/html,<html><body><h1 id='title'>Jarvis Template</h1></body></html>",
            "category": "utility",
            "auth_template": False,
            "recommended_session_name": None,
            "provider": None,
            "healthcheck_url": None,
            "healthcheck_selector": None,
            "logged_out_markers": [],
            "healthy_markers": [],
            "steps": [
                {"action": "wait_for", "selector": "#title"},
                {"action": "extract_text", "selector": "#title"},
            ],
        },
        {
            "name": "Search Form Demo",
            "description": "Fill a text box and confirm the page is ready.",
            "start_url": "data:text/html,<html><body><input id='q'><button id='go'>Go</button></body></html>",
            "category": "utility",
            "auth_template": False,
            "recommended_session_name": None,
            "provider": None,
            "healthcheck_url": None,
            "healthcheck_selector": None,
            "logged_out_markers": [],
            "healthy_markers": [],
            "steps": [
                {"action": "wait_for", "selector": "#q"},
                {"action": "fill", "selector": "#q", "value": "jarvis"},
                {"action": "extract_html", "selector": "body"},
            ],
        },
        {
            "name": "GitHub Auth Capture",
            "description": "Open GitHub sign-in, wait for login, then capture page state and save the authenticated session.",
            "start_url": "https://github.com/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "github-auth",
            "provider": "github",
            "healthcheck_url": "https://github.com/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "login", "sign up"],
            "healthy_markers": ["your repositories", "signed in"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Discord Auth Capture",
            "description": "Open Discord login, wait for the authenticated shell, then save browser storage state for later reuse.",
            "start_url": "https://discord.com/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "discord-auth",
            "provider": "discord",
            "healthcheck_url": "https://discord.com/app",
            "healthcheck_selector": "body",
            "logged_out_markers": ["welcome back", "login", "log in"],
            "healthy_markers": ["friends", "servers", "messages"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Google Auth Capture",
            "description": "Open Google sign-in and preserve the authenticated session for later workflows after login completes.",
            "start_url": "https://accounts.google.com/",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "google-auth",
            "provider": "google",
            "healthcheck_url": "https://mail.google.com/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "choose an account"],
            "healthy_markers": ["inbox", "compose"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Slack Auth Capture",
            "description": "Open Slack sign-in and preserve workspace session cookies for later workflows.",
            "start_url": "https://slack.com/signin",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "slack-auth",
            "provider": "slack",
            "healthcheck_url": "https://app.slack.com/client",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "enter your workspace"],
            "healthy_markers": ["threads", "channels", "later"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Microsoft Auth Capture",
            "description": "Open Microsoft sign-in and store the authenticated session for Outlook and Microsoft 365 workflows.",
            "start_url": "https://login.microsoftonline.com/",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "microsoft-auth",
            "provider": "microsoft",
            "healthcheck_url": "https://outlook.office.com/mail/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "pick an account"],
            "healthy_markers": ["outlook", "inbox", "mail"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Linear Auth Capture",
            "description": "Open Linear sign-in and preserve the session for issue triage workflows.",
            "start_url": "https://linear.app/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "linear-auth",
            "provider": "linear",
            "healthcheck_url": "https://linear.app/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "log in"],
            "healthy_markers": ["my issues", "inbox", "linear"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "OpenAI Auth Capture",
            "description": "Open the OpenAI platform sign-in and preserve the session for dashboard workflows.",
            "start_url": "https://platform.openai.com/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "openai-auth",
            "provider": "openai",
            "healthcheck_url": "https://platform.openai.com/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "login", "sign up"],
            "healthy_markers": ["dashboard", "api keys", "projects"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Notion Auth Capture",
            "description": "Open Notion sign-in and preserve the authenticated workspace session.",
            "start_url": "https://www.notion.so/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "notion-auth",
            "provider": "notion",
            "healthcheck_url": "https://www.notion.so/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "continue with", "sign up"],
            "healthy_markers": ["search", "updates", "workspace"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Jira Auth Capture",
            "description": "Open Atlassian sign-in and save a Jira-ready browser session.",
            "start_url": "https://id.atlassian.com/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "jira-auth",
            "provider": "jira",
            "healthcheck_url": "https://id.atlassian.com/manage-profile/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "continue", "sign up"],
            "healthy_markers": ["profile", "account", "atlassian"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Figma Auth Capture",
            "description": "Open Figma sign-in and preserve the authenticated design session.",
            "start_url": "https://www.figma.com/login",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "figma-auth",
            "provider": "figma",
            "healthcheck_url": "https://www.figma.com/files/recent",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "continue with google", "sign up"],
            "healthy_markers": ["drafts", "recents", "team"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Gmail Auth Capture",
            "description": "Open Gmail after Google auth and preserve an inbox-ready session.",
            "start_url": "https://mail.google.com/",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "gmail-auth",
            "provider": "gmail",
            "healthcheck_url": "https://mail.google.com/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "choose an account"],
            "healthy_markers": ["inbox", "compose", "gmail"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
        {
            "name": "GitHub Issues Inbox",
            "description": "Open GitHub issues and capture the active triage surface.",
            "start_url": "https://github.com/issues",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "github-auth",
            "provider": "github",
            "healthcheck_url": "https://github.com/issues",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "login"],
            "healthy_markers": ["issues", "pull requests"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "GitHub PR Triage",
            "description": "Open pull requests and capture the current review and merge queue.",
            "start_url": "https://github.com/pulls",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "github-auth",
            "provider": "github",
            "healthcheck_url": "https://github.com/pulls",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "login"],
            "healthy_markers": ["pull requests", "review requests", "assigned to you"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Gmail Inbox Sweep",
            "description": "Open Gmail and capture the current inbox state for triage.",
            "start_url": "https://mail.google.com/mail/u/0/#inbox",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "gmail-auth",
            "provider": "gmail",
            "healthcheck_url": "https://mail.google.com/mail/u/0/#inbox",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "choose an account"],
            "healthy_markers": ["inbox", "compose"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Gmail Draft Review",
            "description": "Open Gmail drafts and capture the current outbound queue.",
            "start_url": "https://mail.google.com/mail/u/0/#drafts",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "gmail-auth",
            "provider": "gmail",
            "healthcheck_url": "https://mail.google.com/mail/u/0/#drafts",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "choose an account"],
            "healthy_markers": ["drafts", "compose"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Notion Workspace Search",
            "description": "Open Notion and capture the current workspace search surface.",
            "start_url": "https://www.notion.so/",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "notion-auth",
            "provider": "notion",
            "healthcheck_url": "https://www.notion.so/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "sign up"],
            "healthy_markers": ["search", "notion"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Figma Recent Files",
            "description": "Open Figma recent files and capture the active design workspace surface.",
            "start_url": "https://www.figma.com/files/recent",
            "category": "workflow",
            "auth_template": False,
            "recommended_session_name": "figma-auth",
            "provider": "figma",
            "healthcheck_url": "https://www.figma.com/files/recent",
            "healthcheck_selector": "body",
            "logged_out_markers": ["log in", "sign up"],
            "healthy_markers": ["recent", "drafts", "figma"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
            ],
        },
        {
            "name": "Outlook Auth Capture",
            "description": "Open Outlook on the web and preserve the authenticated inbox session.",
            "start_url": "https://outlook.office.com/mail/",
            "category": "auth",
            "auth_template": True,
            "recommended_session_name": "outlook-auth",
            "provider": "outlook",
            "healthcheck_url": "https://outlook.office.com/mail/",
            "healthcheck_selector": "body",
            "logged_out_markers": ["sign in", "pick an account"],
            "healthy_markers": ["inbox", "new mail", "outlook"],
            "steps": [
                {"action": "wait_for", "selector": "body", "timeout_ms": 30000},
                {"action": "screenshot", "timeout_ms": 30000},
                {"action": "extract_text", "selector": "body", "timeout_ms": 30000},
            ],
        },
    ]


def _get_browser_workflow_templates() -> List[Dict[str, Any]]:
    raw = _task_memory.get_setting("browser_workflow_templates")
    defaults = _browser_templates_default()
    items = list(defaults)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                stored = [item for item in parsed if isinstance(item, dict)]
                if stored:
                    merged: Dict[str, Dict[str, Any]] = {}
                    for item in defaults:
                        name = str(item.get("name") or "").strip().lower()
                        if name:
                            merged[name] = item
                    for item in stored:
                        name = str(item.get("name") or "").strip().lower()
                        if name:
                            merged[name] = item
                    items = list(merged.values())
        except Exception:
            pass
    return items


def _auth_browser_workflow_templates() -> List[Dict[str, Any]]:
    return [item for item in _get_browser_workflow_templates() if bool(item.get("auth_template"))]


def _workflow_library_templates(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    target = str(provider or "").strip().lower()
    items = [
        item
        for item in _get_browser_workflow_templates()
        if str(item.get("category") or "").strip().lower() == "workflow"
    ]
    if target:
        items = [item for item in items if str(item.get("provider") or "").strip().lower() == target]
    return items


def _set_browser_workflow_templates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items[:50]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "description": str(item.get("description") or "").strip(),
                "start_url": str(item.get("start_url") or "").strip() or None,
                "category": str(item.get("category") or "custom").strip() or "custom",
                "auth_template": bool(item.get("auth_template")),
                "recommended_session_name": str(item.get("recommended_session_name") or "").strip() or None,
                "provider": str(item.get("provider") or "").strip() or None,
                "healthcheck_url": str(item.get("healthcheck_url") or "").strip() or None,
                "healthcheck_selector": str(item.get("healthcheck_selector") or "").strip() or None,
                "logged_out_markers": [str(v).strip() for v in list(item.get("logged_out_markers") or []) if str(v).strip()][:12],
                "healthy_markers": [str(v).strip() for v in list(item.get("healthy_markers") or []) if str(v).strip()][:12],
                "steps": [dict(step) for step in list(item.get("steps") or []) if isinstance(step, dict)][:40],
            }
        )
    _task_memory.set_setting("browser_workflow_templates", json.dumps(normalized))
    return normalized


def _browser_template_by_name(name: Optional[str]) -> Optional[Dict[str, Any]]:
    target = str(name or "").strip().lower()
    if not target:
        return None
    for item in _get_browser_workflow_templates():
        if str(item.get("name") or "").strip().lower() == target:
            return item
    return None


def _browser_template_for_session(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    template_name = str(session.get("template_name") or "").strip()
    if template_name:
        found = _browser_template_by_name(template_name)
        if found:
            return found
    provider = str(session.get("provider") or "").strip().lower()
    if provider:
        for item in _auth_browser_workflow_templates():
            if str(item.get("provider") or "").strip().lower() == provider:
                return item
    return None


def _session_health_from_run(session: Dict[str, Any], template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    template = template or {}
    start_url = str(template.get("healthcheck_url") or template.get("start_url") or "about:blank").strip() or "about:blank"
    selector = str(template.get("healthcheck_selector") or "body").strip() or "body"
    steps = [
        {"action": "wait_for", "selector": selector, "timeout_ms": 20000},
        {"action": "extract_text", "selector": selector, "timeout_ms": 20000},
    ]
    result = _run_browser_workflow(
        start_url=start_url,
        steps=steps,
        headless=True,
        storage_state=session.get("storage_state") if isinstance(session.get("storage_state"), dict) else None,
        capture_storage_state=False,
    )
    extracted = " ".join(str(item.get("text") or "").strip() for item in list(result.get("extracted") or []))
    haystack = f"{str(result.get('final_url') or '')} {extracted}".lower()
    logged_out_markers = [str(v).strip().lower() for v in list(template.get("logged_out_markers") or []) if str(v).strip()]
    healthy_markers = [str(v).strip().lower() for v in list(template.get("healthy_markers") or []) if str(v).strip()]
    status = "healthy"
    if any(marker in haystack for marker in logged_out_markers):
        status = "login_required"
    elif healthy_markers and not any(marker in haystack for marker in healthy_markers):
        status = "warning"
    return {
        "status": status,
        "details": {
            "checked_url": start_url,
            "final_url": result.get("final_url"),
            "matched_logged_out": [marker for marker in logged_out_markers if marker in haystack],
            "matched_healthy": [marker for marker in healthy_markers if marker in haystack],
            "provider": template.get("provider"),
            "template_name": template.get("name"),
            "text_preview": extracted[:500],
        },
    }


def _verify_tool_result(tool_name: str, result: Dict[str, Any], args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = result if isinstance(result, dict) else {"value": result}
    tool = str(tool_name or "unknown").strip()
    checks: List[Dict[str, Any]] = []
    confidence: float = 0.6
    status = "verified"
    if tool == "get_time":
        ok = isinstance(payload.get("utc"), str) and "T" in str(payload.get("utc"))
        checks.append({"name": "utc_present", "ok": ok})
        confidence = 0.99 if ok else 0.25
    elif tool == "db_ping":
        ok = bool(payload.get("ok"))
        checks.append({"name": "db_ok", "ok": ok})
        confidence = 0.98 if ok else 0.2
    elif tool == "system_info":
        ok = bool(payload.get("python")) and bool(payload.get("platform"))
        checks.append({"name": "runtime_fields", "ok": ok})
        confidence = 0.97 if ok else 0.3
    elif tool == "list_files":
        ok = isinstance(payload.get("entries"), list) and bool(payload.get("path"))
        checks.append({"name": "entries_list", "ok": ok})
        confidence = 0.95 if ok else 0.35
    elif tool == "read_file":
        ok = isinstance(payload.get("text"), str) and bool(payload.get("path"))
        checks.append({"name": "text_loaded", "ok": ok})
        confidence = 0.94 if ok else 0.35
    elif tool in {"repo_write_file", "write_file"}:
        ok = int(payload.get("bytes_written") or 0) >= 0 and bool(payload.get("path"))
        checks.append({"name": "write_ack", "ok": ok})
        confidence = 0.91 if ok else 0.35
    elif tool == "shell_run":
        exit_code = payload.get("exit_code")
        ok = isinstance(exit_code, int) and int(exit_code) == 0
        checks.append({"name": "exit_code_zero", "ok": ok, "value": exit_code})
        confidence = 0.86 if ok else 0.42
    elif tool in {"browser_open", "browser_search", "desktop_launch", "desktop_control"}:
        preview = not bool(payload.get("executed"))
        ok = bool(payload.get("kind")) and bool(payload.get("target"))
        checks.append({"name": "control_response_shape", "ok": ok})
        status = "preview" if preview else ("verified" if bool(payload.get("ok")) else "failed")
        confidence = 0.74 if preview else (0.9 if bool(payload.get("ok")) else 0.35)
    elif tool == "browser_workflow":
        if not bool(payload.get("executed", True)):
            checks.append({"name": "workflow_preview", "ok": True})
            status = "preview"
            confidence = 0.72
        else:
            steps = list(payload.get("results") or [])
            ok_steps = [step for step in steps if bool(step.get("ok"))]
            checks.append({"name": "steps_ok", "ok": len(ok_steps) == len(steps), "value": f"{len(ok_steps)}/{len(steps)}"})
            extracted = list(payload.get("extracted") or [])
            checks.append({"name": "extracted_output", "ok": len(extracted) > 0})
            confidence = 0.93 if len(ok_steps) == len(steps) and extracted else (0.82 if len(ok_steps) == len(steps) else 0.4)
            status = "verified" if len(ok_steps) == len(steps) else "failed"
    else:
        ok = not bool(payload.get("error"))
        checks.append({"name": "generic_no_error", "ok": ok})
        confidence = 0.8 if ok else 0.4
    if any(not bool(check.get("ok")) for check in checks):
        if status != "preview":
            status = "failed"
    return {"status": status, "confidence": round(confidence, 3), "checks": checks}


def _pending_approval_key(session_id: str) -> str:
    return f"pending_approval:{session_id}"


def _save_pending_approval(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = dict(payload)
    entry["created_at"] = _now_iso()
    _task_memory.set_setting(_pending_approval_key(session_id), json.dumps(entry))
    return entry


def _get_pending_approval(session_id: str) -> Optional[Dict[str, Any]]:
    raw = _task_memory.get_setting(_pending_approval_key(session_id))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clear_pending_approval(session_id: str) -> bool:
    return _task_memory.delete_setting(_pending_approval_key(session_id))


def _message_is_approval(text: str) -> bool:
    value = str(text or "").strip().lower()
    return value in {
        "approve",
        "approved",
        "approve it",
        "yes",
        "yes do it",
        "do it",
        "run it",
        "execute it",
        "confirm",
        "confirmed",
        "go ahead",
    }


def _message_is_rejection(text: str) -> bool:
    value = str(text or "").strip().lower()
    return value in {"cancel", "deny", "reject", "stop", "never mind", "dont do it", "don't do it"}


def _result_needs_approval(result: Any) -> Tuple[bool, str]:
    if not isinstance(result, dict):
        return False, ""
    if bool(result.get("approval_required")):
        return True, str(result.get("reason") or "approval required")
    if bool(result.get("blocked")) and "confirm=true" in str(result.get("reason") or "").lower():
        return True, str(result.get("reason") or "approval required")
    return False, ""


def _execute_tool_with_approval(
    *,
    tool_name: str,
    args: Dict[str, Any],
    session_id: Optional[str],
    label: Optional[str] = None,
    approve: bool = False,
) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    safe_args = dict(args or {})
    if approve:
        safe_args["confirm"] = True
    try:
        result = _TOOLS[tool_name].handler(safe_args)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        result = {
            "ok": False,
            "blocked": exc.status_code in {401, 403},
            "status_code": exc.status_code,
            "error": detail,
            "reason": detail,
        }
    pending = None
    needs_approval, reason = _result_needs_approval(result)
    if needs_approval and session_id:
        pending = _save_pending_approval(
            session_id,
            {
                "tool": tool_name,
                "args": args,
                "label": label or tool_name,
                "reason": reason,
            },
        )
        result = dict(result)
        result["pending_approval"] = pending
    verification = _record_verified_tool_run(tool_name=tool_name, result=result, args=safe_args, session_id=session_id)
    return result, verification, pending


def _approval_prompt_text(entry: Dict[str, Any]) -> str:
    label = str(entry.get("label") or entry.get("tool") or "action").strip()
    reason = str(entry.get("reason") or "approval required").strip()
    return f"{label} is ready but waiting for approval. {reason}. Say `approve` to run it or `cancel` to clear it."


def _execute_pending_approval(session_id: str) -> Tuple[str, Dict[str, Any]]:
    pending = _get_pending_approval(session_id)
    if pending is None:
        return "There is no pending approval in this chat right now.", {"pending_approval": None}
    tool_name = str(pending.get("tool") or "").strip()
    args = pending.get("args") if isinstance(pending.get("args"), dict) else {}
    if tool_name not in _TOOLS:
        _clear_pending_approval(session_id)
        return "The pending action is no longer available, so I cleared it.", {"pending_approval": None}
    result, verification, new_pending = _execute_tool_with_approval(
        tool_name=tool_name,
        args=args,
        session_id=session_id,
        label=str(pending.get("label") or tool_name),
        approve=True,
    )
    if new_pending is None:
        _clear_pending_approval(session_id)
        return (
            f"Approved and executed `{tool_name}`.",
            {"tool": tool_name, "result": result, "verification": verification},
        )
    return _approval_prompt_text(new_pending), {"tool": tool_name, "result": result, "verification": verification, "pending_approval": new_pending}


def _run_autonomous_mission(
    *,
    workspace_id: Optional[int],
    session_id: Optional[str],
    limit: int,
    auto_approve: bool,
    retry_limit: int = 1,
    prior_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sid = session_id or str(uuid4())
    prior_checkpoint = (prior_result or {}).get("checkpoint") if isinstance(prior_result, dict) else {}
    stored_remaining = list(prior_checkpoint.get("remaining_actions") or []) if isinstance(prior_checkpoint, dict) else []
    plan = _task_memory.next_best_actions(workspace_id=workspace_id, limit=limit)
    actions = stored_remaining[:limit] if stored_remaining else list(plan.get("actions") or [])[:limit]
    executed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    retries_used = 0
    completed_titles: List[str] = list(prior_checkpoint.get("completed_titles") or []) if isinstance(prior_checkpoint, dict) else []

    for index, item in enumerate(actions):
        execution = item.get("execution") if isinstance(item.get("execution"), dict) else {}
        kind = str(execution.get("kind") or "").strip().lower()
        attempts = 0
        max_attempts = max(1, int(retry_limit) + 1)
        while attempts < max_attempts:
            attempts += 1
            try:
                if kind in {"activate_workspace", "reminder_done", "workspace_policy"}:
                    result = execute_next_action(NextActionExecuteRequest(action=item, session_id=sid, approve=auto_approve))
                    executed.append({"title": item.get("title"), "kind": kind, "result": result, "attempt": attempts})
                    completed_titles.append(str(item.get("title") or kind))
                    break

                if kind == "chat":
                    message = str(execution.get("message") or item.get("action") or "").strip()
                    if not message:
                        break
                    response = agent_chat(AgentChatRequest(message=message, session_id=sid))
                    entry = {
                        "title": item.get("title"),
                        "kind": "chat",
                        "reply": response.reply,
                        "tool_result": response.tool_result,
                        "attempt": attempts,
                    }
                    pending = response.tool_result.get("pending_approval") if isinstance(response.tool_result, dict) else None
                    if pending and not auto_approve:
                        blocked.append(entry)
                        remaining_actions = actions[index:]
                        checkpoint = {
                            "completed_titles": completed_titles,
                            "remaining_actions": remaining_actions,
                            "retry_limit": int(retry_limit),
                            "retry_attempts_used": retries_used,
                            "last_blocked_title": item.get("title"),
                        }
                        return {
                            "ok": False,
                            "session_id": sid,
                            "workspace_id": plan.get("workspace_id"),
                            "summary": plan.get("summary"),
                            "executed": executed,
                            "blocked": blocked,
                            "failed": failed,
                            "actions_considered": len(actions),
                            "checkpoint": checkpoint,
                            "can_resume": True,
                            "resumed": bool(stored_remaining),
                        }
                    if pending and auto_approve:
                        reply, tool_result = _execute_pending_approval(sid)
                        entry["reply"] = reply
                        entry["tool_result"] = tool_result
                    executed.append(entry)
                    completed_titles.append(str(item.get("title") or kind))
                    break

                executed.append({"title": item.get("title"), "kind": kind or "noop", "result": None, "attempt": attempts})
                completed_titles.append(str(item.get("title") or kind or "noop"))
                break
            except Exception as exc:
                retries_used += 1
                error_entry = {
                    "title": item.get("title"),
                    "kind": kind or "unknown",
                    "attempt": attempts,
                    "error": str(exc),
                }
                if attempts >= max_attempts:
                    failed.append(error_entry)
                    remaining_actions = actions[index:]
                    checkpoint = {
                        "completed_titles": completed_titles,
                        "remaining_actions": remaining_actions,
                        "retry_limit": int(retry_limit),
                        "retry_attempts_used": retries_used,
                        "last_failed_title": item.get("title"),
                    }
                    return {
                        "ok": False,
                        "session_id": sid,
                        "workspace_id": plan.get("workspace_id"),
                        "summary": plan.get("summary"),
                        "executed": executed,
                        "blocked": blocked,
                        "failed": failed,
                        "actions_considered": len(actions),
                        "checkpoint": checkpoint,
                        "can_resume": True,
                        "resumed": bool(stored_remaining),
                    }

    checkpoint = {
        "completed_titles": completed_titles,
        "remaining_actions": [],
        "retry_limit": int(retry_limit),
        "retry_attempts_used": retries_used,
    }
    return {
        "ok": len(blocked) == 0 and len(failed) == 0,
        "session_id": sid,
        "workspace_id": plan.get("workspace_id"),
        "summary": plan.get("summary"),
        "executed": executed,
        "blocked": blocked,
        "failed": failed,
        "actions_considered": len(actions),
        "checkpoint": checkpoint,
        "can_resume": False,
        "resumed": bool(stored_remaining),
    }


def _record_verified_tool_run(
    *,
    tool_name: str,
    result: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    verification = _verify_tool_result(tool_name, result, args=args)
    log_id = _task_memory.log_tool_execution(
        tool_name=tool_name,
        status=str(verification.get("status") or "unknown"),
        confidence=verification.get("confidence") if isinstance(verification.get("confidence"), (int, float)) else None,
        verification=verification,
        detail={"args": args or {}, "result": result},
        session_id=session_id,
    )
    enriched = dict(verification)
    enriched["log_id"] = log_id
    return enriched


def _is_risky_command(command: str) -> bool:
    c = (command or "").lower()
    risky_patterns = [
        r"(^|\s)rm\s+-",
        r"git\s+reset\b",
        r"git\s+clean\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bmkfs\b",
        r"(^|\s)dd\s+",
        r"chmod\s+000",
        r"chmod\s+-r",
        r"chown\s+-r",
        r"\bdel\s+/",
        r"\bformat\s+",
    ]
    return any(re.search(p, c) is not None for p in risky_patterns)


def _is_sensitive_path(path: Path) -> bool:
    rel = path.relative_to(_APP_ROOT)
    rel_str = str(rel).replace("\\", "/").lower()
    sensitive_prefixes = [".git/", ".github/workflows/"]
    sensitive_files = [".env", "docker-compose.yml"]
    if any(rel_str.startswith(p) for p in sensitive_prefixes):
        return True
    return rel.name.lower() in sensitive_files


def _requires_confirm_for_risk() -> bool:
    return _current_profile() in {"safe", "dev"}


def _build_plan(message: str) -> List[str]:
    msg = (message or "").strip().lower()
    if msg.startswith("/tool "):
        return ["Validate tool arguments", "Execute tool", "Return result"]
    if any(k in msg for k in ["fix", "edit", "write", "update", "change"]):
        return ["Understand requested change", "Inspect target files", "Apply edits", "Verify behavior", "Summarize"]
    if any(k in msg for k in ["run", "command", "shell", "terminal"]):
        return ["Parse command intent", "Assess risk profile", "Execute command", "Report output"]
    return ["Understand request", "Select tools if needed", "Execute steps", "Summarize result"]


def _get_quantum_processor() -> Any:
    global _quantum_processor, _quantum_error
    if _quantum_processor is not None:
        return _quantum_processor
    if _quantum_error:
        raise HTTPException(status_code=503, detail=f"Quantum processor unavailable: {_quantum_error}")

    try:
        from src.quantum.quantum_processor import QuantumProcessor

        qp = QuantumProcessor()
        creator_key = (os.getenv("JARVIS_QUANTUM_CREATOR_KEY") or _DEFAULT_QUANTUM_CREATOR_KEY).strip()
        qp.authenticate_creator(creator_key)
        _quantum_processor = qp
        return _quantum_processor
    except Exception as exc:
        _quantum_error = str(exc)
        raise HTTPException(status_code=503, detail=f"Quantum processor unavailable: {_quantum_error}")


def _tool_quantum_superposition(args: Dict[str, Any]) -> Dict[str, Any]:
    states_raw = args.get("states")
    if not isinstance(states_raw, list) or len(states_raw) < 2:
        raise HTTPException(status_code=400, detail="states must be a list with at least 2 entries")
    states = [str(s) for s in states_raw if str(s).strip()]
    if len(states) < 2:
        raise HTTPException(status_code=400, detail="states must contain at least 2 non-empty values")
    qp = _get_quantum_processor()
    result = _json_safe(qp.create_quantum_superposition(states))
    _task_memory.add_quantum_event(
        event_type="superposition",
        states=states,
    )
    return result


def _tool_quantum_entangle(args: Dict[str, Any]) -> Dict[str, Any]:
    system_a = str(args.get("system_a", "")).strip()
    system_b = str(args.get("system_b", "")).strip()
    if not system_a or not system_b:
        raise HTTPException(status_code=400, detail="system_a and system_b are required")
    qp = _get_quantum_processor()
    result = _json_safe(qp.quantum_entangle_systems(system_a, system_b))
    ent = result.get("entanglement") if isinstance(result.get("entanglement"), dict) else {}
    _task_memory.add_quantum_event(
        event_type="entangle",
        entanglement_strength=ent.get("entanglement_strength") if isinstance(ent.get("entanglement_strength"), (int, float)) else None,
        correlation_coefficient=ent.get("correlation_coefficient") if isinstance(ent.get("correlation_coefficient"), (int, float)) else None,
        states=[system_a, system_b],
    )
    return result


def _tool_quantum_measure(args: Dict[str, Any]) -> Dict[str, Any]:
    basis = str(args.get("measurement_basis", "computational")).strip() or "computational"
    qp = _get_quantum_processor()
    result = _json_safe(qp.measure_quantum_state(basis))
    meas = result.get("measurement") if isinstance(result.get("measurement"), dict) else {}
    _task_memory.add_quantum_event(
        event_type="measurement",
        measurement_basis=basis,
        outcome=meas.get("outcome") if isinstance(meas.get("outcome"), int) else None,
        measurement_probability=meas.get("measurement_probability") if isinstance(meas.get("measurement_probability"), (int, float)) else None,
    )
    return result


def _tool_quantum_decipher(args: Dict[str, Any]) -> Dict[str, Any]:
    hours_raw = args.get("hours", 24)
    try:
        hours = max(1, min(int(hours_raw), 24 * 365))
    except Exception:
        hours = 24
    event_type_raw = args.get("event_type")
    event_type = str(event_type_raw).strip() if event_type_raw is not None else None
    if event_type == "":
        event_type = None
    events = _task_memory.list_quantum_events(limit=500, event_type=event_type, since_hours=hours)
    return _quantum_decipher_analysis(events, hours=hours)


def _tool_quantum_experiment(args: Dict[str, Any]) -> Dict[str, Any]:
    preset = str(args.get("preset") or "quick").strip().lower()
    if preset not in {"quick", "balanced", "deep"}:
        preset = "quick"
    measure_count = args.get("measure_count")
    entangle_count = args.get("entangle_count")
    return _run_quantum_experiment(preset=preset, measure_count=measure_count, entangle_count=entangle_count)


def _tool_quantum_remediate(args: Dict[str, Any]) -> Dict[str, Any]:
    hours = int(args.get("hours") or 24)
    force = bool(args.get("force") or False)
    return _run_quantum_remediation(hours=hours, force=force)


def _json_safe(value: Any) -> Dict[str, Any]:
    def _default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return str(obj)

    return json.loads(json.dumps(value, default=_default))


def _quantum_alert_config_default() -> Dict[str, Any]:
    return {
        "enabled": True,
        "window_hours": 24,
        "min_measurements": 5,
        "min_entangles": 3,
        "outcome_one_min_pct": 20.0,
        "outcome_one_max_pct": 80.0,
        "entanglement_strength_min": 0.90,
    }


def _get_quantum_alert_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_alert_config")
    base = _quantum_alert_config_default()
    if not raw:
        return base
    try:
        payload = json.loads(raw)
    except Exception:
        return base
    if not isinstance(payload, dict):
        return base
    merged = dict(base)
    for k, v in payload.items():
        if k in merged:
            merged[k] = v
    return merged


def _set_quantum_alert_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    safe_cfg = {
        "enabled": bool(cfg.get("enabled", True)),
        "window_hours": int(cfg.get("window_hours", 24)),
        "min_measurements": int(cfg.get("min_measurements", 5)),
        "min_entangles": int(cfg.get("min_entangles", 3)),
        "outcome_one_min_pct": float(cfg.get("outcome_one_min_pct", 20.0)),
        "outcome_one_max_pct": float(cfg.get("outcome_one_max_pct", 80.0)),
        "entanglement_strength_min": float(cfg.get("entanglement_strength_min", 0.90)),
    }
    safe_cfg["window_hours"] = max(1, min(safe_cfg["window_hours"], 24 * 30))
    safe_cfg["min_measurements"] = max(1, min(safe_cfg["min_measurements"], 5000))
    safe_cfg["min_entangles"] = max(1, min(safe_cfg["min_entangles"], 5000))
    safe_cfg["outcome_one_min_pct"] = max(0.0, min(safe_cfg["outcome_one_min_pct"], 100.0))
    safe_cfg["outcome_one_max_pct"] = max(0.0, min(safe_cfg["outcome_one_max_pct"], 100.0))
    safe_cfg["entanglement_strength_min"] = max(0.0, min(safe_cfg["entanglement_strength_min"], 1.0))
    _task_memory.set_setting("quantum_alert_config", json.dumps(safe_cfg))
    return safe_cfg


def _evaluate_quantum_alerts(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("enabled", True)):
        return {"active": False, "alerts": [], "stats": _task_memory.quantum_stats(hours=int(cfg.get("window_hours", 24)))}

    stats = _task_memory.quantum_stats(hours=int(cfg.get("window_hours", 24)))
    alerts: List[Dict[str, Any]] = []

    measurements = int(stats.get("measurements") or 0)
    entangles = int(stats.get("entangles") or 0)
    outcomes = stats.get("measurement_outcomes") or {}
    ones = int(outcomes.get("1") or 0)
    one_pct = (ones / measurements * 100.0) if measurements > 0 else None

    if measurements >= int(cfg.get("min_measurements", 5)) and one_pct is not None:
        if one_pct < float(cfg.get("outcome_one_min_pct", 20.0)) or one_pct > float(cfg.get("outcome_one_max_pct", 80.0)):
            alerts.append(
                {
                    "code": "measurement_outcome_drift",
                    "severity": "warning",
                    "message": f"Outcome=1 ratio {one_pct:.1f}% is outside expected range.",
                    "value": one_pct,
                }
            )

    avg_ent = stats.get("avg_entanglement_strength")
    if entangles >= int(cfg.get("min_entangles", 3)) and isinstance(avg_ent, (int, float)):
        if float(avg_ent) < float(cfg.get("entanglement_strength_min", 0.90)):
            alerts.append(
                {
                    "code": "entanglement_strength_low",
                    "severity": "critical",
                    "message": f"Average entanglement strength {float(avg_ent):.3f} below threshold.",
                    "value": float(avg_ent),
                }
            )

    return {"active": len(alerts) > 0, "alerts": alerts, "stats": stats}


def _requires_extra_approval() -> bool:
    return _is_truthy(os.getenv("JARVIS_APPROVAL_FOR_DESTRUCTIVE", "false"))


def _goal_to_actions(goal: str) -> List[Dict[str, Any]]:
    """
    Convert a simple goal sentence into executable tool actions.

    This is intentionally deterministic so behavior is stable and auditable.
    """
    g = (goal or "").strip()
    lower = g.lower()
    actions: List[Dict[str, Any]] = []

    if "time" in lower:
        actions.append({"tool": "get_time", "args": {}, "label": "Fetch UTC time"})
    if "system info" in lower or "system status" in lower:
        actions.append({"tool": "system_info", "args": {}, "label": "Collect system info"})
    if "db ping" in lower or "database" in lower:
        actions.append({"tool": "db_ping", "args": {}, "label": "Check DB connectivity"})

    # Quantum templates
    if "quantum status" in lower:
        actions.append({"tool": "system_info", "args": {}, "label": "Collect system info (quantum context)"})
        actions.append(
            {
                "tool": "__chat__",
                "args": {"message": "/tool quantum_measure {\"measurement_basis\":\"computational\"}"},
                "label": "Run baseline quantum measurement",
            }
        )

    m_q_measure = re.search(r"\bquantum measure(?:\s+(?:basis|in)\s+([a-zA-Z0-9_\-]+))?", lower)
    if m_q_measure:
        basis = m_q_measure.group(1) if m_q_measure.group(1) else "computational"
        actions.append(
            {
                "tool": "quantum_measure",
                "args": {"measurement_basis": basis},
                "label": f"Quantum measure ({basis})",
            }
        )

    m_q_sup = re.search(r"\bquantum superposition(?:\s*:\s*|\s+)(.+)$", goal, flags=re.IGNORECASE)
    if m_q_sup:
        raw = m_q_sup.group(1).strip()
        states = [s.strip() for s in raw.split(",") if s.strip()]
        if len(states) >= 2:
            actions.append(
                {
                    "tool": "quantum_superposition",
                    "args": {"states": states[:64]},
                    "label": f"Quantum superposition ({len(states[:64])} states)",
                }
            )

    m_q_ent = re.search(r"\bquantum entangle(?:\s*:\s*|\s+)([^,]+),\s*([^,\n]+)", goal, flags=re.IGNORECASE)
    if m_q_ent:
        a = m_q_ent.group(1).strip()
        b = m_q_ent.group(2).strip()
        if a and b:
            actions.append(
                {
                    "tool": "quantum_entangle",
                    "args": {"system_a": a, "system_b": b},
                    "label": f"Quantum entangle {a} <-> {b}",
                }
            )

    m_q_decipher = re.search(r"\bquantum (?:de?cipher|decypher)(?:\s+(\d+)\s*h(?:ours?)?)?", lower)
    if m_q_decipher:
        hours = int(m_q_decipher.group(1)) if m_q_decipher.group(1) else 24
        hours = max(1, min(hours, 24 * 365))
        actions.append(
            {
                "tool": "quantum_decipher",
                "args": {"hours": hours},
                "label": f"Quantum decipher ({hours}h)",
            }
        )

    m_q_experiment = re.search(r"\bquantum experiment(?:\s+(quick|balanced|deep))?", lower)
    if m_q_experiment:
        preset = m_q_experiment.group(1) if m_q_experiment.group(1) else "quick"
        actions.append(
            {
                "tool": "quantum_experiment",
                "args": {"preset": preset},
                "label": f"Quantum experiment ({preset})",
            }
        )

    m_q_remediate = re.search(r"\bquantum remediation(?:\s+(\d+)\s*h(?:ours?)?)?", lower)
    if m_q_remediate:
        hours = int(m_q_remediate.group(1)) if m_q_remediate.group(1) else 24
        actions.append(
            {
                "tool": "quantum_remediate",
                "args": {"hours": max(1, min(hours, 24 * 365)), "force": False},
                "label": f"Quantum remediation ({hours}h)",
            }
        )

    m_list = re.search(r"\blist files(?:\s+(?:in|under|from)\s+([^\s]+))?", lower)
    if m_list:
        target = m_list.group(1) if m_list.group(1) else "data"
        actions.append(
            {
                "tool": "list_files",
                "args": {"path": target, "recursive": False, "max_entries": 200},
                "label": f"List files under {target}",
            }
        )

    m_read = re.search(r"\bread file\s+([^\s]+)", lower)
    if m_read:
        actions.append(
            {
                "tool": "read_file",
                "args": {"path": m_read.group(1)},
                "label": f"Read file {m_read.group(1)}",
            }
        )

    m_open = re.search(r"\bopen (?:url|website|site)\s+(.+)$", g, flags=re.IGNORECASE)
    if m_open:
        actions.append(
            {
                "tool": "browser_open",
                "args": {"url": m_open.group(1).strip()},
                "label": f"Open browser target {m_open.group(1).strip()}",
            }
        )

    m_search = re.search(r"\bsearch (?:the web )?for\s+(.+)$", g, flags=re.IGNORECASE)
    if m_search:
        actions.append(
            {
                "tool": "browser_search",
                "args": {"query": m_search.group(1).strip()},
                "label": f"Search browser for {m_search.group(1).strip()}",
            }
        )

    m_launch = re.search(r"\blaunch (?:app|desktop|program)?\s*:?(.+)$", g, flags=re.IGNORECASE)
    if m_launch and m_launch.group(1).strip():
        actions.append(
            {
                "tool": "desktop_launch",
                "args": {"app": m_launch.group(1).strip()},
                "label": f"Launch desktop target {m_launch.group(1).strip()}",
            }
        )

    m_switch_workspace = re.search(r"\b(?:switch|set|activate)\s+workspace(?:\s+to)?\s+(.+)$", g, flags=re.IGNORECASE)
    if m_switch_workspace:
        actions.append(
            {
                "tool": "__chat__",
                "args": {"message": f"switch workspace to {m_switch_workspace.group(1).strip()}"},
                "label": f"Switch workspace to {m_switch_workspace.group(1).strip()}",
            }
        )

    m_shell = re.search(r"\brun command:\s*(.+)$", g, flags=re.IGNORECASE)
    if m_shell:
        command = m_shell.group(1).strip()
        actions.append(
            {
                "tool": "shell_run",
                "args": {"command": command},
                "label": f"Run shell command: {command}",
            }
        )

    # Fallback: use LLM reasoning through /chat if no deterministic action was found.
    if not actions:
        actions.append({"tool": "__chat__", "args": {"message": g}, "label": "Ask Jarvis brain"})

    return actions


def _action_requires_approval(action: Dict[str, Any], profile: str) -> bool:
    tool = str(action.get("tool", ""))
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    workspace_id = args.get("workspace_id")
    _resolved_workspace_id, _workspace, policy = _resolve_workspace_context(workspace_id if isinstance(workspace_id, int) else None)

    if tool in {"browser_open", "browser_search", "desktop_launch", "browser_workflow"} and bool(policy.get("require_confirmation")):
        return True
    if tool == "shell_run":
        if not bool(policy.get("shell_allowed", True)):
            return True
        if bool(policy.get("require_confirmation")):
            return True
    if tool == "repo_write_file":
        if not bool(policy.get("repo_write_allowed", True)):
            return True
        if bool(policy.get("require_confirmation")):
            return True

    if _requires_extra_approval():
        if tool == "shell_run" and _is_risky_command(str(args.get("command", ""))):
            return True
        if tool == "repo_write_file":
            try:
                p = _safe_path(str(args.get("path", "")), allowed_roots=[_APP_ROOT])
                if _is_sensitive_path(p):
                    return True
            except Exception:
                return True

    if profile in {"safe", "dev"}:
        if tool == "shell_run" and _is_risky_command(str(args.get("command", ""))):
            return True
        if tool == "repo_write_file":
            try:
                p = _safe_path(str(args.get("path", "")), allowed_roots=[_APP_ROOT])
                if _is_sensitive_path(p):
                    return True
            except Exception:
                return True

    return False


def _tool_shell_run(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very guarded command runner for the container.

    - Disabled unless JARVIS_ALLOW_SHELL=true
    - No pipes/redirection/chaining
    - Allowlist of commands and subcommands
    """
    if not _is_truthy(os.getenv("JARVIS_ALLOW_SHELL", "")):
        raise HTTPException(status_code=403, detail="shell tool is disabled (set JARVIS_ALLOW_SHELL=true)")

    command = args.get("command", "")
    confirm = bool(args.get("confirm", False))
    workspace_id = args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None
    timeout_s = int(args.get("timeout_s", 30))
    timeout_s = max(1, min(timeout_s, 300))
    _resolved_workspace_id, workspace, policy = _enforce_workspace_capability("shell", workspace_id=workspace_id)

    if not isinstance(command, str) or not command.strip():
        raise HTTPException(status_code=400, detail="command must be a non-empty string")

    if _policy_confirmation_required(policy, confirm=confirm):
        return {
            "blocked": True,
            "reason": "Workspace policy requires confirm=true before shell execution",
            "profile": _current_profile(),
            "command": command,
            "workspace_name": workspace.get("name") if workspace else None,
        }

    if (_requires_confirm_for_risk() or _strict_confirm_enabled()) and _is_risky_command(command) and not confirm:
        return {
            "blocked": True,
            "reason": "Risky command requires confirm=true under current policy",
            "profile": _current_profile(),
            "command": command,
        }

    if _dangerous_full_access_enabled():
        # Full access mode intentionally allows arbitrary shell commands.
        proc = subprocess.run(
            ["/bin/sh", "-lc", command],
            cwd=str(_APP_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    else:
        # Block common shell metacharacters to prevent chaining/exfil via redirection.
        forbidden = ["|", "&&", "||", ";", ">", "<", "$(", "`"]
        if any(tok in command for tok in forbidden):
            raise HTTPException(status_code=400, detail="command contains forbidden shell operators")

        argv = shlex.split(command)
        if not argv:
            raise HTTPException(status_code=400, detail="command parse failed")

        exe = argv[0]
        allowed = {"ls", "cat", "head", "tail", "wc", "pytest", "python", "git"}
        if exe not in allowed:
            raise HTTPException(status_code=403, detail=f"command not allowed: {exe}")

        if exe == "python":
            # Allow only safe invocations.
            if argv[1:] not in (["--version"], ["-V"], ["-m", "pytest"]):
                raise HTTPException(status_code=403, detail="python invocation not allowed")

        if exe == "git":
            if len(argv) < 2 or argv[1] not in {"status", "diff", "log", "show", "rev-parse"}:
                raise HTTPException(status_code=403, detail="git subcommand not allowed")

        proc = subprocess.run(
            argv,
            cwd=str(_APP_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def _truncate(s: str) -> str:
        s = s or ""
        return s if len(s) <= 8000 else (s[:8000] + "...")

    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": _truncate(proc.stdout),
        "stderr": _truncate(proc.stderr),
    }


def _tool_repo_write_file(args: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_truthy(os.getenv("JARVIS_ALLOW_REPO_WRITE", "")):
        raise HTTPException(status_code=403, detail="repo write is disabled (set JARVIS_ALLOW_REPO_WRITE=true)")

    path = str(args.get("path", ""))
    content = args.get("content", "")
    append = bool(args.get("append", False))
    confirm = bool(args.get("confirm", False))
    workspace_id = args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None
    _resolved_workspace_id, workspace, policy = _enforce_workspace_capability("repo_write", workspace_id=workspace_id)

    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content must be a string")
    if len(content) > 200_000:
        raise HTTPException(status_code=400, detail="content too large")

    target = _safe_path(path, allowed_roots=[_APP_ROOT])

    if _policy_confirmation_required(policy, confirm=confirm):
        return {
            "blocked": True,
            "reason": "Workspace policy requires confirm=true before repo writes",
            "profile": _current_profile(),
            "path": str(target),
            "workspace_name": workspace.get("name") if workspace else None,
        }

    if (_requires_confirm_for_risk() or _strict_confirm_enabled()) and _is_sensitive_path(target) and not confirm:
        return {
            "blocked": True,
            "reason": "Sensitive file write requires confirm=true under current policy",
            "profile": _current_profile(),
            "path": str(target),
        }

    # In normal mode, prevent self-footguns and secret leaks.
    if not _dangerous_full_access_enabled():
        rel = target.relative_to(_APP_ROOT)
        if rel.parts and rel.parts[0] in {".git"}:
            raise HTTPException(status_code=403, detail="writes to .git are blocked")
        if rel.name in {".env"}:
            raise HTTPException(status_code=403, detail="writes to .env are blocked")

    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if append else "wb"
    with open(target, mode) as f:
        f.write(content.encode("utf-8"))

    return {"path": str(target), "bytes_written": len(content.encode("utf-8")), "append": append}


def _tool_browser_open(args: Dict[str, Any]) -> Dict[str, Any]:
    return _browser_open(
        str(args.get("url", "")),
        workspace_id=args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None,
        confirm=bool(args.get("confirm", False)),
    )


def _tool_browser_search(args: Dict[str, Any]) -> Dict[str, Any]:
    return _browser_search(
        str(args.get("query", "")),
        workspace_id=args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None,
        confirm=bool(args.get("confirm", False)),
    )


def _tool_desktop_launch(args: Dict[str, Any]) -> Dict[str, Any]:
    return _desktop_launch(
        str(args.get("app", "")),
        workspace_id=args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None,
        confirm=bool(args.get("confirm", False)),
    )


def _tool_desktop_control(args: Dict[str, Any]) -> Dict[str, Any]:
    return _desktop_control(
        action=str(args.get("action", "")),
        target=args.get("target") if isinstance(args.get("target"), str) else None,
        text=args.get("text") if isinstance(args.get("text"), str) else None,
        keys=[str(item) for item in list(args.get("keys") or []) if str(item).strip()],
        workspace_id=args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None,
        confirm=bool(args.get("confirm", False)),
    )


def _tool_browser_workflow(args: Dict[str, Any]) -> Dict[str, Any]:
    steps = args.get("steps")
    if not isinstance(steps, list) or not steps:
        raise HTTPException(status_code=400, detail="steps are required")
    workspace_id = args.get("workspace_id") if isinstance(args.get("workspace_id"), int) else None
    _resolved_workspace_id, workspace, policy = _enforce_workspace_capability("browser", workspace_id=workspace_id)
    if _policy_confirmation_required(policy, confirm=bool(args.get("confirm", False))):
        return {
            "ok": True,
            "executed": False,
            "approval_required": True,
            "workspace_name": workspace.get("name") if workspace else None,
            "workspace_policy": policy,
            "results": [],
            "extracted": [],
            "runtime": _browser_workflow_runtime(),
        }
    session_name = str(args.get("session_name") or "").strip() or None
    stored_session = _task_memory.get_browser_session(session_name) if session_name else None
    return _run_browser_workflow(
        start_url=str(args.get("start_url") or "").strip() or None,
        steps=[dict(step) if isinstance(step, dict) else {} for step in steps],
        headless=bool(args.get("headless", True)),
        storage_state=(
            args.get("storage_state")
            if isinstance(args.get("storage_state"), dict)
            else (stored_session.get("storage_state") if stored_session else None)
        ),
        capture_storage_state=bool(args.get("capture_storage_state", False)),
    )


_TOOLS: Dict[str, _Tool] = {
    "get_time": _Tool(
        name="get_time",
        description="Get the current UTC timestamp.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_tool_get_time,
    ),
    "db_ping": _Tool(
        name="db_ping",
        description="Verify the configured SQL database is reachable (runs SELECT 1).",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_tool_db_ping,
    ),
    "echo": _Tool(
        name="echo",
        description="Echo back text.",
        args_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=_tool_echo,
    ),
    "list_files": _Tool(
        name="list_files",
        description="List files under the app/data/models directories.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to list (relative to app root or absolute)."},
                "recursive": {"type": "boolean", "description": "Recurse into subdirectories."},
                "max_entries": {"type": "integer", "description": "Maximum entries to return (<=2000)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_list_files,
    ),
    "read_file": _Tool(
        name="read_file",
        description="Read a text file under app/data/models (with a size cap).",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative to app root or absolute)."},
                "max_bytes": {"type": "integer", "description": "Max bytes to return (<=200000)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_read_file,
    ),
    "write_file": _Tool(
        name="write_file",
        description="Write a file under /data (disabled by default).",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination path under /home/jarvisuser/app/data."},
                "content": {"type": "string", "description": "UTF-8 text content."},
                "append": {"type": "boolean", "description": "Append instead of overwrite."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=_tool_write_file,
    ),
    "system_info": _Tool(
        name="system_info",
        description="Get basic system/runtime info.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=_tool_system_info,
    ),
    "shell_run": _Tool(
        name="shell_run",
        description="Run a guarded shell command in the API container (disabled by default).",
        args_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {"type": "integer"},
                "confirm": {"type": "boolean"},
                "workspace_id": {"type": "integer"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=_tool_shell_run,
    ),
    "repo_write_file": _Tool(
        name="repo_write_file",
        description="Write a file under the repo root (disabled by default).",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
                "confirm": {"type": "boolean"},
                "workspace_id": {"type": "integer"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=_tool_repo_write_file,
    ),
    "browser_open": _Tool(
        name="browser_open",
        description="Open a URL in the default browser or preview the action when host control is disabled.",
        args_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}, "workspace_id": {"type": "integer"}, "confirm": {"type": "boolean"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        handler=_tool_browser_open,
    ),
    "browser_search": _Tool(
        name="browser_search",
        description="Search the configured search engine in the browser or preview the action when host control is disabled.",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "workspace_id": {"type": "integer"}, "confirm": {"type": "boolean"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_tool_browser_search,
    ),
    "desktop_launch": _Tool(
        name="desktop_launch",
        description="Launch a named desktop target like dashboard, files, terminal, or browser.",
        args_schema={
            "type": "object",
            "properties": {"app": {"type": "string"}, "workspace_id": {"type": "integer"}, "confirm": {"type": "boolean"}},
            "required": ["app"],
            "additionalProperties": False,
        },
        handler=_tool_desktop_launch,
    ),
    "desktop_control": _Tool(
        name="desktop_control",
        description="Run richer desktop actions such as typing text, sending hotkeys, opening URLs, or launching apps.",
        args_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "target": {"type": "string"},
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "workspace_id": {"type": "integer"},
                "confirm": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler=_tool_desktop_control,
    ),
    "browser_workflow": _Tool(
        name="browser_workflow",
        description="Run a real Playwright browser workflow with multi-step actions like goto, click, fill, wait, and extract.",
        args_schema={
            "type": "object",
            "properties": {
                "start_url": {"type": "string"},
                "headless": {"type": "boolean"},
                "workspace_id": {"type": "integer"},
                "confirm": {"type": "boolean"},
                "capture_storage_state": {"type": "boolean"},
                "session_name": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "selector": {"type": "string"},
                            "value": {"type": "string"},
                            "timeout_ms": {"type": "integer"},
                        },
                        "required": ["action"],
                    },
                },
            },
            "required": ["steps"],
            "additionalProperties": False,
        },
        handler=_tool_browser_workflow,
    ),
    "quantum_superposition": _Tool(
        name="quantum_superposition",
        description="Create a quantum superposition from a list of states.",
        args_schema={
            "type": "object",
            "properties": {"states": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
            "required": ["states"],
            "additionalProperties": False,
        },
        handler=_tool_quantum_superposition,
    ),
    "quantum_entangle": _Tool(
        name="quantum_entangle",
        description="Entangle two named systems.",
        args_schema={
            "type": "object",
            "properties": {"system_a": {"type": "string"}, "system_b": {"type": "string"}},
            "required": ["system_a", "system_b"],
            "additionalProperties": False,
        },
        handler=_tool_quantum_entangle,
    ),
    "quantum_measure": _Tool(
        name="quantum_measure",
        description="Measure quantum state with a basis.",
        args_schema={
            "type": "object",
            "properties": {"measurement_basis": {"type": "string"}},
            "additionalProperties": False,
        },
        handler=_tool_quantum_measure,
    ),
    "quantum_decipher": _Tool(
        name="quantum_decipher",
        description="Analyze recent quantum events and return interpreted patterns.",
        args_schema={
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "minimum": 1, "maximum": 8760},
                "event_type": {"type": "string"},
            },
            "additionalProperties": False,
        },
        handler=_tool_quantum_decipher,
    ),
    "quantum_experiment": _Tool(
        name="quantum_experiment",
        description="Run a preset quantum experiment sequence and produce a decipher snapshot.",
        args_schema={
            "type": "object",
            "properties": {
                "preset": {"type": "string", "enum": ["quick", "balanced", "deep"]},
                "measure_count": {"type": "integer", "minimum": 1, "maximum": 100},
                "entangle_count": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "additionalProperties": False,
        },
        handler=_tool_quantum_experiment,
    ),
    "quantum_remediate": _Tool(
        name="quantum_remediate",
        description="Run quantum remediation if thresholds are exceeded (or force run).",
        args_schema={
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "minimum": 1, "maximum": 8760},
                "force": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        handler=_tool_quantum_remediate,
    ),
}


def _parse_tool_message(message: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """
    Format:
      /tool <tool_name> <optional json-like args>

    Example:
      /tool echo {"text":"hello"}
    """
    text = message.strip()
    if not text.lower().startswith("/tool "):
        return None

    remainder = text[6:].strip()
    if not remainder:
        raise HTTPException(status_code=400, detail="Missing tool name")

    parts = remainder.split(" ", 1)
    tool_name = parts[0].strip()
    raw_args = parts[1].strip() if len(parts) > 1 else ""

    if not raw_args:
        return tool_name, {}

    # Keep dependencies minimal: accept strict JSON only.
    import json

    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Tool args must be valid JSON: {exc.msg}",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Tool args must be a JSON object")

    return tool_name, parsed


def _basic_brain(message: str) -> str:
    msg = message.strip().lower()
    if msg in {"hi", "hello", "hey"}:
        return "Hello. Try /tool get_time or /tool echo {\"text\":\"...\"}."
    if "help" in msg:
        return "Commands: /tool <name> <json_args>. See GET /agent/tools for available tools."
    return "I'm running. Use /tool get_time, or ask for help."


def _multi_agent_fast_model(available: List[str], fallback: str) -> str:
    for candidate in ("llama3.2:1b", "phi3:mini", "qwen2.5:1.5b"):
        if candidate in available:
            return candidate
    return fallback


def _multi_agent_workers(provider: str) -> int:
    raw = os.getenv("JARVIS_MULTI_AGENT_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 4))
        except Exception:
            pass
    # Ollama commonly serializes generation on CPU runtimes; keep modest parallelism.
    if provider == "ollama":
        configured_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "").strip()
        if configured_parallel:
            try:
                return max(1, min(int(configured_parallel), 4))
            except Exception:
                pass
        if (os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip() or "").lower() not in {"", "none"}:
            return 4
        return 2
    return 4


def _normalize_role_payload(role: str, task: str, data: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(data.get("summary") or "").strip()
    if not summary:
        summary = _basic_brain(f"{role}: {task}")

    def _clean_line(text: Any) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Unwrap quoted python/json list strings: "['a','b']" -> a | b
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    flat = [str(v).strip() for v in parsed if str(v).strip()]
                    if flat:
                        return " | ".join(flat[:3])
            except Exception:
                pass
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].strip()
        return s

    def _to_lines(value: Any, default_line: str) -> List[str]:
        if isinstance(value, list):
            out = [_clean_line(v) for v in value]
            out = [v for v in out if v]
            if out:
                return out[:6]
        if isinstance(value, str) and value.strip():
            cleaned = _clean_line(value)
            if "|" in cleaned:
                items = [x.strip() for x in cleaned.split("|") if x.strip()]
                if items:
                    return items[:6]
            return [cleaned] if cleaned else [default_line]
        return [default_line]

    return {
        "summary": _clean_line(summary),
        "plan": _to_lines(data.get("plan"), f"{role} plan pending"),
        "risks": _to_lines(data.get("risks"), "No major risks identified"),
        "actions": _to_lines(data.get("actions"), "No immediate action"),
    }


def _parse_role_tagged_text(role: str, task: str, text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    summary = ""
    plan: List[str] = []
    risks: List[str] = []
    actions: List[str] = []
    for ln in lines:
        upper = ln.upper()
        if upper.startswith("SUMMARY:"):
            summary = ln.split(":", 1)[1].strip()
        elif upper.startswith("PLAN:"):
            plan = [x.strip() for x in ln.split(":", 1)[1].split("|") if x.strip()][:3]
        elif upper.startswith("RISKS:"):
            risks = [x.strip() for x in ln.split(":", 1)[1].split("|") if x.strip()][:3]
        elif upper.startswith("ACTIONS:"):
            actions = [x.strip() for x in ln.split(":", 1)[1].split("|") if x.strip()][:3]
    return _normalize_role_payload(
        role,
        task,
        {"summary": summary, "plan": plan, "risks": risks, "actions": actions},
    )


def _multi_agent_role_models(runtime: Dict[str, Any], force_fast: bool = True) -> List[str]:
    available = [str(m).strip() for m in (runtime.get("available_models") or []) if str(m).strip()]
    selected = str(runtime.get("model") or ollama_model()).strip()
    if force_fast:
        preferred = ["llama3.2:1b", "phi3:mini", "qwen2.5:1.5b", selected]
    else:
        preferred = [selected, "llama3.2:3b", "llama3.2:1b", "phi3:mini"]
    dedup: List[str] = []
    for m in preferred:
        if m and m in available and m not in dedup:
            dedup.append(m)
    if selected and selected not in dedup:
        dedup.append(selected)
    if not dedup:
        dedup.append(ollama_model())
    return dedup


def _parse_role_output(role: str, task: str, raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()
    if not txt:
        return _normalize_role_payload(role, task, {})
    if txt.startswith("```"):
        txt = txt.strip("`").strip()
        if txt.lower().startswith("json"):
            txt = txt[4:].strip()
    try:
        maybe = json.loads(txt)
        if isinstance(maybe, dict):
            return _normalize_role_payload(role, task, maybe)
    except Exception:
        pass
    tagged = _parse_role_tagged_text(role, task, txt)
    if str(tagged.get("summary") or "").strip():
        return tagged
    return _normalize_role_payload(role, task, {"summary": txt})


def _agent_role_reason(role: str, task: str, session_id: str, force_fast: bool = True) -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if is_openai_configured():
            provider = "openai"
        else:
            provider = "basic"
    if provider != "ollama":
        return _normalize_role_payload(
            role,
            task,
            {
                "summary": _basic_brain(f"{role}: {task}"),
                "plan": ["Understand task", "Select relevant tools", "Execute safely"],
                "risks": ["LLM provider not configured for rich role output"],
                "actions": ["Set LLM_PROVIDER=ollama for richer multi-agent output"],
            },
        )

    runtime = _effective_ollama_runtime()
    models = _multi_agent_role_models(runtime=runtime, force_fast=force_fast)
    prompt = (
        f"You are the {role} specialist in Jarvis multi-agent mode.\n"
        "Return strict JSON with keys summary, plan, risks, actions.\n"
        "summary must be one short sentence.\n"
        "plan/risks/actions must be JSON arrays of plain strings (no nested arrays, no markdown, no brackets in strings).\n"
        "Each array may contain up to 3 concise items.\n"
        f"Task: {task}\n"
    )
    base_timeout = max(8, min(int(os.getenv("JARVIS_MULTI_AGENT_ROLE_TIMEOUT_S", "40")), 180))
    attempts: List[Tuple[str, int]] = []
    errors: List[str] = []
    for model_name in models:
        for timeout_s in (base_timeout, min(base_timeout + 20, 180)):
            attempts.append((model_name, timeout_s))
            try:
                raw = ollama_chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system="Be concise and practical. Output JSON only.",
                    temperature=0.2 if force_fast else 0.1,
                    format_json=True,
                    options={"num_predict": 140},
                    timeout_s=timeout_s,
                )
                payload = _parse_role_output(role, task, raw)
                payload["model"] = model_name
                payload["attempts"] = len(attempts)
                return payload
            except Exception as exc:
                errors.append(f"{model_name}@{timeout_s}s: {exc}")

    return _normalize_role_payload(
        role,
        task,
        {
            "summary": f"{role} reasoning unavailable: all role-model attempts failed.",
            "plan": ["Retry multi-agent run", "Switch to fast mode or smaller model"],
            "risks": errors[:3] or ["No role model response"],
            "actions": ["Check Ollama service health and model load"],
        },
    ) | {"model": "unavailable", "attempts": len(attempts)}


def _token_estimate(text: str) -> int:
    return max(1, int(len((text or "").strip()) / 4))


def _score_multi_agent_run(result: Dict[str, Any], latency_ms: int) -> Dict[str, Any]:
    agents = result.get("agents") if isinstance(result.get("agents"), dict) else {}
    role_names = ["planner", "researcher", "coder", "operator"]
    structure_points = 0
    max_points = len(role_names) * 4
    actions_flat: List[str] = []
    for rn in role_names:
        role_data = agents.get(rn)
        if not isinstance(role_data, dict):
            continue
        if str(role_data.get("summary") or "").strip():
            structure_points += 1
        for key in ("plan", "risks", "actions"):
            values = role_data.get(key)
            if isinstance(values, list) and any(str(v).strip() for v in values):
                structure_points += 1
        for v in (role_data.get("actions") or []):
            if isinstance(v, str) and v.strip():
                actions_flat.append(v.strip().lower())

    structure_score = (structure_points / max_points) if max_points else 0.0
    unique_actions = len(set(actions_flat))
    actionability_score = min(1.0, unique_actions / 6.0)
    latency_score = max(0.0, 1.0 - (latency_ms / 120000.0))
    overall = (0.55 * structure_score) + (0.25 * actionability_score) + (0.20 * latency_score)
    return {
        "overall_score_pct": round(overall * 100.0, 1),
        "structure_score_pct": round(structure_score * 100.0, 1),
        "actionability_score_pct": round(actionability_score * 100.0, 1),
        "latency_score_pct": round(latency_score * 100.0, 1),
        "latency_ms": latency_ms,
        "unique_actions": unique_actions,
    }


def _memory_quality_report(session_id: str, max_messages: int = 120) -> Dict[str, Any]:
    msgs = _memory.load(session_id, max_messages=max_messages)
    if not msgs:
        return {
            "session_id": session_id,
            "messages": 0,
            "overall_score_pct": 0.0,
            "notes": ["No messages found for this session."],
        }
    user_msgs = [m.text for m in msgs if m.role == "user"]
    assistant_msgs = [m.text for m in msgs if m.role == "assistant"]
    turn_balance = min(1.0, len(assistant_msgs) / max(1, len(user_msgs)))
    avg_user_len = sum(len(t) for t in user_msgs) / max(1, len(user_msgs))
    avg_assistant_len = sum(len(t) for t in assistant_msgs) / max(1, len(assistant_msgs))
    user_unique_ratio = len(set(t.strip().lower() for t in user_msgs if t.strip())) / max(1, len(user_msgs))
    assistant_unique_ratio = len(set(t.strip().lower() for t in assistant_msgs if t.strip())) / max(1, len(assistant_msgs))

    # Lightweight coherence signal: assistant response length should not collapse.
    coherence = min(1.0, avg_assistant_len / max(80.0, avg_user_len))
    coherence = max(0.0, coherence)
    repetition = (user_unique_ratio * 0.5) + (assistant_unique_ratio * 0.5)
    overall = (0.4 * turn_balance) + (0.35 * coherence) + (0.25 * repetition)

    return {
        "session_id": session_id,
        "messages": len(msgs),
        "user_messages": len(user_msgs),
        "assistant_messages": len(assistant_msgs),
        "turn_balance_score_pct": round(turn_balance * 100.0, 1),
        "coherence_score_pct": round(coherence * 100.0, 1),
        "repetition_score_pct": round(repetition * 100.0, 1),
        "overall_score_pct": round(overall * 100.0, 1),
        "avg_user_chars": round(avg_user_len, 1),
        "avg_assistant_chars": round(avg_assistant_len, 1),
    }


def _extract_memory_candidate(message: str) -> Optional[Dict[str, Any]]:
    raw = (message or "").strip()
    lower = raw.lower()
    if lower.startswith("remember that "):
        return {
            "content": raw[14:].strip(),
            "tags": ["remembered"],
            "importance": 4,
            "memory_type": "general",
            "source": "chat",
        }
    if lower.startswith("remember "):
        return {
            "content": raw[9:].strip(),
            "tags": ["remembered"],
            "importance": 4,
            "memory_type": "general",
            "source": "chat",
        }
    if lower.startswith("my name is "):
        return {
            "content": raw,
            "tags": ["profile", "identity"],
            "importance": 5,
            "memory_type": "identity",
            "subject": "user_name",
            "source": "chat",
        }
    if lower.startswith("i prefer ") or lower.startswith("i like "):
        return {
            "content": raw,
            "tags": ["preference"],
            "importance": 4,
            "memory_type": "preference",
            "source": "chat",
        }
    if "project" in lower and any(k in lower for k in ["working on", "building", "my project"]):
        return {
            "content": raw,
            "tags": ["project"],
            "importance": 4,
            "memory_type": "project",
            "subject": "active_project",
            "source": "chat",
        }
    return None


def _memory_context_for_prompt(message: str, limit: int = 4) -> str:
    bundle = _task_memory.memory_context_bundle(query=message, limit=max(4, limit))
    matches = bundle.get("items") or []
    active_workspace_id = _task_memory.get_active_workspace_id()
    workspace = _task_memory.get_project_workspace(active_workspace_id) if active_workspace_id else None
    if not matches and not workspace:
        return ""
    lines = []
    for item in matches:
        tags = item.get("tags") or []
        tag_txt = f" [{', '.join(tags[:3])}]" if tags else ""
        type_txt = str(item.get("memory_type") or "general")
        score_txt = item.get("effective_importance")
        score_label = f" (score {score_txt})" if isinstance(score_txt, (int, float)) else ""
        lines.append(f"- ({type_txt}) {item.get('content')}{tag_txt}{score_label}")
    profile = bundle.get("profile") or []
    projects = bundle.get("projects") or []
    sections: List[str] = []
    if profile:
        sections.append("User profile memory:\n" + "\n".join(lines[: min(3, len(lines))]))
    if projects:
        proj_lines = []
        for item in projects[:2]:
            proj_lines.append(f"- {item.get('content')}")
        sections.append("Active project memory:\n" + "\n".join(proj_lines))
    if workspace:
        ws_lines = [
            f"Workspace: {workspace.get('name')}",
            f"Focus: {workspace.get('focus') or 'n/a'}",
        ]
        for item in list(workspace.get("memories") or [])[:3]:
            ws_lines.append(f"- {item.get('content')}")
        sections.append("Active workspace context:\n" + "\n".join(ws_lines))
    sections.append("Relevant memory matches:\n" + "\n".join(lines[:limit]))
    return "\n\n".join([s for s in sections if s.strip()])


def _format_memory_group(title: str, items: List[Dict[str, Any]], limit: int = 6) -> str:
    if not items:
        return f"{title}: none saved yet."
    lines = [title + ":"]
    for item in items[:limit]:
        type_txt = str(item.get("memory_type") or "general")
        score_txt = item.get("effective_importance")
        score_label = f" (score {score_txt})" if isinstance(score_txt, (int, float)) else ""
        lines.append(f"- ({type_txt}) {item.get('content')}{score_label}")
    return "\n".join(lines)

def _ocr_from_image(img: Image.Image) -> Dict[str, Any]:
    if pytesseract is None:
        return {
            "text": "",
            "confidence": None,
            "engine": "unavailable",
            "boxes": [],
            "error": "pytesseract is not installed",
        }
    try:
        gray = img.convert("L")
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        texts: List[str] = []
        confs: List[float] = []
        boxes: List[Dict[str, Any]] = []
        for idx, (text, conf) in enumerate(zip(data.get("text", []), data.get("conf", []))):
            t = str(text or "").strip()
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if not t:
                continue
            texts.append(t)
            if c >= 0:
                confs.append(c)
            lefts = data.get("left", [])
            tops = data.get("top", [])
            widths = data.get("width", [])
            heights = data.get("height", [])
            left = int(lefts[idx]) if idx < len(lefts) else 0
            top = int(tops[idx]) if idx < len(tops) else 0
            width = int(widths[idx]) if idx < len(widths) else 0
            height = int(heights[idx]) if idx < len(heights) else 0
            if width > 0 and height > 0:
                boxes.append(
                    {
                        "text": t,
                        "confidence": c if c >= 0 else None,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                    }
                )
        combined = " ".join(texts).strip()
        confidence = round(sum(confs) / len(confs), 1) if confs else None
        return {"text": combined, "confidence": confidence, "engine": "tesseract", "boxes": boxes}
    except Exception as exc:
        return {"text": "", "confidence": None, "engine": "tesseract", "boxes": [], "error": str(exc)}


def _pick_ollama_vision_model() -> Optional[str]:
    override = _vision_model_override()
    if override:
        return override
    configured = os.getenv("OLLAMA_VISION_MODEL", "").strip()
    if configured:
        return configured
    models = [m.strip() for m in ollama_list_models() if isinstance(m, str)]
    preferred_markers = ("llava", "moondream", "bakllava", "gemma3", "minicpm-v")
    for marker in preferred_markers:
        for model in models:
            if marker in model.lower():
                return model
    return None


def _multimodal_vision_reasoning(data: bytes, *, ocr_text: str = "", prompt: Optional[str] = None) -> Dict[str, Any]:
    prompt_text = (prompt or "").strip() or (
        "Analyze this screenshot or image for Jarvis. "
        "Describe what is visible, the likely context, and any actionable signals in 3 short sentences."
    )
    if ocr_text.strip():
        prompt_text += f"\nOCR text detected:\n{ocr_text[:1200]}"

    runtime = _effective_vision_runtime()
    provider = str(runtime.get("provider") or "auto")
    if provider == "auto":
        if is_openai_configured():
            provider = "openai"
        elif is_ollama_configured():
            provider = "ollama"
        else:
            provider = "heuristic"
    if provider == "heuristic":
        return {"summary": "", "provider": "heuristic", "model": None}

    if provider == "openai" and is_openai_configured():
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = str(runtime.get("model") or os.getenv("OPENAI_VISION_MODEL", "").strip() or os.getenv("OPENAI_MODEL", "gpt-4.1"))
        try:
            text = openai_vision_analyze(
                api_key=api_key,
                model=model,
                prompt=prompt_text,
                image_bytes=data,
                timeout_s=int(os.getenv("OPENAI_VISION_TIMEOUT_S", "90")),
            )
            if text.strip():
                return {"summary": text.strip(), "provider": "openai", "model": model}
        except Exception as exc:
            return {"summary": "", "provider": "openai", "model": model, "error": str(exc)}

    if is_ollama_configured():
        model = str(runtime.get("model") or _pick_ollama_vision_model() or "")
        if model:
            try:
                text = ollama_chat_with_images(
                    model=model,
                    prompt=prompt_text,
                    image_bytes=[data],
                    system="You are Jarvis visual intelligence. Be concrete, concise, and action-oriented.",
                    temperature=0.1,
                    timeout_s=int(os.getenv("OLLAMA_VISION_TIMEOUT_S", "120")),
                )
                if text.strip():
                    return {"summary": text.strip(), "provider": "ollama", "model": model}
            except Exception as exc:
                return {"summary": "", "provider": "ollama", "model": model, "error": str(exc)}

    return {"summary": "", "provider": provider or "none", "model": runtime.get("model")}


def _vision_summary_from_image(data: bytes, prompt: Optional[str] = None) -> Dict[str, Any]:
    img = Image.open(BytesIO(data))
    img = img.convert("RGB")
    width, height = img.size
    stat = ImageStat.Stat(img)
    avg_rgb = [round(v, 1) for v in stat.mean[:3]]
    brightness = round(sum(avg_rgb) / 3.0, 1)
    palette_img = img.resize((96, 96))
    quantized = palette_img.quantize(colors=4)
    palette = quantized.getpalette() or []
    color_counts = quantized.getcolors() or []
    top_colors: List[str] = []
    for count, color_index in sorted(color_counts, reverse=True)[:3]:
        base = int(color_index) * 3
        if base + 2 < len(palette):
            rgb = palette[base: base + 3]
            top_colors.append("#%02x%02x%02x" % (rgb[0], rgb[1], rgb[2]))

    orientation = "landscape" if width >= height else "portrait"
    mood = "dark" if brightness < 90 else ("balanced" if brightness < 180 else "bright")
    heuristic_summary = (
        f"Captured {orientation} image at {width}x{height}. "
        f"Overall scene appears {mood} with dominant colors {', '.join(top_colors) or 'unavailable'}."
    )
    ocr = _ocr_from_image(img)
    multimodal = _multimodal_vision_reasoning(data, ocr_text=str(ocr.get("text") or ""), prompt=prompt)
    summary_parts = [heuristic_summary]
    if str(ocr.get("text") or "").strip():
        preview = str(ocr["text"]).strip().replace("\n", " ")
        summary_parts.append(f"Visible text: {preview[:180]}")
    if str(multimodal.get("summary") or "").strip():
        summary_parts.append(str(multimodal["summary"]).strip())
    summary = " ".join(summary_parts).strip()
    details = {
        "width": width,
        "height": height,
        "orientation": orientation,
        "brightness": brightness,
        "avg_rgb": avg_rgb,
        "dominant_colors": top_colors,
        "ocr_text": str(ocr.get("text") or ""),
        "ocr_confidence": ocr.get("confidence"),
        "ocr_engine": ocr.get("engine"),
        "ocr_boxes": ocr.get("boxes") or [],
        "ocr_error": ocr.get("error"),
        "multimodal_summary": str(multimodal.get("summary") or ""),
        "multimodal_provider": multimodal.get("provider"),
        "multimodal_model": multimodal.get("model"),
        "multimodal_error": multimodal.get("error"),
        "vision_mode": "multimodal+ocr" if str(multimodal.get("summary") or "").strip() else "heuristic+ocr",
    }
    return {"summary": summary, "details": details}


def _watcher_result_base(job: Dict[str, Any], watcher_type: str, workspace_id: Optional[int], limit: int, min_score: float) -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": "watcher",
        "watcher_type": watcher_type,
        "workspace_id": workspace_id,
        "limit": limit,
        "min_score": min_score,
        "job_name": job.get("name"),
        "timestamp": _now_iso(),
    }


def _watcher_trigger_level(triggers: List[Dict[str, Any]], *, min_score: float) -> str:
    top = max([float(item.get("score") or 0.0) for item in triggers] or [0.0])
    if top >= max(min_score + 2.0, 9.0):
        return "critical"
    if top >= max(min_score + 0.8, 7.5):
        return "high"
    return "elevated"


def _run_project_watcher(job: Dict[str, Any]) -> Dict[str, Any]:
    metadata = job.get("metadata") or {}
    watcher_type = str(metadata.get("watcher_type") or "project").strip().lower() or "project"
    workspace_id = metadata.get("workspace_id") or _task_memory.get_active_workspace_id()
    limit = max(1, min(int(metadata.get("limit") or 3), 6))
    min_score = float(metadata.get("min_score") or 6.0)
    base = _watcher_result_base(job, watcher_type, workspace_id, limit, min_score)

    if watcher_type == "project":
        next_actions = _task_memory.next_best_actions(workspace_id=workspace_id, limit=max(limit + 2, 5))
        actions = list(next_actions.get("actions") or [])
        triggers = [item for item in actions if float(item.get("score") or 0.0) >= min_score]
        generated_created: List[Dict[str, Any]] = []
        if not triggers:
            generated = _task_memory.generate_proactive_reminders(workspace_id=workspace_id, limit=3)
            next_actions = _task_memory.next_best_actions(workspace_id=workspace_id, limit=max(limit + 2, 5))
            actions = list(next_actions.get("actions") or [])
            triggers = [item for item in actions if float(item.get("score") or 0.0) >= min_score]
            generated_created = generated.get("created") or []
        if not triggers:
            return base | {
                "triggered": False,
                "summary": "Project watcher scanned the workspace and found no actions above the trigger threshold.",
                "next_actions": actions[:limit],
                "generated_reminders": generated_created,
                "escalation": "quiet",
            }
        mission = _run_autonomous_mission(
            workspace_id=int(workspace_id) if workspace_id else None,
            session_id=job.get("session_id"),
            limit=limit,
            auto_approve=bool(job.get("auto_approve")),
        )
        return base | {
            "triggered": True,
            "summary": f"Project watcher triggered mission mode from {len(triggers)} high-priority actions.",
            "triggers": triggers[:limit],
            "next_actions": actions[:limit],
            "generated_reminders": generated_created,
            "mission": mission,
            "escalation": _watcher_trigger_level(triggers, min_score=min_score),
        }

    if watcher_type == "calendar":
        now = datetime.now(timezone.utc)
        upcoming = []
        for event in _calendar_events_get():
            starts = _parse_iso_dt(event.get("starts_at") if isinstance(event, dict) else None)
            if starts is None:
                continue
            if starts.tzinfo is None:
                starts = starts.replace(tzinfo=timezone.utc)
            hours = (starts - now).total_seconds() / 3600.0
            if -0.5 <= hours <= 6.0:
                enriched = dict(event)
                enriched["hours_until"] = round(hours, 2)
                upcoming.append(enriched)
        upcoming.sort(key=lambda item: float(item.get("hours_until") or 999))
        triggers = [item for item in upcoming if float(item.get("hours_until") or 999) <= 2.0]
        if not triggers:
            return base | {
                "triggered": False,
                "summary": "Calendar watcher scanned upcoming events and found no urgent schedule pressure.",
                "calendar_events": upcoming[:limit],
                "escalation": "quiet",
            }
        mission = _run_autonomous_mission(
            workspace_id=int(workspace_id) if workspace_id else None,
            session_id=job.get("session_id"),
            limit=min(limit, 3),
            auto_approve=bool(job.get("auto_approve")),
        )
        return base | {
            "triggered": True,
            "summary": f"Calendar watcher found {len(triggers)} event(s) within the next two hours.",
            "calendar_events": upcoming[:limit],
            "triggers": triggers[:limit],
            "mission": mission,
            "escalation": "high",
        }

    if watcher_type == "email":
        reminders = _task_memory.list_proactive_reminders(limit=12, status="open", workspace_id=workspace_id, due_within_hours=48)
        triggers = [item for item in reminders if str(item.get("priority") or "").lower() in {"critical", "high"}]
        if not triggers:
            return base | {
                "triggered": False,
                "summary": "Email watcher found no high-priority reminder or inbox pressure.",
                "reminders": reminders[:limit],
                "escalation": "quiet",
            }
        briefing = _task_memory.memory_briefing(period="now", recent_project_hours=24)
        delivery = _dispatch_briefing_deliveries(briefing)
        return base | {
            "triggered": True,
            "summary": f"Email watcher found {len(triggers)} high-priority reminder(s) and prepared a briefing payload.",
            "triggers": triggers[:limit],
            "briefing": briefing,
            "delivery": delivery,
            "escalation": "high",
        }

    if watcher_type == "desktop":
        awareness = _desktop_awareness_payload(workspace_id=workspace_id)
        next_actions = _task_memory.next_best_actions(workspace_id=workspace_id, limit=max(limit, 4))
        actions = list(next_actions.get("actions") or [])
        focus_mode = str(awareness.get("focus_mode") or "standby")
        triggers = [item for item in actions if float(item.get("score") or 0.0) >= min_score]
        if focus_mode not in {"coding", "communication", "meeting"} and not triggers:
            return base | {
                "triggered": False,
                "summary": "Desktop watcher found no elevated focus context or urgent action.",
                "awareness": awareness,
                "next_actions": actions[:limit],
                "escalation": "quiet",
            }
        return base | {
            "triggered": True,
            "summary": f"Desktop watcher detected {focus_mode} context with {len(triggers)} elevated next action(s).",
            "awareness": awareness,
            "triggers": triggers[:limit],
            "next_actions": actions[:limit],
            "escalation": _watcher_trigger_level(triggers or [{"score": min_score + 1}], min_score=min_score),
        }

    if watcher_type == "github":
        intel = _integration_intelligence()
        next_actions = _task_memory.next_best_actions(workspace_id=workspace_id, limit=max(limit, 4))
        actions = list(next_actions.get("actions") or [])
        github_recos = [item for item in list(intel.get("recommendations") or []) if str(item.get("channel") or "") == "github"]
        triggers = [item for item in actions if float(item.get("score") or 0.0) >= min_score and str(item.get("source") or "") in {"trust", "project", "policy"}]
        if not github_recos and not triggers:
            return base | {
                "triggered": False,
                "summary": "GitHub watcher found no repo pressure or elevated project risk.",
                "recommendations": github_recos,
                "next_actions": actions[:limit],
                "escalation": "quiet",
            }
        return base | {
            "triggered": True,
            "summary": "GitHub watcher sees repo follow-up that should be handled in the next cycle.",
            "recommendations": github_recos,
            "triggers": triggers[:limit],
            "next_actions": actions[:limit],
            "escalation": _watcher_trigger_level(triggers or [{"score": min_score + 0.5}], min_score=min_score),
        }

    return base | {
        "triggered": False,
        "summary": f"Watcher type {watcher_type} is not yet specialized.",
        "escalation": "quiet",
    }


def _execute_autonomous_job(job: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(job.get("mode") or "multi_agent")
    goal = str(job.get("goal") or "").strip()
    session_id = job.get("session_id")
    if mode == "goal":
        return _execute_goal_run(goal=goal, session_id=session_id, auto_approve=bool(job.get("auto_approve")))
    if mode == "briefing":
        metadata = job.get("metadata") or {}
        period = str(metadata.get("period") or "morning")
        recent_project_hours = metadata.get("recent_project_hours")
        briefing = _task_memory.memory_briefing(period=period, recent_project_hours=recent_project_hours)
        reminders = _task_memory.generate_proactive_reminders(
            workspace_id=_task_memory.get_active_workspace_id(),
            limit=3,
        )
        delivery = _dispatch_briefing_deliveries(briefing)
        return {
            "ok": True,
            "mode": "briefing",
            "name": job.get("name"),
            "goal": goal,
            "session_id": session_id,
            "briefing": briefing,
            "reminders": reminders.get("created") or [],
            "text": briefing.get("text"),
            "delivery": delivery,
            "timestamp": _now_iso(),
        }
    if mode == "watcher":
        return _run_project_watcher(job)
    return run_multi_agent(MultiAgentRunRequest(task=goal, session_id=session_id, fast_synthesis=True))


def _operator_from_roles(task: str, planner: Dict[str, Any], coder: Dict[str, Any], researcher: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    actions: List[str] = []
    for role_data in [planner, coder, researcher or {}]:
        for a in (role_data.get("actions") or []):
            if isinstance(a, str) and a.strip() and a.strip() not in actions:
                actions.append(a.strip())
    plan = [
        "Prioritize actions by risk and dependency",
        "Assign owners and expected completion windows",
        "Execute changes and verify post-change metrics",
    ]
    return _normalize_role_payload(
        "operator",
        task,
        {
            "summary": "Convert role outputs into an execution-ready operational plan.",
            "plan": plan,
            "actions": actions[:3] or ["Create an execution checklist and start with the highest-risk fix"],
        },
    ) | {"model": "synthesized"}


def _needs_deep_synthesis(task: str, planner: Dict[str, Any], coder: Dict[str, Any]) -> bool:
    t = (task or "").lower()
    deep_markers = [
        "research",
        "compare",
        "benchmark",
        "architecture",
        "compliance",
        "security",
        "incident",
        "audit",
        "policy",
        "migration",
    ]
    if any(k in t for k in deep_markers):
        return True

    # Expand if the first-pass output quality appears weak.
    for role_data in (planner, coder):
        if not isinstance(role_data, dict):
            return True
        if str(role_data.get("summary") or "").strip() == "":
            return True
        if str(role_data.get("model") or "").strip() in {"unavailable", ""}:
            return True
        attempts = role_data.get("attempts")
        if isinstance(attempts, int) and attempts > 2:
            return True

    return False


@router.get("/tools", response_model=List[ToolSpec])
def list_tools():
    return [
        ToolSpec(name=t.name, description=t.description, args_schema=t.args_schema)
        for t in sorted(_TOOLS.values(), key=lambda x: x.name)
    ]


@router.get("/profile", response_model=AgentProfileResponse)
def get_profile():
    return AgentProfileResponse(
        profile=_current_profile(),
        dangerous_full_access=_dangerous_full_access_enabled(),
    )


@router.post("/profile", response_model=AgentProfileResponse)
def set_profile(payload: AgentProfileUpdateRequest):
    profile = _set_profile(payload.profile)
    return AgentProfileResponse(
        profile=profile,
        dangerous_full_access=_dangerous_full_access_enabled(),
    )


@router.get("/llm/config")
def get_llm_config():
    provider = (os.getenv("LLM_PROVIDER", "").strip().lower() or "auto")
    ollama_runtime = _effective_ollama_runtime()
    return {
        "provider": provider,
        "mode": ollama_runtime["mode"],
        "model": ollama_runtime["model"],
        "available_models": ollama_runtime["available_models"],
        "temperature": ollama_runtime["temperature"],
        "streaming_supported": True,
    }


@router.post("/llm/config")
def set_llm_config(payload: AgentLlmConfigRequest):
    mode = _set_llm_mode(payload.mode)
    model = _set_llm_model_override(payload.model)
    runtime = _effective_ollama_runtime()
    _audit_quantum_op("llm_config_set", "ok", {"mode": mode, "model": model or runtime.get("model")})
    return {
        "ok": True,
        "mode": mode,
        "model": runtime.get("model"),
        "available_models": runtime.get("available_models") or [],
        "temperature": runtime.get("temperature"),
    }


@router.get("/vision/config")
def get_vision_config():
    return _effective_vision_runtime()


@router.post("/vision/config")
def set_vision_config(payload: AgentVisionConfigRequest):
    provider = _set_vision_provider(payload.provider)
    model = _set_vision_model_override(payload.model)
    runtime = _effective_vision_runtime()
    _audit_quantum_op("vision_config_set", "ok", {"provider": provider, "model": model or runtime.get("model")})
    return {
        "ok": True,
        "provider": runtime.get("provider"),
        "model": runtime.get("model"),
        "available_models": runtime.get("available_models") or [],
        "ocr_available": runtime.get("ocr_available"),
        "openai_configured": runtime.get("openai_configured"),
    }


@router.get("/policy")
def get_agent_policy():
    return _get_agent_policy()


@router.post("/policy")
def set_agent_policy(payload: AgentPolicyConfigRequest):
    cfg = _set_agent_policy(payload.model_dump())
    return {"ok": True, **cfg}


@router.post("/activate_advanced")
def activate_advanced_features():
    runtime = _effective_ollama_runtime()
    _set_llm_mode("quality")
    if runtime.get("model"):
        _set_llm_model_override(str(runtime.get("model")))
    policy = _set_agent_policy({"strict_confirm": True})

    profile = _current_profile()
    if _dangerous_full_access_enabled():
        profile = _set_profile("full")

    existing = _task_memory.list_goal_schedules(limit=500)
    existing_goals = {str(s.get("goal") or "").strip().lower() for s in existing}
    created: List[Dict[str, Any]] = []
    defaults = [
        {"goal": "quantum decipher 24h", "interval_minutes": 15},
        {"goal": "quantum health score 24h", "interval_minutes": 15},
        {"goal": "quantum remediation 24h", "interval_minutes": 60},
    ]
    for d in defaults:
        g = str(d["goal"]).strip().lower()
        if g in existing_goals:
            continue
        sid = _task_memory.create_goal_schedule(
            goal=str(d["goal"]),
            interval_minutes=int(d["interval_minutes"]),
            session_id=None,
            auto_approve=True,
            enabled=True,
        )
        created.append({"id": sid, "goal": d["goal"], "interval_minutes": d["interval_minutes"]})

    _audit_quantum_op(
        "activate_advanced",
        "ok",
        {
            "profile": profile,
            "mode": "quality",
            "model": runtime.get("model"),
            "schedules_created": len(created),
        },
    )
    return {
        "ok": True,
        "profile": profile,
        "llm": {"mode": "quality", "model": runtime.get("model"), "temperature": 0.2},
        "policy": policy,
        "schedules_created": created,
    }


@router.get("/tasks")
def list_tasks(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    tasks = _task_memory.list_tasks(session_id=session_id, status=status, limit=limit)
    return {
        "tasks": tasks,
        "count": len(tasks),
    }


@router.post("/tasks")
def create_task(payload: AgentTaskCreateRequest):
    task_id = _task_memory.create_task(payload.task, session_id=payload.session_id)
    return {"id": task_id, "status": "open"}


@router.post("/tasks/{task_id}/status")
def update_task_status(task_id: int, payload: AgentTaskStatusRequest):
    ok = _task_memory.update_task_status(task_id, payload.status, note=payload.note)
    if not ok:
        raise HTTPException(status_code=404, detail="task not found")
    return {"id": task_id, "status": payload.status, "ok": True}


@router.post("/self_test")
def self_test():
    tests: List[Dict[str, Any]] = []

    def _record(name: str, fn: Callable[[], Any]) -> None:
        try:
            result = fn()
            tests.append({"name": name, "ok": True, "result": result})
        except Exception as exc:
            tests.append({"name": name, "ok": False, "error": str(exc)})

    _record("get_time", lambda: _TOOLS["get_time"].handler({}))
    _record("list_files", lambda: _TOOLS["list_files"].handler({"path": "data"}))
    _record("system_info", lambda: _TOOLS["system_info"].handler({}))

    if _is_truthy(os.getenv("JARVIS_ALLOW_SHELL", "")):
        _record("shell_run", lambda: _TOOLS["shell_run"].handler({"command": "pwd"}))
    if _is_truthy(os.getenv("JARVIS_ALLOW_WRITE", "")):
        _record(
            "write_file",
            lambda: _TOOLS["write_file"].handler({"path": "data/self_test.txt", "content": "ok"}),
        )
    if _is_truthy(os.getenv("JARVIS_ALLOW_REPO_WRITE", "")):
        _record(
            "repo_write_file",
            lambda: _TOOLS["repo_write_file"].handler({"path": "scratch/self_test_repo.txt", "content": "ok"}),
        )

    passed = sum(1 for t in tests if t.get("ok"))
    return {"ok": passed == len(tests), "passed": passed, "total": len(tests), "tests": tests}


@router.get("/quantum/status")
def quantum_status():
    available = True
    error = None
    creator_authorized = False
    operations_performed = 0
    try:
        qp = _get_quantum_processor()
        creator_authorized = bool(getattr(qp, "creator_authorized", False))
        operations_performed = int(getattr(qp, "operations_performed", 0))
    except HTTPException as exc:
        available = False
        error = str(exc.detail)
    return {
        "available": available,
        "creator_authorized": creator_authorized,
        "operations_performed": operations_performed,
        "error": error,
    }


@router.post("/quantum/superposition")
def quantum_superposition(payload: QuantumSuperpositionRequest):
    return _tool_quantum_superposition({"states": payload.states})


@router.post("/quantum/entangle")
def quantum_entangle(payload: QuantumEntangleRequest):
    return _tool_quantum_entangle({"system_a": payload.system_a, "system_b": payload.system_b})


@router.post("/quantum/measure")
def quantum_measure(payload: QuantumMeasureRequest):
    return _tool_quantum_measure({"measurement_basis": payload.measurement_basis})


@router.get("/quantum/history")
def quantum_history(
    limit: int = Query(default=100, ge=1, le=500),
    event_type: Optional[str] = None,
    since_hours: Optional[int] = Query(default=None, ge=1, le=24 * 365),
):
    return {
        "events": _task_memory.list_quantum_events(
            limit=limit,
            event_type=event_type,
            since_hours=since_hours,
        )
    }


@router.get("/quantum/stats")
def quantum_stats(hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _task_memory.quantum_stats(hours=hours)


@router.get("/quantum/alerts")
def quantum_alerts():
    cfg = _get_quantum_alert_config()
    evaluation = _evaluate_quantum_alerts(cfg)
    _sync_incidents_from_alerts(evaluation["alerts"])
    return {"config": cfg, "active": evaluation["active"], "alerts": evaluation["alerts"], "stats": evaluation["stats"]}


@router.post("/quantum/alerts/config")
def quantum_alerts_config(payload: QuantumAlertConfigRequest):
    cfg = _set_quantum_alert_config(payload.model_dump())
    evaluation = _evaluate_quantum_alerts(cfg)
    return {"config": cfg, "active": evaluation["active"], "alerts": evaluation["alerts"]}


def _to_csv(rows: List[Dict[str, Any]], *, columns: List[str]) -> str:
    def esc(value: Any) -> str:
        s = "" if value is None else str(value)
        s = s.replace('"', '""')
        if any(ch in s for ch in [",", "\n", '"']):
            return f'"{s}"'
        return s

    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(esc(r.get(c)) for c in columns))
    return "\n".join(lines) + "\n"


def _parse_iso_datetime(value: Optional[str], *, field_name: str) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}; expected ISO-8601 datetime") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_created_at(event: Dict[str, Any]) -> Optional[datetime]:
    created_at = event.get("created_at")
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _filter_events_by_time_window(
    events: List[Dict[str, Any]],
    *,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
) -> List[Dict[str, Any]]:
    if start_at is None and end_at is None:
        return events
    filtered: List[Dict[str, Any]] = []
    for event in events:
        event_dt = _event_created_at(event)
        if event_dt is None:
            continue
        if start_at is not None and event_dt < start_at:
            continue
        if end_at is not None and event_dt > end_at:
            continue
        filtered.append(event)
    return filtered


def _severity_rank(level: str) -> int:
    order = {"info": 0, "warning": 1, "critical": 2}
    return order.get(str(level or "warning").lower(), 1)


def _bucket_start(dt: datetime, bucket_minutes: int) -> datetime:
    minute = (dt.minute // bucket_minutes) * bucket_minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def _quantum_timeline(events: List[Dict[str, Any]], *, hours: int, bucket_minutes: int) -> Dict[str, Any]:
    bm = max(1, min(int(bucket_minutes), 240))
    buckets: Dict[str, Dict[str, Any]] = {}
    for e in events:
        dt = _event_created_at(e)
        if dt is None:
            continue
        key = _bucket_start(dt, bm).isoformat()
        row = buckets.setdefault(
            key,
            {"ts": key, "events": 0, "measurements": 0, "entangles": 0, "superpositions": 0},
        )
        row["events"] += 1
        et = str(e.get("event_type") or "")
        if et == "measurement":
            row["measurements"] += 1
        elif et == "entangle":
            row["entangles"] += 1
        elif et == "superposition":
            row["superpositions"] += 1
    points = [buckets[k] for k in sorted(buckets.keys())]
    return {"window_hours": hours, "bucket_minutes": bm, "points": points}


def _compute_z_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std <= 1e-9:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _quantum_anomalies(events: List[Dict[str, Any]], *, hours: int, z_threshold: float = 2.0) -> Dict[str, Any]:
    timeline = _quantum_timeline(events, hours=hours, bucket_minutes=60)
    values = [float(p["events"]) for p in timeline["points"]]
    zscores = _compute_z_scores(values)
    anomalies: List[Dict[str, Any]] = []
    for i, p in enumerate(timeline["points"]):
        z = zscores[i] if i < len(zscores) else 0.0
        if abs(z) >= z_threshold:
            anomalies.append(
                {
                    "ts": p["ts"],
                    "events": p["events"],
                    "z_score": round(z, 3),
                    "direction": "high" if z > 0 else "low",
                }
            )
    return {
        "window_hours": hours,
        "z_threshold": z_threshold,
        "total_points": len(timeline["points"]),
        "anomalies": anomalies,
    }


def _quantum_basis_analysis(events: List[Dict[str, Any]], *, hours: int) -> Dict[str, Any]:
    measurements = [e for e in events if e.get("event_type") == "measurement"]
    by_basis: Dict[str, Dict[str, Any]] = {}
    for m in measurements:
        basis = str(m.get("measurement_basis") or "unknown")
        row = by_basis.setdefault(basis, {"basis": basis, "count": 0, "ones": 0, "zeros": 0, "other": 0})
        row["count"] += 1
        outcome = m.get("outcome")
        if outcome == 1:
            row["ones"] += 1
        elif outcome == 0:
            row["zeros"] += 1
        else:
            row["other"] += 1
    rows = []
    for _, r in sorted(by_basis.items(), key=lambda kv: kv[1]["count"], reverse=True):
        total = max(1, int(r["count"]))
        ratio = (int(r["ones"]) / total) * 100.0
        rows.append({**r, "outcome_one_ratio_pct": round(ratio, 2)})
    global_ratio = 0.0
    if measurements:
        ones = sum(1 for m in measurements if m.get("outcome") == 1)
        global_ratio = (ones / len(measurements)) * 100.0
    return {"window_hours": hours, "global_outcome_one_ratio_pct": round(global_ratio, 2), "bases": rows}


def _quantum_health_score(
    *,
    stats: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score = 100.0
    reasons: List[str] = []
    critical = sum(1 for a in alerts if str(a.get("severity")).lower() == "critical")
    warning = sum(1 for a in alerts if str(a.get("severity")).lower() == "warning")
    score -= critical * 25.0
    score -= warning * 10.0
    if anomalies:
        score -= min(30.0, float(len(anomalies)) * 5.0)
        reasons.append(f"{len(anomalies)} anomaly buckets detected.")
    out = stats.get("measurement_outcomes") or {}
    m_total = int(stats.get("measurements") or 0)
    if m_total > 0:
        ones = int(out.get("1") or 0)
        ratio = ones / max(1, m_total)
        bias = abs(ratio - 0.5) * 100.0
        if bias > 20.0:
            score -= 20.0
            reasons.append("Measurement outcomes heavily biased.")
        elif bias > 10.0:
            score -= 10.0
            reasons.append("Measurement outcomes moderately biased.")
    avg_ent = stats.get("avg_entanglement_strength")
    if isinstance(avg_ent, (int, float)):
        if avg_ent < 0.8:
            score -= 15.0
            reasons.append("Average entanglement strength below 0.80.")
        elif avg_ent < 0.9:
            score -= 8.0
            reasons.append("Average entanglement strength below 0.90.")
    score = max(0.0, min(100.0, score))
    tier = "excellent" if score >= 85 else "good" if score >= 70 else "degraded" if score >= 50 else "critical"
    return {"score": round(score, 1), "tier": tier, "reasons": reasons}


def _minimal_pdf_bytes(title: str, lines: List[str]) -> bytes:
    content_lines = [f"({title}) Tj", "T*"]
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_lines.append(f"({safe[:150]}) Tj")
        content_lines.append("T*")
    stream = "BT /F1 11 Tf 50 780 Td " + " ".join(content_lines) + " ET"
    stream_bytes = stream.encode("utf-8")
    objs: List[bytes] = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
    objs.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objs.append(f"5 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("utf-8") + stream_bytes + b"\nendstream endobj\n")

    out = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out += obj
    xref_pos = len(out)
    out += f"xref\n0 {len(offsets)}\n".encode("utf-8")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("utf-8")
    out += f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("utf-8")
    return out


def _quantum_remediation_default() -> Dict[str, Any]:
    return {
        "enabled": True,
        "bias_threshold_pct": 20.0,
        "entanglement_min": 0.8,
        "measure_iterations": 3,
        "entangle_iterations": 2,
        "preferred_basis": "computational",
        "auto_rollback": True,
        "rollback_measure_iterations": 2,
    }


def _get_quantum_remediation_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_remediation_config")
    cfg = _quantum_remediation_default()
    if not raw:
        return cfg
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg.update(parsed)
    except Exception:
        pass
    return cfg


def _set_quantum_remediation_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _quantum_remediation_default()
    cfg.update(data)
    _task_memory.set_setting("quantum_remediation_config", json.dumps(cfg))
    return cfg


def _quantum_notification_default() -> Dict[str, Any]:
    return {
        "enabled": False,
        "channel": "generic",
        "webhook_url": "",
        "webhook_url_warning": "",
        "webhook_url_critical": "",
        "min_severity": "warning",
    }


def _get_quantum_notification_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_notification_config")
    cfg = _quantum_notification_default()
    if not raw:
        return cfg
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg.update(parsed)
    except Exception:
        pass
    return cfg


def _set_quantum_notification_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _quantum_notification_default()
    cfg.update(data)
    _task_memory.set_setting("quantum_notification_config", json.dumps(cfg))
    return cfg


def _quantum_rbac_default() -> Dict[str, Any]:
    return {"role": "operator"}


def _get_quantum_rbac_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_rbac_config")
    cfg = _quantum_rbac_default()
    if not raw:
        return cfg
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg.update(parsed)
    except Exception:
        pass
    role = str(cfg.get("role") or "operator").strip().lower()
    if role not in {"viewer", "operator", "admin"}:
        role = "operator"
    cfg["role"] = role
    return cfg


def _set_quantum_rbac_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _quantum_rbac_default()
    cfg.update(data)
    role = str(cfg.get("role") or "operator").strip().lower()
    if role not in {"viewer", "operator", "admin"}:
        raise HTTPException(status_code=400, detail="invalid role")
    cfg["role"] = role
    _task_memory.set_setting("quantum_rbac_config", json.dumps(cfg))
    return cfg


def _quantum_allowed_actions_for_role(role: str) -> List[str]:
    role_norm = str(role or "operator").lower()
    if role_norm == "admin":
        return ["view", "dispatch_notifications", "simulate", "runbook", "annotate", "incident_manage", "remediate", "policy_apply", "playbook_run", "rbac_manage"]
    if role_norm == "operator":
        return ["view", "dispatch_notifications", "simulate", "runbook", "annotate", "incident_manage", "remediate", "policy_apply", "playbook_run"]
    return ["view"]


def _require_quantum_action(action: str) -> str:
    cfg = _get_quantum_rbac_config()
    role = str(cfg.get("role") or "operator")
    allowed = _quantum_allowed_actions_for_role(role)
    if action == "rbac_manage" and _current_profile() == "full":
        return role
    if action not in allowed:
        raise HTTPException(status_code=403, detail=f"rbac denied for action '{action}' with role '{role}'")
    return role


def _dispatch_webhook_notification(config: Dict[str, Any], payload: Dict[str, Any], *, url_override: Optional[str] = None) -> Dict[str, Any]:
    if not bool(config.get("enabled")):
        return {"sent": False, "reason": "notifications disabled"}
    url = str(url_override or config.get("webhook_url") or "").strip()
    if not url:
        return {"sent": False, "reason": "webhook_url missing"}
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "JarvisAI-QuantumOps/1.0",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            status = int(getattr(resp, "status", 200))
        return {"sent": 200 <= status < 300, "status": status}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


def _audit_quantum_op(op_type: str, status: str, details: Dict[str, Any]) -> None:
    try:
        _task_memory.add_quantum_ops_audit(op_type=op_type, status=status, details=details)
    except Exception:
        pass


def _discord_embed_color_for_severity(severity: str) -> int:
    sev = str(severity or "warning").lower()
    if sev == "critical":
        return 15158332  # red
    if sev == "warning":
        return 15844367  # yellow/orange
    return 3066993  # green-ish


def _notification_payload_for_channel(config: Dict[str, Any], payload: Dict[str, Any], *, severity: Optional[str] = None) -> Dict[str, Any]:
    channel = str(config.get("channel") or "generic").lower()
    if channel == "slack":
        text = payload.get("message") or payload.get("kind") or "Jarvis Quantum Notification"
        if payload.get("alerts"):
            text = f"{text}: {len(payload.get('alerts') or [])} alert(s)"
        return {"text": str(text), "jarvis_payload": payload}
    if channel == "discord":
        sev = str(severity or payload.get("severity") or "warning").lower()
        title = payload.get("message") or payload.get("kind") or "Jarvis Quantum Notification"
        alerts = payload.get("alerts") if isinstance(payload.get("alerts"), list) else []
        fields: List[Dict[str, Any]] = []
        for a in alerts[:8]:
            fields.append(
                {
                    "name": f"{str(a.get('severity') or 'warning').upper()} - {str(a.get('code') or 'alert')}",
                    "value": str(a.get("message") or a.get("value") or "No message"),
                    "inline": False,
                }
            )
        if not fields:
            fields.append({"name": "Details", "value": str(title), "inline": False})
        embed = {
            "title": "Jarvis Quantum Alert",
            "description": str(title),
            "color": _discord_embed_color_for_severity(sev),
            "fields": fields,
            "footer": {"text": "Jarvis Quantum Ops"},
            "timestamp": _now_iso().replace("+00:00", "Z"),
        }
        return {"content": "", "embeds": [embed], "jarvis_payload": payload}
    return payload


def _webhook_for_severity(config: Dict[str, Any], severity: Optional[str]) -> str:
    sev = str(severity or "warning").lower()
    critical = str(config.get("webhook_url_critical") or "").strip()
    warning = str(config.get("webhook_url_warning") or "").strip()
    default = str(config.get("webhook_url") or "").strip()
    if sev == "critical" and critical:
        return critical
    if sev in {"warning", "info"} and warning:
        return warning
    return default


def _dispatch_configured_notification(config: Dict[str, Any], payload: Dict[str, Any], *, severity: Optional[str] = None) -> Dict[str, Any]:
    target_url = _webhook_for_severity(config, severity)
    wrapped = _notification_payload_for_channel(config, payload, severity=severity)
    return _dispatch_webhook_notification(config, wrapped, url_override=target_url)


def _memory_saved_filters_get() -> List[Dict[str, Any]]:
    raw = _task_memory.get_setting("memory_archived_saved_filters")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items: List[Dict[str, Any]] = []
            for item in parsed[:24]:
                if isinstance(item, dict):
                    items.append(
                        {
                            "name": str(item.get("name") or "").strip()[:60],
                            "query": str(item.get("query") or "").strip()[:160],
                            "tag": str(item.get("tag") or "").strip()[:60],
                            "memory_type": str(item.get("memory_type") or "").strip()[:40],
                        }
                    )
            return [item for item in items if item["name"]]
    except Exception:
        pass
    return []


def _memory_saved_filters_set(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for item in items[:24]:
        name = str(item.get("name") or "").strip()[:60]
        if not name:
            continue
        clean.append(
            {
                "name": name,
                "query": str(item.get("query") or "").strip()[:160],
                "tag": str(item.get("tag") or "").strip()[:60],
                "memory_type": str(item.get("memory_type") or "").strip()[:40],
            }
        )
    _task_memory.set_setting("memory_archived_saved_filters", json.dumps(clean))
    return clean


def _briefing_delivery_default() -> Dict[str, Any]:
    return {
        "enabled": False,
        "discord_enabled": False,
        "discord_webhook_url": "",
        "email_enabled": False,
        "email_to": "",
        "mobile_enabled": False,
        "mobile_push_url": "",
        "mobile_channel": "ntfy",
    }


def _get_briefing_delivery_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("memory_briefing_delivery_config")
    cfg = _briefing_delivery_default()
    if not raw:
        return cfg
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg.update(parsed)
    except Exception:
        pass
    return cfg


def _set_briefing_delivery_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _briefing_delivery_default()
    cfg.update(data)
    cfg["mobile_channel"] = str(cfg.get("mobile_channel") or "ntfy").strip().lower()
    if cfg["mobile_channel"] not in {"ntfy", "generic"}:
        raise HTTPException(status_code=400, detail="invalid mobile channel")
    for field_name in ["discord_webhook_url", "mobile_push_url"]:
        value = str(cfg.get(field_name) or "").strip()
        if value and not (value.startswith("http://") or value.startswith("https://")):
            raise HTTPException(status_code=400, detail=f"{field_name} must start with http:// or https://")
        cfg[field_name] = value
    cfg["email_to"] = str(cfg.get("email_to") or "").strip()
    _task_memory.set_setting("memory_briefing_delivery_config", json.dumps(cfg))
    return cfg


def _dispatch_briefing_discord(webhook_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    config = {"enabled": True, "channel": "discord", "webhook_url": webhook_url}
    wrapped = {
        "message": payload.get("text") or "Jarvis briefing update",
        "alerts": [
            {
                "severity": "info",
                "code": payload.get("period") or "briefing",
                "message": payload.get("text") or "Jarvis briefing delivered",
            }
        ],
        "severity": "info",
    }
    return _dispatch_webhook_notification(config, _notification_payload_for_channel(config, wrapped, severity="info"))


def _dispatch_briefing_email(email_to: str, subject: str, body_text: str) -> Dict[str, Any]:
    host = str(os.getenv("SMTP_HOST") or "").strip()
    port = int(str(os.getenv("SMTP_PORT") or "587").strip() or "587")
    username = str(os.getenv("SMTP_USER") or "").strip()
    password = str(os.getenv("SMTP_PASSWORD") or "").strip()
    from_email = str(os.getenv("SMTP_FROM") or username or "jarvis@localhost").strip()
    if not host:
        return {"sent": False, "reason": "smtp_host_missing"}
    msg = MIMEText(body_text, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = email_to
    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            try:
                server.starttls()
            except Exception:
                pass
            if username:
                server.login(username, password)
            server.sendmail(from_email, [email_to], msg.as_string())
        return {"sent": True, "target": email_to}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


def _dispatch_briefing_mobile(push_url: str, payload: Dict[str, Any], channel: str = "ntfy") -> Dict[str, Any]:
    text = str(payload.get("text") or "Jarvis briefing update")
    headers = {"User-Agent": "JarvisAI-Briefing/1.0"}
    data = text.encode("utf-8")
    if str(channel or "ntfy").lower() == "ntfy":
        headers["Title"] = f"Jarvis {str(payload.get('period') or 'briefing').title()} Briefing"
        headers["Priority"] = "default"
        headers["Tags"] = "jarvis,memo"
        req = urlrequest.Request(url=push_url, data=data, method="POST", headers=headers)
    else:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        req = urlrequest.Request(url=push_url, data=body, method="POST", headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            status = int(getattr(resp, "status", 200))
        return {"sent": 200 <= status < 300, "status": status}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


def _dispatch_briefing_deliveries(briefing: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _get_briefing_delivery_config()
    if not bool(cfg.get("enabled")):
        return {"enabled": False, "results": []}
    text = str(briefing.get("text") or "").strip()
    payload = {
        "kind": "jarvis_memory_briefing",
        "period": briefing.get("period"),
        "text": text,
        "timestamp": _now_iso(),
    }
    subject = f"Jarvis {str(briefing.get('period') or 'briefing').title()} Briefing"
    results: List[Dict[str, Any]] = []
    if bool(cfg.get("discord_enabled")) and str(cfg.get("discord_webhook_url") or "").strip():
        results.append({"channel": "discord", **_dispatch_briefing_discord(str(cfg.get("discord_webhook_url")), payload)})
    if bool(cfg.get("email_enabled")) and str(cfg.get("email_to") or "").strip():
        results.append({"channel": "email", **_dispatch_briefing_email(str(cfg.get("email_to")), subject, text)})
    if bool(cfg.get("mobile_enabled")) and str(cfg.get("mobile_push_url") or "").strip():
        mobile_result = _dispatch_briefing_mobile(
            str(cfg.get("mobile_push_url")),
            payload,
            channel=str(cfg.get("mobile_channel") or "ntfy"),
        )
        if isinstance(mobile_result, dict) and "channel" in mobile_result:
            mobile_result = {k: v for k, v in mobile_result.items() if k != "channel"}
        results.append(
            {
                "channel": "mobile",
                **mobile_result,
            }
        )
    return {"enabled": True, "results": results}


def _get_quantum_incidents() -> List[Dict[str, Any]]:
    raw = _task_memory.get_setting("quantum_incidents")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def _set_quantum_incidents(items: List[Dict[str, Any]]) -> None:
    _task_memory.set_setting("quantum_incidents", json.dumps(items))


def _incident_default_checklist(code: str) -> List[Dict[str, Any]]:
    rb = _runbook_for_alert_code(code).get("steps") or []
    return [{"id": f"c{i+1}", "text": str(step), "done": False} for i, step in enumerate(rb[:8])]


def _update_incident(
    incident_id: str,
    *,
    status: Optional[str] = None,
    checklist: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    items = _get_quantum_incidents()
    changed: Optional[Dict[str, Any]] = None
    now = _now_iso()
    for item in items:
        if str(item.get("id")) != str(incident_id):
            continue
        if status is not None:
            item["status"] = status
            if status == "closed":
                item["closed_at"] = now
        if checklist is not None:
            item["checklist"] = checklist
        item["updated_at"] = now
        changed = item
        break
    if changed is not None:
        _set_quantum_incidents(items)
    return changed


def _sync_incidents_from_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current = _get_quantum_incidents()
    now = _now_iso()
    active_keys = {f"{a.get('code')}::{a.get('message')}" for a in alerts}
    by_key: Dict[str, Dict[str, Any]] = {}
    for item in current:
        key = f"{item.get('code')}::{item.get('message')}"
        by_key[key] = item
    for a in alerts:
        key = f"{a.get('code')}::{a.get('message')}"
        if key in by_key:
            by_key[key]["status"] = "open"
            by_key[key]["last_event_at"] = now
            by_key[key]["severity"] = a.get("severity")
            continue
        by_key[key] = {
            "id": f"inc-{abs(hash(key)) % 10000000}",
            "status": "open",
            "severity": a.get("severity"),
            "code": a.get("code"),
            "message": a.get("message"),
            "checklist": _incident_default_checklist(str(a.get("code") or "")),
            "opened_at": now,
            "closed_at": None,
            "last_event_at": now,
        }
    for key, item in by_key.items():
        if key not in active_keys and item.get("status") == "open":
            item["status"] = "closed"
            item["closed_at"] = now
    items = sorted(by_key.values(), key=lambda x: str(x.get("last_event_at") or x.get("opened_at") or ""), reverse=True)
    _set_quantum_incidents(items)
    return items


def _parse_iso_safe(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _event_service_name(e: Dict[str, Any]) -> str:
    basis = str(e.get("measurement_basis") or "core").strip().lower()
    if basis in {"computational", "hadamard"}:
        return f"quantum-{basis}"
    return f"quantum-{basis or 'core'}"


def _incident_service_name(inc: Dict[str, Any]) -> str:
    code = str(inc.get("code") or "general").strip().lower()
    if "entangle" in code:
        return "quantum-entanglement"
    if "measurement" in code:
        return "quantum-measurement"
    return "quantum-core"


def _quantum_alert_correlations(*, hours: int, window_minutes: int) -> Dict[str, Any]:
    incidents = _get_quantum_incidents()
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    win = max(1, int(window_minutes))
    groups: Dict[str, Dict[str, Any]] = {}

    for inc in incidents:
        opened = _parse_iso_safe(inc.get("opened_at"))
        bucket = "unknown"
        if opened:
            minute_bucket = (opened.minute // win) * win
            bucket = opened.replace(minute=minute_bucket, second=0, microsecond=0).isoformat()
        service = _incident_service_name(inc)
        signature = f"{service}|{inc.get('code')}|{bucket}"
        g = groups.setdefault(
            signature,
            {
                "signature": signature,
                "service": service,
                "code": inc.get("code"),
                "bucket_start": bucket,
                "incident_ids": [],
                "severity_max": "info",
                "count": 0,
            },
        )
        g["incident_ids"].append(inc.get("id"))
        g["count"] += 1
        sev = str(inc.get("severity") or "info").lower()
        if _severity_rank(sev) > _severity_rank(str(g.get("severity_max") or "info")):
            g["severity_max"] = sev

    event_by_service: Dict[str, int] = {}
    for e in events:
        svc = _event_service_name(e)
        event_by_service[svc] = int(event_by_service.get(svc, 0)) + 1
    for g in groups.values():
        g["event_count"] = int(event_by_service.get(str(g.get("service") or ""), 0))
        g["priority"] = "high" if _severity_rank(str(g.get("severity_max") or "info")) >= _severity_rank("critical") else "normal"

    correlated = sorted(groups.values(), key=lambda x: (int(x.get("count") or 0), int(x.get("event_count") or 0)), reverse=True)
    return {"window_hours": hours, "window_minutes": win, "groups": correlated[:100]}


def _compute_quantum_baselines(hours: int) -> Dict[str, Any]:
    h = max(1, min(int(hours), 24 * 365))
    events = _task_memory.list_quantum_events(limit=5000, since_hours=h)
    stats = _stats_from_events(events, hours=h)
    measurements = max(1, int(stats.get("measurements") or 0))
    out = stats.get("measurement_outcomes") or {}
    one_ratio_pct = (float(out.get("1") or 0) / measurements) * 100.0
    baseline = {
        "window_hours": h,
        "generated_at": _now_iso(),
        "avg_measurements_per_hour": round(float(stats.get("measurements") or 0) / float(h), 3),
        "avg_entangles_per_hour": round(float(stats.get("entangles") or 0) / float(h), 3),
        "outcome_one_ratio_pct": round(one_ratio_pct, 3),
        "avg_entanglement_strength": round(float(stats.get("avg_entanglement_strength") or 0.0), 4),
        "avg_correlation_coefficient": round(float(stats.get("avg_correlation_coefficient") or 0.0), 4),
    }
    return baseline


def _get_quantum_baselines() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_baselines")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _set_quantum_baselines(baseline: Dict[str, Any]) -> Dict[str, Any]:
    _task_memory.set_setting("quantum_baselines", json.dumps(baseline))
    return baseline


def _quantum_baseline_drift(*, hours: int) -> Dict[str, Any]:
    baseline = _get_quantum_baselines()
    if not baseline:
        baseline = _set_quantum_baselines(_compute_quantum_baselines(max(24, hours)))
    current = _compute_quantum_baselines(hours)
    drift: Dict[str, Any] = {}
    keys = [
        "avg_measurements_per_hour",
        "avg_entangles_per_hour",
        "outcome_one_ratio_pct",
        "avg_entanglement_strength",
        "avg_correlation_coefficient",
    ]
    for k in keys:
        b = float(baseline.get(k) or 0.0)
        c = float(current.get(k) or 0.0)
        delta = c - b
        pct = 0.0 if abs(b) < 1e-9 else (delta / b) * 100.0
        drift[k] = {"baseline": round(b, 4), "current": round(c, 4), "delta": round(delta, 4), "delta_pct": round(pct, 2)}
    return {"baseline": baseline, "current": current, "drift": drift}


def _quantum_root_cause_graph(*, incident_id: Optional[str], hours: int) -> Dict[str, Any]:
    incidents = _get_quantum_incidents()
    selected = None
    if incident_id:
        for inc in incidents:
            if str(inc.get("id")) == str(incident_id):
                selected = inc
                break
    if selected is None and incidents:
        selected = incidents[0]
    events = _task_memory.list_quantum_events(limit=1000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    root_cause = "measurement_pipeline_drift"
    confidence = 0.55
    if (stats.get("avg_entanglement_strength") or 1.0) < 0.8:
        root_cause = "entanglement_link_decay"
        confidence = 0.72
    elif int(stats.get("measurements") or 0) < 5:
        root_cause = "insufficient_sampling"
        confidence = 0.68
    elif abs(float((stats.get("avg_correlation_coefficient") or 0.0))) < 0.2:
        root_cause = "correlation_collapse"
        confidence = 0.61

    nodes: List[Dict[str, Any]] = [{"id": "quantum-core", "type": "system", "label": "Quantum Core"}]
    edges: List[Dict[str, Any]] = []
    if selected:
        iid = f"incident-{selected.get('id')}"
        nodes.append({"id": iid, "type": "incident", "label": str(selected.get("code") or "incident")})
        edges.append({"from": "quantum-core", "to": iid, "kind": "raised"})
    for e in events[:20]:
        eid = f"event-{e.get('id')}"
        nodes.append({"id": eid, "type": "event", "label": str(e.get("event_type") or "event")})
        edges.append({"from": "quantum-core", "to": eid, "kind": "observed"})
    cause_id = f"cause-{root_cause}"
    nodes.append({"id": cause_id, "type": "cause", "label": root_cause})
    edges.append({"from": cause_id, "to": "quantum-core", "kind": "impacts"})
    if selected:
        edges.append({"from": cause_id, "to": f"incident-{selected.get('id')}", "kind": "explains"})

    return {
        "incident": selected,
        "suspected_root_cause": root_cause,
        "confidence": round(confidence, 2),
        "nodes": nodes,
        "edges": edges,
    }


def _quantum_risk_score(*, horizon_hours: int) -> Dict[str, Any]:
    h = max(1, min(int(horizon_hours), 24 * 7))
    stats = _task_memory.quantum_stats(hours=24)
    alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
    anomalies = _quantum_anomalies(_task_memory.list_quantum_events(limit=5000, since_hours=24), hours=24, z_threshold=2.0)
    alert_weight = min(50.0, float(len(alerts_eval.get("alerts") or [])) * 15.0)
    anomaly_weight = min(30.0, float(len(anomalies.get("anomalies") or [])) * 6.0)
    ent = float(stats.get("avg_entanglement_strength") or 1.0)
    stability_weight = max(0.0, (0.9 - ent) * 100.0)
    raw = min(100.0, alert_weight + anomaly_weight + stability_weight)
    tier = "low"
    if raw >= 70:
        tier = "high"
    elif raw >= 35:
        tier = "medium"
    return {
        "horizon_hours": h,
        "risk_score": round(raw, 2),
        "tier": tier,
        "drivers": {
            "active_alerts": len(alerts_eval.get("alerts") or []),
            "anomalies": len(anomalies.get("anomalies") or []),
            "avg_entanglement_strength": round(ent, 4),
        },
        "forecast": {
            "incident_probability_pct": round(min(99.0, raw * 0.9), 2),
            "confidence_pct": round(min(95.0, 45.0 + h * 1.1), 2),
        },
    }


def _quantum_playbook_v2_default() -> Dict[str, Any]:
    return {
        "enabled": True,
        "require_approval": True,
        "canary_checks": True,
        "auto_rollback": True,
        "max_actions": 6,
    }


def _get_quantum_playbook_v2_config() -> Dict[str, Any]:
    raw = _task_memory.get_setting("quantum_playbook_v2_config")
    cfg = _quantum_playbook_v2_default()
    if not raw:
        return cfg
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg.update(parsed)
    except Exception:
        pass
    return cfg


def _set_quantum_playbook_v2_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _quantum_playbook_v2_default()
    cfg.update(data)
    cfg["max_actions"] = max(1, min(int(cfg.get("max_actions") or 6), 20))
    _task_memory.set_setting("quantum_playbook_v2_config", json.dumps(cfg))
    return cfg


def _quantum_slo_panel(*, hours: int) -> Dict[str, Any]:
    h = max(1, min(int(hours), 24 * 365))
    events = _task_memory.list_quantum_events(limit=5000, since_hours=h)
    alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
    total = max(1, len(events))
    errors = len(alerts_eval.get("alerts") or [])
    availability = max(0.0, 100.0 - (errors / total) * 100.0)
    mttr_minutes = round(8.0 + errors * 2.4, 2)
    mttd_minutes = round(2.0 + errors * 1.1, 2)
    burn = round(min(100.0, (errors / total) * 400.0), 2)
    return {
        "window_hours": h,
        "availability_pct": round(availability, 3),
        "error_budget_burn_pct": burn,
        "mttr_minutes": mttr_minutes,
        "mttd_minutes": mttd_minutes,
        "services": [
            {"name": "quantum-core", "slo_target_pct": 99.9, "achieved_pct": round(availability, 3)},
            {"name": "quantum-measurement", "slo_target_pct": 99.5, "achieved_pct": round(max(0.0, availability - 0.4), 3)},
            {"name": "quantum-entanglement", "slo_target_pct": 99.0, "achieved_pct": round(max(0.0, availability - 0.8), 3)},
        ],
    }


def _quantum_decyphering_lab(*, hours: int) -> Dict[str, Any]:
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    measurements = [e for e in events if e.get("event_type") == "measurement"]
    n = max(1, len(measurements))
    ones = sum(1 for m in measurements if int(m.get("outcome") or 0) == 1)
    p1 = ones / n
    p0 = 1.0 - p1
    entropy = 0.0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * math.log2(p)
    signature = "balanced"
    if abs(p1 - 0.5) > 0.2:
        signature = "biased"
    elif len(events) < 10:
        signature = "low-signal"
    behavior = "stable" if (len(events) >= 10 and signature == "balanced") else "volatile"
    confidence = min(99.0, 30.0 + len(events) * 1.6)
    return {
        "window_hours": hours,
        "signals": {
            "event_count": len(events),
            "measurement_count": len(measurements),
            "outcome_one_ratio_pct": round(p1 * 100.0, 3),
            "entropy_bits": round(entropy, 5),
            "signature": signature,
            "behavior": behavior,
            "confidence_pct": round(confidence, 2),
        },
        "explain": f"Entropy near 1.0 means high uncertainty; current signature is '{signature}' and behavior is '{behavior}'.",
    }


def _quantum_generate_postmortem(*, incident_id: Optional[str], hours: int) -> Dict[str, Any]:
    workspace = quantum_incident_workspace(incident_id=incident_id)
    incident = workspace.get("incident")
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    decypher = _quantum_decyphering_lab(hours=hours)
    rc = _quantum_root_cause_graph(incident_id=str(incident.get("id")) if incident else None, hours=hours)
    timeline = []
    for e in events[:15]:
        timeline.append({"ts": e.get("created_at"), "event_type": e.get("event_type"), "basis": e.get("measurement_basis")})
    return {
        "incident": incident,
        "summary": "Quantum incident postmortem generated from live telemetry and incident workspace.",
        "impact": {
            "alerts_triggered": len((_evaluate_quantum_alerts(_get_quantum_alert_config()).get("alerts") or [])),
            "estimated_user_impact": "moderate",
        },
        "suspected_root_cause": rc.get("suspected_root_cause"),
        "root_cause_confidence": rc.get("confidence"),
        "timeline": timeline,
        "decyphering": decypher,
        "action_items": [
            "Add stricter canary checks before remediation.",
            "Increase measurement sample window for better confidence.",
            "Track SLO burn daily in the dashboard.",
        ],
    }


def _policy_packs() -> Dict[str, Dict[str, Any]]:
    return {
        "safe": {
            "alerts": {"window_hours": 24, "min_measurements": 10, "min_entangles": 5, "outcome_one_min_pct": 30.0, "outcome_one_max_pct": 70.0, "entanglement_strength_min": 0.9, "enabled": True},
            "remediation": {"enabled": True, "bias_threshold_pct": 25.0, "entanglement_min": 0.85, "measure_iterations": 2, "entangle_iterations": 1, "preferred_basis": "computational", "auto_rollback": True, "rollback_measure_iterations": 1},
        },
        "aggressive": {
            "alerts": {"window_hours": 12, "min_measurements": 4, "min_entangles": 2, "outcome_one_min_pct": 20.0, "outcome_one_max_pct": 80.0, "entanglement_strength_min": 0.8, "enabled": True},
            "remediation": {"enabled": True, "bias_threshold_pct": 12.0, "entanglement_min": 0.9, "measure_iterations": 5, "entangle_iterations": 3, "preferred_basis": "computational", "auto_rollback": True, "rollback_measure_iterations": 3},
        },
        "research": {
            "alerts": {"window_hours": 48, "min_measurements": 3, "min_entangles": 1, "outcome_one_min_pct": 5.0, "outcome_one_max_pct": 95.0, "entanglement_strength_min": 0.6, "enabled": True},
            "remediation": {"enabled": False, "bias_threshold_pct": 35.0, "entanglement_min": 0.7, "measure_iterations": 1, "entangle_iterations": 0, "preferred_basis": "hadamard", "auto_rollback": False, "rollback_measure_iterations": 0},
        },
    }


def _runbook_for_alert_code(code: str) -> Dict[str, Any]:
    c = (code or "").strip().lower()
    runbooks = {
        "measurement_outcome_bias": [
            "Capture 24h basis-analysis and anomaly snapshot.",
            "Run quick experiment preset to increase sample quality.",
            "If bias persists, apply remediation tune+apply and re-check health.",
        ],
        "entanglement_strength_low": [
            "Run forced remediation to stabilize entanglement links.",
            "Increase entangle_iterations and re-test with timeline.",
            "Escalate if health score remains below 60 for 2 cycles.",
        ],
    }
    steps = runbooks.get(c, ["Review active alerts and run basis-analysis.", "Execute quick experiment and compare NOC deltas.", "Use remediation tune before force actions."])
    return {"code": code, "title": f"Runbook for {code or 'general'}", "steps": steps}


def _run_quantum_experiment(*, preset: str, measure_count: Optional[int], entangle_count: Optional[int]) -> Dict[str, Any]:
    defaults = {
        "quick": {"measure_count": 5, "entangle_count": 2},
        "balanced": {"measure_count": 10, "entangle_count": 5},
        "deep": {"measure_count": 20, "entangle_count": 10},
    }
    selected = defaults[preset]
    measures = int(measure_count) if measure_count is not None else int(selected["measure_count"])
    entangles = int(entangle_count) if entangle_count is not None else int(selected["entangle_count"])
    executions: List[Dict[str, Any]] = []
    for i in range(max(0, entangles)):
        executions.append(
            {
                "kind": "entangle",
                "index": i + 1,
                "result": _tool_quantum_entangle({"system_a": f"core_{i % 3}", "system_b": f"memory_{i % 3}"}),
            }
        )
    for i in range(max(1, measures)):
        basis = "computational" if i % 2 == 0 else "hadamard"
        executions.append(
            {
                "kind": "measurement",
                "index": i + 1,
                "result": _tool_quantum_measure({"measurement_basis": basis}),
            }
        )
    events = _task_memory.list_quantum_events(limit=5000, since_hours=24)
    decipher = _quantum_decipher_analysis(events, hours=24)
    snap_id = _task_memory.create_quantum_decipher_snapshot(decipher)
    result = {
        "preset": preset,
        "measure_count": measures,
        "entangle_count": entangles,
        "executions": len(executions),
        "snapshot_id": snap_id,
        "decipher": decipher,
    }
    _audit_quantum_op("experiment_run", "ok", {"preset": preset, "executions": len(executions), "snapshot_id": snap_id})
    return result


def _run_quantum_remediation(*, hours: int, force: bool) -> Dict[str, Any]:
    cfg = _get_quantum_remediation_config()
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    decipher = _quantum_decipher_analysis(events, hours=hours)
    pre_stats = _stats_from_events(events, hours=hours)
    pre_alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
    pre_anoms = _quantum_anomalies(events, hours=hours, z_threshold=2.0)
    pre_health = _quantum_health_score(stats=pre_stats, alerts=pre_alerts["alerts"], anomalies=pre_anoms["anomalies"])
    avg_ent = decipher["signals"].get("avg_entanglement_strength")
    bias_pct = float(decipher["signals"].get("outcome_bias_pct") or 0.0)
    should_run = bool(force) or (
        bool(cfg.get("enabled"))
        and (bias_pct >= float(cfg.get("bias_threshold_pct") or 20.0) or (isinstance(avg_ent, (int, float)) and avg_ent < float(cfg.get("entanglement_min") or 0.8)))
    )
    actions: List[Dict[str, Any]] = []
    if should_run:
        for i in range(int(cfg.get("entangle_iterations") or 0)):
            actions.append(
                {
                    "kind": "entangle",
                    "index": i + 1,
                    "result": _tool_quantum_entangle({"system_a": f"stabilizer_{i}", "system_b": f"memory_{i}"}),
                }
            )
        for i in range(int(cfg.get("measure_iterations") or 0)):
            actions.append(
                {
                    "kind": "measure",
                    "index": i + 1,
                    "result": _tool_quantum_measure({"measurement_basis": str(cfg.get("preferred_basis") or "computational")}),
                }
            )
    post_events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    post_stats = _stats_from_events(post_events, hours=hours)
    post_alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
    post_anoms = _quantum_anomalies(post_events, hours=hours, z_threshold=2.0)
    post_health = _quantum_health_score(stats=post_stats, alerts=post_alerts["alerts"], anomalies=post_anoms["anomalies"])

    rollback_performed = False
    rollback_steps = 0
    if should_run and bool(cfg.get("auto_rollback")) and float(post_health["score"]) < float(pre_health["score"]):
        rb_iters = int(cfg.get("rollback_measure_iterations") or 0)
        for _i in range(max(0, rb_iters)):
            _tool_quantum_measure({"measurement_basis": "computational"})
            rollback_steps += 1
        rollback_performed = rollback_steps > 0
        post_events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
        post_stats = _stats_from_events(post_events, hours=hours)
        post_alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
        post_anoms = _quantum_anomalies(post_events, hours=hours, z_threshold=2.0)
        post_health = _quantum_health_score(stats=post_stats, alerts=post_alerts["alerts"], anomalies=post_anoms["anomalies"])

    result = {
        "ran": should_run,
        "actions": len(actions),
        "pre_check": decipher,
        "config": cfg,
        "health_pre": pre_health,
        "health_post": post_health,
        "rollback_performed": rollback_performed,
        "rollback_steps": rollback_steps,
    }
    _audit_quantum_op("remediation_run", "ok", {"ran": bool(should_run), "actions": len(actions), "force": bool(force)})
    return result

def _stats_from_events(events: List[Dict[str, Any]], *, hours: int) -> Dict[str, Any]:
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
        "window_hours": hours,
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


def _quantum_decipher_analysis(events: List[Dict[str, Any]], *, hours: int) -> Dict[str, Any]:
    stats = _stats_from_events(events, hours=hours)
    measurements = [e for e in events if e.get("event_type") == "measurement"]
    entangles = [e for e in events if e.get("event_type") == "entangle"]

    zero_count = int(stats["measurement_outcomes"]["0"])
    one_count = int(stats["measurement_outcomes"]["1"])
    measured = max(1, zero_count + one_count)
    one_ratio = one_count / measured
    bias_pct = abs(one_ratio - 0.5) * 100.0
    confidence = min(100.0, (len(events) / 40.0) * 100.0)

    patterns: List[str] = []
    if not events:
        patterns.append("No quantum events available for deciphering.")
    else:
        if bias_pct >= 20:
            patterns.append(f"Strong measurement bias detected ({one_ratio * 100:.1f}% outcome=1).")
        elif bias_pct >= 8:
            patterns.append(f"Moderate measurement tilt observed ({one_ratio * 100:.1f}% outcome=1).")
        else:
            patterns.append(f"Measurement outcomes are near-balanced ({one_ratio * 100:.1f}% outcome=1).")

        avg_strength = stats.get("avg_entanglement_strength")
        if isinstance(avg_strength, (int, float)) and avg_strength >= 0.9:
            patterns.append("Entanglement channel is stable (avg strength >= 0.90).")
        elif isinstance(avg_strength, (int, float)):
            patterns.append("Entanglement channel is weak-to-moderate; correlation hardening recommended.")

        unique_bases = sorted({str(m.get("measurement_basis")) for m in measurements if m.get("measurement_basis")})
        if unique_bases:
            patterns.append("Observed measurement bases: " + ", ".join(unique_bases[:4]))

    recommendations: List[str] = []
    if stats["measurements"] < 5:
        recommendations.append("Increase measurement volume (>=5) before making policy changes.")
    if stats["entangles"] < 3:
        recommendations.append("Run additional entanglement cycles to improve trend reliability.")
    if bias_pct >= 20:
        recommendations.append("Rotate measurement basis and compare bias drift over the next 24h.")
    if not recommendations:
        recommendations.append("Current quantum behavior appears stable; keep alerts enabled and monitor deltas.")

    return {
        "window_hours": hours,
        "events_analyzed": len(events),
        "confidence_pct": round(confidence, 1),
        "signals": {
            "outcome_one_ratio_pct": round(one_ratio * 100.0, 2),
            "outcome_bias_pct": round(bias_pct, 2),
            "avg_entanglement_strength": stats.get("avg_entanglement_strength"),
            "avg_measurement_probability": stats.get("avg_measurement_probability"),
        },
        "patterns": patterns,
        "recommendations": recommendations,
        "stats": stats,
    }


def _download_filename(prefix: str, ext: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}.{ext}"


def _json_download(payload: Dict[str, Any], *, filename: str) -> JSONResponse:
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _csv_download(text: str, *, filename: str) -> PlainTextResponse:
    return PlainTextResponse(
        content=text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _zip_download(files: Dict[str, str], *, filename: str) -> Response:
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for inner_name, content in files.items():
            zf.writestr(inner_name, content)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/quantum/export")
def quantum_export(
    format: str = Query(default="json", pattern="^(json|csv)$"),
    hours: int = Query(default=24, ge=1, le=24 * 365),
    event_type: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
    start_at: Optional[str] = Query(default=None),
    end_at: Optional[str] = Query(default=None),
):
    start_dt = _parse_iso_datetime(start_at, field_name="start_at")
    end_dt = _parse_iso_datetime(end_at, field_name="end_at")
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_at must be <= end_at")

    events = _task_memory.list_quantum_events(limit=limit, event_type=event_type, since_hours=hours)
    events = _filter_events_by_time_window(events, start_at=start_dt, end_at=end_dt)
    stats = _stats_from_events(events, hours=hours)
    payload = {
        "window_hours": hours,
        "event_type": event_type,
        "start_at": start_dt.isoformat() if start_dt else None,
        "end_at": end_dt.isoformat() if end_dt else None,
        "stats": stats,
        "events": events,
    }
    if format == "json":
        return _json_download(payload, filename=_download_filename("quantum_history", "json"))

    csv_rows: List[Dict[str, Any]] = []
    for e in events:
        csv_rows.append(
            {
                "id": e.get("id"),
                "event_type": e.get("event_type"),
                "created_at": e.get("created_at"),
                "measurement_basis": e.get("measurement_basis"),
                "outcome": e.get("outcome"),
                "measurement_probability": e.get("measurement_probability"),
                "entanglement_strength": e.get("entanglement_strength"),
                "correlation_coefficient": e.get("correlation_coefficient"),
                "states": json.dumps(e.get("states") or []),
            }
        )
    text = _to_csv(
        csv_rows,
        columns=[
            "id",
            "event_type",
            "created_at",
            "measurement_basis",
            "outcome",
            "measurement_probability",
            "entanglement_strength",
            "correlation_coefficient",
            "states",
        ],
    )
    return _csv_download(text, filename=_download_filename("quantum_history", "csv"))


@router.get("/quantum/alerts/export")
def quantum_alerts_export(
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    cfg = _get_quantum_alert_config()
    evaluation = _evaluate_quantum_alerts(cfg)
    payload = {
        "generated_at": _now_iso(),
        "config": cfg,
        "active": evaluation["active"],
        "alerts": evaluation["alerts"],
        "stats": evaluation["stats"],
    }
    if format == "json":
        return _json_download(payload, filename=_download_filename("quantum_alerts", "json"))

    rows = []
    for a in payload["alerts"]:
        rows.append(
            {
                "generated_at": payload["generated_at"],
                "active": payload["active"],
                "code": a.get("code"),
                "severity": a.get("severity"),
                "message": a.get("message"),
                "value": a.get("value"),
                "window_hours": payload["config"].get("window_hours"),
            }
        )
    if not rows:
        rows.append(
            {
                "generated_at": payload["generated_at"],
                "active": payload["active"],
                "code": "",
                "severity": "",
                "message": "No active alerts",
                "value": "",
                "window_hours": payload["config"].get("window_hours"),
            }
        )
    text = _to_csv(
        rows,
        columns=["generated_at", "active", "code", "severity", "message", "value", "window_hours"],
    )
    return _csv_download(text, filename=_download_filename("quantum_alerts", "csv"))


@router.get("/quantum/export/all")
def quantum_export_all(
    format: str = Query(default="csv", pattern="^(json|csv)$"),
    hours: int = Query(default=24, ge=1, le=24 * 365),
    event_type: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
    start_at: Optional[str] = Query(default=None),
    end_at: Optional[str] = Query(default=None),
):
    start_dt = _parse_iso_datetime(start_at, field_name="start_at")
    end_dt = _parse_iso_datetime(end_at, field_name="end_at")
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_at must be <= end_at")

    events = _task_memory.list_quantum_events(limit=limit, event_type=event_type, since_hours=hours)
    events = _filter_events_by_time_window(events, start_at=start_dt, end_at=end_dt)
    stats = _stats_from_events(events, hours=hours)

    cfg = _get_quantum_alert_config()
    evaluation = _evaluate_quantum_alerts(cfg)
    alerts_payload = {
        "generated_at": _now_iso(),
        "config": cfg,
        "active": evaluation["active"],
        "alerts": evaluation["alerts"],
        "stats": evaluation["stats"],
    }
    decipher_payload = _quantum_decipher_analysis(events, hours=hours)

    if format == "json":
        files = {
            "quantum_history.json": json.dumps(
                {
                    "window_hours": hours,
                    "event_type": event_type,
                    "start_at": start_dt.isoformat() if start_dt else None,
                    "end_at": end_dt.isoformat() if end_dt else None,
                    "stats": stats,
                    "events": events,
                },
                indent=2,
            ),
            "quantum_alerts.json": json.dumps(alerts_payload, indent=2),
            "quantum_decipher.json": json.dumps(decipher_payload, indent=2),
        }
        return _zip_download(files, filename=_download_filename("quantum_bundle_json", "zip"))

    event_rows: List[Dict[str, Any]] = []
    for e in events:
        event_rows.append(
            {
                "id": e.get("id"),
                "event_type": e.get("event_type"),
                "created_at": e.get("created_at"),
                "measurement_basis": e.get("measurement_basis"),
                "outcome": e.get("outcome"),
                "measurement_probability": e.get("measurement_probability"),
                "entanglement_strength": e.get("entanglement_strength"),
                "correlation_coefficient": e.get("correlation_coefficient"),
                "states": json.dumps(e.get("states") or []),
            }
        )
    alerts_rows: List[Dict[str, Any]] = []
    for a in alerts_payload["alerts"]:
        alerts_rows.append(
            {
                "generated_at": alerts_payload["generated_at"],
                "active": alerts_payload["active"],
                "code": a.get("code"),
                "severity": a.get("severity"),
                "message": a.get("message"),
                "value": a.get("value"),
                "window_hours": alerts_payload["config"].get("window_hours"),
            }
        )
    if not alerts_rows:
        alerts_rows.append(
            {
                "generated_at": alerts_payload["generated_at"],
                "active": alerts_payload["active"],
                "code": "",
                "severity": "",
                "message": "No active alerts",
                "value": "",
                "window_hours": alerts_payload["config"].get("window_hours"),
            }
        )
    decipher_rows = [
        {
            "window_hours": decipher_payload["window_hours"],
            "events_analyzed": decipher_payload["events_analyzed"],
            "confidence_pct": decipher_payload["confidence_pct"],
            "outcome_one_ratio_pct": decipher_payload["signals"]["outcome_one_ratio_pct"],
            "outcome_bias_pct": decipher_payload["signals"]["outcome_bias_pct"],
            "avg_entanglement_strength": decipher_payload["signals"]["avg_entanglement_strength"],
            "avg_measurement_probability": decipher_payload["signals"]["avg_measurement_probability"],
            "patterns": " | ".join(decipher_payload["patterns"]),
            "recommendations": " | ".join(decipher_payload["recommendations"]),
        }
    ]
    files = {
        "quantum_history.csv": _to_csv(
            event_rows,
            columns=[
                "id",
                "event_type",
                "created_at",
                "measurement_basis",
                "outcome",
                "measurement_probability",
                "entanglement_strength",
                "correlation_coefficient",
                "states",
            ],
        ),
        "quantum_alerts.csv": _to_csv(
            alerts_rows,
            columns=["generated_at", "active", "code", "severity", "message", "value", "window_hours"],
        ),
        "quantum_decipher.csv": _to_csv(
            decipher_rows,
            columns=[
                "window_hours",
                "events_analyzed",
                "confidence_pct",
                "outcome_one_ratio_pct",
                "outcome_bias_pct",
                "avg_entanglement_strength",
                "avg_measurement_probability",
                "patterns",
                "recommendations",
            ],
        ),
    }
    return _zip_download(files, filename=_download_filename("quantum_bundle_csv", "zip"))


@router.get("/quantum/decipher")
def quantum_decipher(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    event_type: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
    start_at: Optional[str] = Query(default=None),
    end_at: Optional[str] = Query(default=None),
):
    start_dt = _parse_iso_datetime(start_at, field_name="start_at")
    end_dt = _parse_iso_datetime(end_at, field_name="end_at")
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(status_code=400, detail="start_at must be <= end_at")
    events = _task_memory.list_quantum_events(limit=limit, event_type=event_type, since_hours=hours)
    events = _filter_events_by_time_window(events, start_at=start_dt, end_at=end_dt)
    return _quantum_decipher_analysis(events, hours=hours)


@router.get("/quantum/decypher")
def quantum_decypher_alias(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    event_type: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
    start_at: Optional[str] = Query(default=None),
    end_at: Optional[str] = Query(default=None),
):
    return quantum_decipher(hours=hours, event_type=event_type, limit=limit, start_at=start_at, end_at=end_at)


@router.get("/quantum/timeline")
def quantum_timeline(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    bucket_minutes: int = Query(default=60, ge=1, le=240),
):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    return _quantum_timeline(events, hours=hours, bucket_minutes=bucket_minutes)


@router.get("/quantum/anomalies")
def quantum_anomalies(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    z_threshold: float = Query(default=2.0, ge=0.5, le=6.0),
):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    return _quantum_anomalies(events, hours=hours, z_threshold=z_threshold)


@router.get("/quantum/basis-analysis")
def quantum_basis_analysis(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    return _quantum_basis_analysis(events, hours=hours)


@router.post("/quantum/experiment/run")
def quantum_experiment_run(payload: QuantumExperimentRequest):
    return _run_quantum_experiment(preset=payload.preset, measure_count=payload.measure_count, entangle_count=payload.entangle_count)


@router.post("/quantum/decipher/snapshot")
def quantum_decipher_snapshot(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    decipher = _quantum_decipher_analysis(events, hours=hours)
    snap_id = _task_memory.create_quantum_decipher_snapshot(decipher)
    _audit_quantum_op("snapshot_create", "ok", {"snapshot_id": snap_id, "window_hours": hours, "events": len(events)})
    return {"id": snap_id, "snapshot": decipher}


@router.get("/quantum/decipher/snapshots")
def quantum_decipher_snapshots(limit: int = Query(default=20, ge=1, le=200)):
    return {"snapshots": _task_memory.list_quantum_decipher_snapshots(limit=limit)}


@router.get("/quantum/ops-audit")
def quantum_ops_audit(limit: int = Query(default=100, ge=1, le=500)):
    return {"items": _task_memory.list_quantum_ops_audit(limit=limit)}


@router.get("/quantum/remediation/config")
def quantum_remediation_config_get():
    return _get_quantum_remediation_config()


@router.post("/quantum/remediation/config")
def quantum_remediation_config_set(payload: QuantumRemediationConfigRequest):
    return _set_quantum_remediation_config(payload.model_dump())


@router.post("/quantum/remediation/tune")
def quantum_remediation_tune(hours: int = Query(default=168, ge=24, le=24 * 365), apply: bool = False):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    m = stats.get("measurement_outcomes") or {}
    measurements = max(1, int(stats.get("measurements") or 0))
    one_ratio_pct = (int(m.get("1") or 0) / measurements) * 100.0
    bias_pct = abs(one_ratio_pct - 50.0)
    avg_ent = stats.get("avg_entanglement_strength")
    suggested = _quantum_remediation_default()
    suggested["bias_threshold_pct"] = round(max(8.0, min(35.0, bias_pct + 5.0)), 2)
    if isinstance(avg_ent, (int, float)):
        suggested["entanglement_min"] = round(max(0.6, min(0.95, float(avg_ent) - 0.05)), 3)
    suggested["measure_iterations"] = 4 if bias_pct > 15 else 2
    suggested["entangle_iterations"] = 3 if (isinstance(avg_ent, (int, float)) and avg_ent < 0.85) else 1
    if apply:
        applied_cfg = _set_quantum_remediation_config(suggested)
        _audit_quantum_op("remediation_tune", "ok", {"window_hours": hours, "applied": True, "suggested": applied_cfg})
        return {"window_hours": hours, "applied": True, "config": applied_cfg, "stats": stats}
    _audit_quantum_op("remediation_tune", "ok", {"window_hours": hours, "applied": False, "suggested": suggested})
    return {"window_hours": hours, "applied": False, "suggested_config": suggested, "stats": stats}


@router.post("/quantum/remediation/run")
def quantum_remediation_run(hours: int = Query(default=24, ge=1, le=24 * 365), force: bool = False):
    role = _require_quantum_action("remediate")
    _audit_quantum_op("rbac_allow", "ok", {"action": "remediate", "role": role})
    return _run_quantum_remediation(hours=hours, force=force)


@router.get("/quantum/health-score")
def quantum_health_score(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
    anomalies = _quantum_anomalies(events, hours=hours, z_threshold=2.0)
    score = _quantum_health_score(stats=stats, alerts=alerts_eval["alerts"], anomalies=anomalies["anomalies"])
    return {
        "window_hours": hours,
        "score": score["score"],
        "tier": score["tier"],
        "reasons": score["reasons"],
        "stats": stats,
        "alerts_active": alerts_eval["active"],
        "anomaly_count": len(anomalies["anomalies"]),
    }


@router.get("/quantum/noc")
def quantum_noc(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
    anomalies = _quantum_anomalies(events, hours=hours, z_threshold=2.0)
    score = _quantum_health_score(stats=stats, alerts=alerts_eval["alerts"], anomalies=anomalies["anomalies"])
    snaps = _task_memory.list_quantum_decipher_snapshots(limit=2)
    delta = None
    if len(snaps) >= 2:
        a = float((snaps[0].get("signals") or {}).get("outcome_bias_pct") or 0.0)
        b = float((snaps[1].get("signals") or {}).get("outcome_bias_pct") or 0.0)
        delta = round(a - b, 3)
    return {
        "window_hours": hours,
        "health_score": score["score"],
        "health_tier": score["tier"],
        "active_alerts": int(len(alerts_eval["alerts"])),
        "anomaly_count": int(len(anomalies["anomalies"])),
        "last_snapshot_delta_bias_pct": delta,
        "events": int(stats.get("total_events") or 0),
    }


@router.get("/quantum/incidents")
def quantum_incidents(limit: int = Query(default=200, ge=1, le=1000)):
    items = _get_quantum_incidents()
    return {"incidents": items[:limit]}


@router.get("/quantum/incidents/workspace")
def quantum_incident_workspace(incident_id: Optional[str] = None):
    items = _get_quantum_incidents()
    target = None
    if incident_id:
        for i in items:
            if str(i.get("id")) == str(incident_id):
                target = i
                break
    if target is None and items:
        target = items[0]
    incident_annotations: List[Dict[str, Any]] = []
    if target:
        incident_annotations = _task_memory.list_quantum_annotations(limit=100, item_type="incident", item_id=str(target.get("id")))
    return {"incident": target, "annotations": incident_annotations, "incidents": items[:50]}


@router.post("/quantum/incidents/{incident_id}/status")
def quantum_incident_status_update(incident_id: str, status: str = Query(..., pattern="^(open|acked|closed)$")):
    role = _require_quantum_action("incident_manage")
    changed = _update_incident(incident_id, status=status)
    if changed is None:
        raise HTTPException(status_code=404, detail="incident not found")
    _audit_quantum_op("incident_status_update", "ok", {"incident_id": incident_id, "status": status, "role": role})
    return {"incident": changed}


@router.post("/quantum/incidents/{incident_id}/checklist")
def quantum_incident_checklist_toggle(incident_id: str, item_id: str = Query(...), done: bool = Query(...)):
    role = _require_quantum_action("incident_manage")
    items = _get_quantum_incidents()
    changed = None
    for inc in items:
        if str(inc.get("id")) != str(incident_id):
            continue
        checklist = inc.get("checklist") if isinstance(inc.get("checklist"), list) else []
        for c in checklist:
            if str(c.get("id")) == str(item_id):
                c["done"] = bool(done)
                changed = inc
                break
        inc["checklist"] = checklist
        break
    if changed is None:
        raise HTTPException(status_code=404, detail="incident/checklist item not found")
    _set_quantum_incidents(items)
    _audit_quantum_op("incident_checklist_toggle", "ok", {"incident_id": incident_id, "item_id": item_id, "done": bool(done), "role": role})
    return {"incident": changed}


@router.get("/quantum/stream")
async def quantum_stream():
    async def event_gen():
        for _ in range(300):
            events = _task_memory.list_quantum_events(limit=5000, since_hours=24)
            alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
            incidents = _sync_incidents_from_alerts(alerts_eval["alerts"])
            noc = quantum_noc(hours=24)
            payload = {
                "ts": _now_iso(),
                "noc": noc,
                "alerts": alerts_eval,
                "incidents": incidents[:25],
                "recent_audit": _task_memory.list_quantum_ops_audit(limit=10),
                "risk": _quantum_risk_score(horizon_hours=24),
                "slo": _quantum_slo_panel(hours=24),
            }
            yield f"event: quantum\n"
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/quantum/annotations")
def quantum_annotations_create(payload: QuantumAnnotationCreateRequest):
    aid = _task_memory.add_quantum_annotation(
        item_type=payload.item_type,
        item_id=payload.item_id,
        note=payload.note,
        author=payload.author,
    )
    _audit_quantum_op("annotation_create", "ok", {"id": aid, "item_type": payload.item_type, "item_id": payload.item_id})
    return {"id": aid, "ok": True}


@router.get("/quantum/annotations")
def quantum_annotations_list(
    limit: int = Query(default=200, ge=1, le=1000),
    item_type: Optional[str] = None,
    item_id: Optional[str] = None,
):
    return {"annotations": _task_memory.list_quantum_annotations(limit=limit, item_type=item_type, item_id=item_id)}


@router.get("/quantum/memory-graph")
def quantum_memory_graph(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=400, since_hours=hours)
    alerts_eval = _evaluate_quantum_alerts(_get_quantum_alert_config())
    audits = _task_memory.list_quantum_ops_audit(limit=100)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    nodes.append({"id": "root", "type": "system", "label": "Quantum Core"})
    for e in events[:60]:
        nid = f"event-{e.get('id')}"
        nodes.append({"id": nid, "type": "event", "label": f"{e.get('event_type')} #{e.get('id')}"})
        edges.append({"from": "root", "to": nid, "kind": "records"})
    for a in alerts_eval["alerts"]:
        aid = f"alert-{a.get('code')}"
        nodes.append({"id": aid, "type": "alert", "label": str(a.get("code"))})
        edges.append({"from": "root", "to": aid, "kind": "triggers"})
    for op in audits[:40]:
        oid = f"audit-{op.get('id')}"
        nodes.append({"id": oid, "type": "audit", "label": f"{op.get('op_type')} ({op.get('status')})"})
        edges.append({"from": "root", "to": oid, "kind": "logs"})
    return {"nodes": nodes, "edges": edges}


@router.get("/quantum/replay/agent")
def quantum_agent_replay(limit: int = Query(default=20, ge=1, le=200), session_id: Optional[str] = None):
    runs = _task_memory.list_goal_runs(limit=limit, session_id=session_id)
    timeline: List[Dict[str, Any]] = []
    for r in runs:
        timeline.append(
            {
                "type": "run",
                "run_id": r.get("id"),
                "status": r.get("status"),
                "goal": r.get("goal"),
                "created_at": r.get("created_at"),
            }
        )
        for s in (r.get("steps") or []):
            timeline.append(
                {
                    "type": "step",
                    "run_id": r.get("id"),
                    "index": s.get("index"),
                    "tool": s.get("tool"),
                    "status": s.get("status"),
                    "label": s.get("label"),
                }
            )
    return {"timeline": timeline}


@router.get("/quantum/simulate")
def quantum_simulate(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    outcome_one_min_pct: float = Query(default=20.0, ge=0.0, le=100.0),
    outcome_one_max_pct: float = Query(default=80.0, ge=0.0, le=100.0),
    entanglement_strength_min: float = Query(default=0.9, ge=0.0, le=1.0),
):
    cfg = _get_quantum_alert_config()
    sim_cfg = {**cfg, "outcome_one_min_pct": outcome_one_min_pct, "outcome_one_max_pct": outcome_one_max_pct, "entanglement_strength_min": entanglement_strength_min, "window_hours": hours}
    evald = _evaluate_quantum_alerts(sim_cfg)
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    health = _quantum_health_score(stats=stats, alerts=evald["alerts"], anomalies=_quantum_anomalies(events, hours=hours, z_threshold=2.0)["anomalies"])
    return {"simulated_config": sim_cfg, "alerts": evald["alerts"], "active": evald["active"], "health": health}


@router.get("/quantum/policy-packs")
def quantum_policy_packs():
    return {"packs": _policy_packs()}


@router.post("/quantum/policy-packs/{pack_name}/apply")
def quantum_policy_apply(pack_name: str):
    role = _require_quantum_action("policy_apply")
    packs = _policy_packs()
    key = str(pack_name or "").strip().lower()
    if key not in packs:
        raise HTTPException(status_code=404, detail="policy pack not found")
    selected = packs[key]
    alerts_cfg = _set_quantum_alert_config(selected["alerts"])
    rem_cfg = _set_quantum_remediation_config(selected["remediation"])
    _audit_quantum_op("policy_apply", "ok", {"pack": key, "role": role})
    return {"pack": key, "alerts_config": alerts_cfg, "remediation_config": rem_cfg}


@router.get("/quantum/runbook/{code}")
def quantum_runbook(code: str):
    return _runbook_for_alert_code(code)


@router.get("/quantum/replay")
def quantum_replay(
    hours: int = Query(default=24, ge=1, le=24 * 365),
    end_at: Optional[str] = Query(default=None),
):
    end_dt = _parse_iso_datetime(end_at, field_name="end_at") if end_at else datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=hours)
    events = _task_memory.list_quantum_events(limit=5000, since_hours=24 * 365)
    window_events = _filter_events_by_time_window(events, start_at=start_dt, end_at=end_dt)
    stats = _stats_from_events(window_events, hours=hours)
    decipher = _quantum_decipher_analysis(window_events, hours=hours)
    anomalies = _quantum_anomalies(window_events, hours=hours, z_threshold=2.0)
    alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
    return {
        "window": {"start_at": start_dt.isoformat(), "end_at": end_dt.isoformat(), "hours": hours},
        "events": len(window_events),
        "stats": stats,
        "decipher": decipher,
        "anomalies": anomalies,
        "alerts": alerts,
    }


@router.get("/quantum/summary.pdf")
def quantum_summary_pdf(hours: int = Query(default=24, ge=1, le=24 * 365)):
    events = _task_memory.list_quantum_events(limit=5000, since_hours=hours)
    stats = _stats_from_events(events, hours=hours)
    decipher = _quantum_decipher_analysis(events, hours=hours)
    health = quantum_health_score(hours=hours)
    lines = [
        f"Window: {hours}h",
        f"Events: {stats.get('total_events')}, Measurements: {stats.get('measurements')}, Entangles: {stats.get('entangles')}",
        f"Health Score: {health.get('score')} ({health.get('tier')})",
        f"Outcome=1 Ratio: {decipher['signals'].get('outcome_one_ratio_pct')}%",
        "Patterns: " + " | ".join(decipher.get("patterns") or []),
        "Recommendations: " + " | ".join(decipher.get("recommendations") or []),
    ]
    pdf_bytes = _minimal_pdf_bytes("Jarvis Quantum Summary", lines)
    _audit_quantum_op("summary_pdf", "ok", {"window_hours": hours, "events": len(events)})
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_download_filename("quantum_summary", "pdf")}"'},
    )


@router.get("/quantum/notifications/config")
def quantum_notifications_config_get():
    return _get_quantum_notification_config()


@router.post("/quantum/notifications/config")
def quantum_notifications_config_set(payload: QuantumNotificationConfigRequest):
    data = payload.model_dump()
    urls = [
        str(data.get("webhook_url") or "").strip(),
        str(data.get("webhook_url_warning") or "").strip(),
        str(data.get("webhook_url_critical") or "").strip(),
    ]
    for u in urls:
        if u and (not u.startswith("http://") and not u.startswith("https://")):
            raise HTTPException(status_code=400, detail="all webhook URLs must start with http:// or https://")
    if data.get("enabled") and not any(urls):
        raise HTTPException(status_code=400, detail="at least one webhook URL is required when notifications are enabled")
    cfg = _set_quantum_notification_config(data)
    _audit_quantum_op("notification_config_set", "ok", {"enabled": bool(cfg.get("enabled")), "channel": cfg.get("channel")})
    return cfg


@router.post("/quantum/notifications/test")
def quantum_notifications_test():
    cfg = _get_quantum_notification_config()
    payload = {"kind": "quantum_test", "timestamp": _now_iso(), "message": "Jarvis quantum webhook test ping", "severity": "warning"}
    res = _dispatch_configured_notification(cfg, payload, severity="warning")
    _audit_quantum_op("notification_test", "ok" if bool(res.get("sent")) else "failed", {"result": res})
    return {"config": cfg, "result": res}


@router.post("/quantum/notifications/dispatch")
def quantum_notifications_dispatch(hours: int = Query(default=24, ge=1, le=24 * 365)):
    role = _require_quantum_action("dispatch_notifications")
    cfg = _get_quantum_notification_config()
    alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
    _sync_incidents_from_alerts(alerts["alerts"])
    min_sev = str(cfg.get("min_severity") or "warning")
    filtered = [a for a in alerts["alerts"] if _severity_rank(str(a.get("severity") or "warning")) >= _severity_rank(min_sev)]
    if not filtered:
        _audit_quantum_op("notification_dispatch", "ok", {"sent": False, "reason": "no_alerts", "alerts_total": len(alerts["alerts"])})
        return {"sent": False, "reason": "no alerts at or above configured severity", "alerts_total": len(alerts["alerts"])}
    routed: Dict[str, List[Dict[str, Any]]] = {"critical": [], "warning": [], "info": []}
    for a in filtered:
        sev = str(a.get("severity") or "warning").lower()
        if sev not in routed:
            sev = "warning"
        routed[sev].append(a)
    dispatch_results: Dict[str, Any] = {}
    sent_any = False
    total_sent_alerts = 0
    for sev, items in routed.items():
        if not items:
            continue
        payload = {"kind": "quantum_alerts", "timestamp": _now_iso(), "window_hours": hours, "alerts": items, "severity": sev}
        res = _dispatch_configured_notification(cfg, payload, severity=sev)
        dispatch_results[sev] = res
        if bool(res.get("sent")):
            sent_any = True
            total_sent_alerts += len(items)
    _audit_quantum_op(
        "notification_dispatch",
        "ok" if sent_any else "failed",
        {"alerts_sent": total_sent_alerts, "results": dispatch_results, "role": role},
    )
    return {"sent": sent_any, "alerts_sent": total_sent_alerts, "results": dispatch_results}


@router.get("/quantum/correlations")
def quantum_correlations(hours: int = Query(default=24, ge=1, le=24 * 365), window_minutes: int = Query(default=30, ge=1, le=240)):
    return _quantum_alert_correlations(hours=hours, window_minutes=window_minutes)


@router.post("/quantum/baselines/recompute")
def quantum_baselines_recompute(hours: int = Query(default=24 * 7, ge=24, le=24 * 365)):
    role = _require_quantum_action("simulate")
    baseline = _set_quantum_baselines(_compute_quantum_baselines(hours))
    _audit_quantum_op("baselines_recompute", "ok", {"hours": hours, "role": role})
    return {"baseline": baseline}


@router.get("/quantum/baselines")
def quantum_baselines(hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _quantum_baseline_drift(hours=hours)


@router.get("/quantum/root-cause")
def quantum_root_cause(incident_id: Optional[str] = None, hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _quantum_root_cause_graph(incident_id=incident_id, hours=hours)


@router.get("/quantum/risk-score")
def quantum_risk_score(horizon_hours: int = Query(default=24, ge=1, le=24 * 7)):
    return _quantum_risk_score(horizon_hours=horizon_hours)


@router.get("/quantum/playbooks-v2/config")
def quantum_playbook_v2_config_get():
    return _get_quantum_playbook_v2_config()


@router.post("/quantum/playbooks-v2/config")
def quantum_playbook_v2_config_set(payload: Dict[str, Any]):
    role = _require_quantum_action("playbook_run")
    cfg = _set_quantum_playbook_v2_config(payload)
    _audit_quantum_op("playbook_v2_config_set", "ok", {"role": role, "config": cfg})
    return cfg


@router.post("/quantum/playbooks-v2/run")
def quantum_playbook_v2_run(payload: QuantumPlaybookRunRequest):
    role = _require_quantum_action("playbook_run")
    cfg = _get_quantum_playbook_v2_config()
    if bool(cfg.get("require_approval")) and not bool(payload.approve):
        return {"ok": False, "blocked": True, "reason": "approval required", "config": cfg}

    runbook = _runbook_for_alert_code("general")
    workspace = quantum_incident_workspace(incident_id=payload.incident_id)
    incident = workspace.get("incident")
    plan_steps = (runbook.get("steps") or [])[: int(cfg.get("max_actions") or 6)]
    executed: List[Dict[str, Any]] = []
    health_pre = quantum_health_score(hours=24)
    if not payload.dry_run:
        rem = _run_quantum_remediation(hours=24, force=True)
        executed.append({"action": "remediation", "result": rem})
    health_post = quantum_health_score(hours=24)

    out = {
        "ok": True,
        "role": role,
        "dry_run": bool(payload.dry_run),
        "incident": incident,
        "canary_checks_passed": bool(cfg.get("canary_checks")),
        "steps": plan_steps,
        "executed": executed,
        "health_pre": health_pre,
        "health_post": health_post,
        "rollback_ready": bool(cfg.get("auto_rollback")),
    }
    _audit_quantum_op("playbook_v2_run", "ok", {"role": role, "dry_run": bool(payload.dry_run), "incident_id": payload.incident_id})
    return out


@router.get("/quantum/postmortem")
def quantum_postmortem_generate(incident_id: Optional[str] = None, hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _quantum_generate_postmortem(incident_id=incident_id, hours=hours)


@router.get("/quantum/postmortem.pdf")
def quantum_postmortem_pdf(incident_id: Optional[str] = None, hours: int = Query(default=24, ge=1, le=24 * 365)):
    report = _quantum_generate_postmortem(incident_id=incident_id, hours=hours)
    incident = report.get("incident") if isinstance(report.get("incident"), dict) else {}
    lines = [
        f"Window: {hours}h",
        f"Incident ID: {incident.get('id') or 'n/a'}",
        f"Code: {incident.get('code') or 'n/a'}",
        f"Severity: {incident.get('severity') or 'n/a'}",
        f"Root Cause: {report.get('suspected_root_cause') or 'unknown'}",
        f"Confidence: {report.get('root_cause_confidence') or 'n/a'}",
        "Actions: " + " | ".join(report.get("action_items") or []),
    ]
    pdf_bytes = _minimal_pdf_bytes("Jarvis Quantum Postmortem", lines)
    _audit_quantum_op("postmortem_pdf", "ok", {"incident_id": incident.get("id"), "window_hours": hours})
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{_download_filename("quantum_postmortem", "pdf")}"'},
    )


@router.get("/quantum/slo")
def quantum_slo(hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _quantum_slo_panel(hours=hours)


@router.get("/quantum/decyphering/lab")
def quantum_decyphering_lab(hours: int = Query(default=24, ge=1, le=24 * 365)):
    return _quantum_decyphering_lab(hours=hours)


@router.get("/quantum/rbac")
def quantum_rbac_get():
    cfg = _get_quantum_rbac_config()
    role = str(cfg.get("role") or "operator")
    return {"role": role, "actions": _quantum_allowed_actions_for_role(role)}


@router.post("/quantum/rbac")
def quantum_rbac_set(payload: QuantumRbacConfigRequest):
    actor = _require_quantum_action("rbac_manage")
    cfg = _set_quantum_rbac_config(payload.model_dump())
    role = str(cfg.get("role") or "operator")
    _audit_quantum_op("rbac_set", "ok", {"actor_role": actor, "role": role})
    return {"role": role, "actions": _quantum_allowed_actions_for_role(role)}


@router.post("/quantum/sandbox/run")
def quantum_sandbox_run(payload: QuantumSandboxRunRequest):
    role = _require_quantum_action("simulate")
    real_alerts = _evaluate_quantum_alerts(_get_quantum_alert_config())
    fake_alert = {
        "code": payload.inject_alert_code or "sandbox_alert",
        "severity": payload.inject_severity,
        "message": f"Sandbox injected drift={payload.drift_pct}%",
        "value": payload.drift_pct,
    }
    merged_alerts = list(real_alerts.get("alerts") or []) + [fake_alert]
    fake_health = _quantum_health_score(
        stats=_task_memory.quantum_stats(hours=payload.hours),
        alerts=merged_alerts,
        anomalies=_quantum_anomalies(_task_memory.list_quantum_events(limit=5000, since_hours=payload.hours), hours=payload.hours, z_threshold=2.0).get("anomalies") or [],
    )
    result = {
        "name": payload.name,
        "hours": payload.hours,
        "role": role,
        "sandbox": True,
        "injected_alert": fake_alert,
        "base_alert_count": len(real_alerts.get("alerts") or []),
        "simulated_alert_count": len(merged_alerts),
        "simulated_health": fake_health,
        "recommendation": "Enable playbook dry-run first, then promote to operator action.",
    }
    _audit_quantum_op("sandbox_run", "ok", {"name": payload.name, "role": role, "drift_pct": payload.drift_pct})
    return result

def _execute_goal_run(*, goal: str, session_id: Optional[str], auto_approve: bool) -> Dict[str, Any]:
    sid = session_id or str(uuid4())
    goal_text = goal.strip()
    profile = _current_profile()
    plan = _build_plan(goal_text)

    task_id = _task_memory.create_task(goal_text, session_id=sid)
    _task_memory.update_task_status(task_id, "in_progress", note="Goal runner started")
    run_id = _task_memory.create_goal_run(task_id=task_id, session_id=sid, goal=goal_text, plan=plan)

    actions = _goal_to_actions(goal_text)
    steps: List[Dict[str, Any]] = []
    blocked = False
    failed = False
    final_result: Dict[str, Any] = {"goal": goal_text}

    for i, action in enumerate(actions, start=1):
        tool = str(action.get("tool", ""))
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        label = str(action.get("label", tool))

        if _action_requires_approval(action, profile) and not auto_approve:
            blocked = True
            steps.append(
                {
                    "index": i,
                    "label": label,
                    "tool": tool,
                    "status": "blocked",
                    "reason": "requires approval",
                    "profile": profile,
                }
            )
            break

        if tool in {"shell_run", "repo_write_file"}:
            args["confirm"] = True

        last_error: Optional[str] = None
        result: Any = None
        ok = False
        for _attempt in (1, 2):
            try:
                if tool == "__chat__":
                    response = agent_chat(AgentChatRequest(message=str(args.get("message", "")), session_id=sid))
                    result = {
                        "reply": response.reply,
                        "tool_result": response.tool_result,
                        "plan": response.plan,
                    }
                else:
                    if tool not in _TOOLS:
                        raise HTTPException(status_code=400, detail=f"unknown tool in goal plan: {tool}")
                    result = _TOOLS[tool].handler(args)
                ok = True
                break
            except Exception as exc:
                last_error = str(exc)

        if not ok:
            failed = True
            steps.append(
                {
                    "index": i,
                    "label": label,
                    "tool": tool,
                    "status": "failed",
                    "error": last_error or "unknown error",
                }
            )
            break
        verification = None
        if tool not in {"__chat__"} and isinstance(result, dict):
            verification = _record_verified_tool_run(tool_name=tool, result=result, args=args, session_id=sid)
        steps.append({"index": i, "label": label, "tool": tool, "status": "ok", "result": result, "verification": verification})

    if blocked:
        status = "blocked"
        _task_memory.update_task_status(task_id, "blocked", note="Awaiting approval")
        final_result["status"] = "blocked"
    elif failed:
        status = "failed"
        _task_memory.update_task_status(task_id, "failed", note="Goal execution failed")
        final_result["status"] = "failed"
    else:
        status = "done"
        _task_memory.update_task_status(task_id, "done", note="Goal execution complete")
        final_result["status"] = "done"

    final_result["steps_executed"] = len(steps)
    _task_memory.update_goal_run(run_id, status=status, steps=steps, result=final_result)

    return {
        "ok": status == "done",
        "run_id": run_id,
        "task_id": task_id,
        "session_id": sid,
        "goal": goal_text,
        "profile": profile,
        "status": status,
        "plan": plan,
        "steps": steps,
        "result": final_result,
    }


@router.post("/goals/run")
def run_goal(payload: GoalRunRequest):
    return _execute_goal_run(goal=payload.goal, session_id=payload.session_id, auto_approve=payload.auto_approve)


@router.get("/goals/history", response_model=GoalHistoryResponse)
def goal_history(limit: int = Query(default=20, ge=1, le=100), session_id: Optional[str] = None):
    return GoalHistoryResponse(runs=_task_memory.list_goal_runs(limit=limit, session_id=session_id))


@router.post("/goals/schedule")
def create_goal_schedule(payload: GoalScheduleCreateRequest):
    schedule_id = _task_memory.create_goal_schedule(
        goal=payload.goal.strip(),
        interval_minutes=payload.interval_minutes,
        session_id=payload.session_id,
        auto_approve=payload.auto_approve,
        enabled=payload.enabled,
    )
    return {"id": schedule_id, "ok": True}


@router.get("/goals/schedule")
def list_goal_schedules(limit: int = Query(default=100, ge=1, le=500)):
    return {"schedules": _task_memory.list_goal_schedules(limit=limit)}


@router.post("/goals/schedule/{schedule_id}")
def update_goal_schedule(schedule_id: int, payload: GoalScheduleUpdateRequest):
    ok = _task_memory.update_goal_schedule(
        schedule_id,
        enabled=payload.enabled,
        interval_minutes=payload.interval_minutes,
        auto_approve=payload.auto_approve,
        goal=payload.goal.strip() if isinstance(payload.goal, str) else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="schedule not found")
    return {"id": schedule_id, "ok": True}


@router.post("/multi-agent/run")
def run_multi_agent(payload: MultiAgentRunRequest):
    session_id = payload.session_id or str(uuid4())
    task = payload.task.strip()
    _memory.append(session_id, StoredMessage(role="user", text=f"[multi-agent] {task}"))

    provider = os.getenv("LLM_PROVIDER", "").strip().lower() or "auto"
    if provider == "auto":
        provider = "ollama" if is_ollama_configured() else "basic"
    fast_synthesis = bool(payload.fast_synthesis)
    first_pass_roles = ["planner", "coder"] if fast_synthesis else ["planner", "researcher", "coder", "operator"]
    agent_results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_multi_agent_workers(provider)) as pool:
        future_map = {
            pool.submit(_agent_role_reason, role, task, session_id, True): role
            for role in first_pass_roles
        }
        for fut in as_completed(future_map):
            role = future_map[fut]
            try:
                result = fut.result()
                if isinstance(result, dict):
                    agent_results[role] = result
                else:
                    agent_results[role] = _normalize_role_payload(role, task, {"summary": str(result)})
            except Exception as exc:
                agent_results[role] = _normalize_role_payload(
                    role,
                    task,
                    {"summary": f"{role} failed: {exc}", "risks": [f"{role} exception"], "actions": ["Retry role worker"]},
                )

    planner = agent_results.get("planner", _normalize_role_payload("planner", task, {}))
    coder = agent_results.get("coder", _normalize_role_payload("coder", task, {}))

    expanded = False
    if fast_synthesis and _needs_deep_synthesis(task, planner, coder):
        expanded = True
        with ThreadPoolExecutor(max_workers=_multi_agent_workers(provider)) as pool:
            future_map = {
                pool.submit(_agent_role_reason, "researcher", task, session_id, True): "researcher",
                pool.submit(_agent_role_reason, "operator", task, session_id, True): "operator",
            }
            for fut in as_completed(future_map):
                role = future_map[fut]
                try:
                    result = fut.result()
                    if isinstance(result, dict):
                        agent_results[role] = result
                    else:
                        agent_results[role] = _normalize_role_payload(role, task, {"summary": str(result)})
                except Exception as exc:
                    agent_results[role] = _normalize_role_payload(
                        role,
                        task,
                        {"summary": f"{role} failed: {exc}", "risks": [f"{role} exception"], "actions": ["Retry role worker"]},
                    )

    researcher = agent_results.get("researcher", _normalize_role_payload("researcher", task, {}))
    operator = agent_results.get("operator")
    if not isinstance(operator, dict):
        operator = _operator_from_roles(task=task, planner=planner, coder=coder, researcher=researcher)

    synthesis_lines = [
        f"Planner: {planner.get('summary')}",
        f"Researcher: {researcher.get('summary')}",
        f"Coder: {coder.get('summary')}",
        f"Operator: {operator.get('summary')}",
        "Top actions:",
    ]
    merged_actions: List[str] = []
    for r in (planner, researcher, coder, operator):
        for a in (r.get("actions") or []):
            if isinstance(a, str) and a.strip() and a.strip() not in merged_actions:
                merged_actions.append(a.strip())
    for a in merged_actions[:8]:
        synthesis_lines.append(f"- {a}")
    synthesis = "\n".join(synthesis_lines)
    _memory.append(session_id, StoredMessage(role="assistant", text=synthesis))
    return {
        "session_id": session_id,
        "task": task,
        "fast_synthesis": fast_synthesis,
        "expanded": expanded,
        "first_pass_roles": first_pass_roles,
        "agents": {"planner": planner, "researcher": researcher, "coder": coder, "operator": operator},
        "synthesis": synthesis,
        "actions": merged_actions[:8],
        "timestamp": _now_iso(),
    }


@router.get("/memory/quality")
def memory_quality(session_id: str = Query(..., min_length=1), max_messages: int = Query(default=120, ge=20, le=400)):
    return _memory_quality_report(session_id=session_id, max_messages=max_messages)


@router.post("/memory/remember")
def remember_memory(payload: LongTermMemoryCreateRequest):
    memory_id = _task_memory.create_long_term_memory(
        content=payload.content.strip(),
        tags=payload.tags,
        importance=payload.importance,
        memory_type=payload.memory_type,
        subject=payload.subject,
        pinned=payload.pinned,
        lane=payload.lane,
        source=payload.source,
        session_id=payload.session_id,
    )
    return {"ok": True, "id": memory_id}


@router.get("/memory/search")
def search_memories(query: str = Query(..., min_length=1), limit: int = Query(default=8, ge=1, le=50)):
    return {"items": _task_memory.search_long_term_memories(query=query, limit=limit)}


@router.get("/memory/long-term")
def list_long_term_memories(
    limit: int = Query(default=50, ge=1, le=200),
    memory_type: Optional[str] = Query(default=None),
    subject: Optional[str] = Query(default=None),
    pinned: Optional[bool] = Query(default=None),
    since_hours: Optional[int] = Query(default=None, ge=1, le=24 * 365),
    archived: bool = Query(default=False),
    query: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    lane: Optional[str] = Query(default=None),
):
    return {
        "items": _task_memory.list_long_term_memories(
            limit=limit,
            memory_type=memory_type,
            subject=subject,
            pinned=pinned,
            since_hours=since_hours,
            archived=archived,
            query=query,
            tag=tag,
            lane=lane,
        )
    }


@router.get("/memory/overview")
def memory_overview(
    limit_per_group: int = Query(default=6, ge=2, le=20),
    since_hours: Optional[int] = Query(default=None, ge=1, le=24 * 365),
    archived: bool = Query(default=False),
):
    return _task_memory.memory_overview(limit_per_group=limit_per_group, since_hours=since_hours, archived=archived)


@router.get("/memory/profile")
def memory_profile(limit: int = Query(default=8, ge=1, le=50), since_hours: Optional[int] = Query(default=None, ge=1, le=24 * 365)):
    overview = _task_memory.memory_overview(limit_per_group=max(3, limit), since_hours=since_hours)
    items = (overview.get("profile") or [])[:limit]
    return {"items": items, "total": len(items)}


@router.get("/memory/projects")
def memory_projects(limit: int = Query(default=8, ge=1, le=50), since_hours: Optional[int] = Query(default=None, ge=1, le=24 * 365)):
    overview = _task_memory.memory_overview(limit_per_group=max(3, limit), since_hours=since_hours)
    items = (overview.get("projects") or [])[:limit]
    return {"items": items, "total": len(items)}


@router.get("/memory/workspaces")
def list_project_workspaces(limit: int = Query(default=20, ge=1, le=100), include_archived: bool = Query(default=False)):
    items = _task_memory.list_project_workspaces(limit=limit, include_archived=include_archived)
    active_workspace_id = _task_memory.get_active_workspace_id()
    return {"items": items, "active_workspace_id": active_workspace_id}


@router.post("/memory/workspaces")
def create_project_workspace(payload: WorkspaceCreateRequest):
    try:
        workspace = _task_memory.create_project_workspace(
            name=payload.name,
            description=payload.description,
            focus=payload.focus,
            color=payload.color,
            status=payload.status,
            memory_ids=payload.memory_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "workspace": workspace}


@router.get("/memory/workspaces/active")
def active_project_workspace():
    active_workspace_id = _task_memory.get_active_workspace_id()
    workspace = _task_memory.get_project_workspace(active_workspace_id) if active_workspace_id else None
    return {"active_workspace_id": active_workspace_id, "workspace": workspace}


@router.post("/memory/workspaces/active")
def set_active_project_workspace(payload: WorkspaceActivationRequest):
    try:
        result = _task_memory.set_active_workspace(payload.workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **result}


@router.get("/memory/workspaces/policy/current")
def get_current_workspace_policy():
    active_workspace_id = _task_memory.get_active_workspace_id()
    workspace = _task_memory.get_project_workspace(active_workspace_id) if active_workspace_id else None
    return {"workspace": workspace, "policy": _task_memory.get_workspace_policy(active_workspace_id)}


@router.get("/memory/workspaces/{workspace_id}/policy")
def get_workspace_policy_route(workspace_id: int):
    workspace = _task_memory.get_project_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return {"workspace": workspace, "policy": _task_memory.get_workspace_policy(workspace_id)}


@router.post("/memory/workspaces/{workspace_id}/policy")
def set_workspace_policy_route(workspace_id: int, payload: WorkspacePolicyRequest):
    workspace = _task_memory.get_project_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    policy = _task_memory.set_workspace_policy(
        workspace_id,
        browser_allowed=payload.browser_allowed,
        desktop_allowed=payload.desktop_allowed,
        shell_allowed=payload.shell_allowed,
        repo_write_allowed=payload.repo_write_allowed,
        require_confirmation=payload.require_confirmation,
    )
    return {"ok": True, "workspace": workspace, "policy": policy}


@router.get("/memory/workspaces/{workspace_id}")
def get_project_workspace(workspace_id: int):
    workspace = _task_memory.get_project_workspace(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return workspace


@router.post("/memory/workspaces/{workspace_id}")
def update_project_workspace(workspace_id: int, payload: WorkspaceUpdateRequest):
    ok = _task_memory.update_project_workspace(
        workspace_id,
        name=payload.name,
        description=payload.description,
        focus=payload.focus,
        color=payload.color,
        status=payload.status,
        memory_ids=payload.memory_ids,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="workspace not found")
    return {"ok": True, "workspace": _task_memory.get_project_workspace(workspace_id)}


@router.get("/memory/graph")
def memory_graph(workspace_id: Optional[int] = Query(default=None, ge=1), limit: int = Query(default=60, ge=2, le=200)):
    return _task_memory.workspace_memory_graph(workspace_id=workspace_id, limit=limit)


@router.get("/memory/reminders")
def proactive_reminders(
    limit: int = Query(default=20, ge=1, le=200),
    status: Optional[str] = Query(default=None, pattern="^(open|done|dismissed)$"),
    workspace_id: Optional[int] = Query(default=None, ge=1),
    due_within_hours: Optional[int] = Query(default=None, ge=1, le=24 * 30),
):
    return {
        "items": _task_memory.list_proactive_reminders(
            limit=limit,
            status=status,
            workspace_id=workspace_id,
            due_within_hours=due_within_hours,
        )
    }


@router.post("/memory/reminders")
def create_proactive_reminder(payload: ReminderCreateRequest):
    try:
        reminder = _task_memory.create_proactive_reminder(
            title=payload.title,
            content=payload.content,
            due_at=payload.due_at,
            priority=payload.priority,
            channel=payload.channel,
            workspace_id=payload.workspace_id,
            memory_id=payload.memory_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "reminder": reminder}


@router.post("/memory/reminders/generate")
def generate_proactive_reminders(workspace_id: Optional[int] = Query(default=None, ge=1), limit: int = Query(default=4, ge=1, le=10)):
    return _task_memory.generate_proactive_reminders(workspace_id=workspace_id, limit=limit)


@router.post("/memory/reminders/dispatch-due")
def dispatch_due_proactive_reminders(limit: int = Query(default=10, ge=1, le=50)):
    return _dispatch_due_reminders(limit=limit)


@router.get("/memory/reminders/voice-feed")
def proactive_reminder_voice_feed(limit: int = Query(default=10, ge=1, le=50)):
    return {"items": _dispatch_due_reminders(limit=limit, include_discord=False, include_voice=True).get("voice", [])}


@router.get("/trust/report")
def trust_report(limit: int = Query(default=100, ge=10, le=500), session_id: Optional[str] = Query(default=None)):
    if session_id:
        recent = _task_memory.list_tool_executions(limit=limit, session_id=session_id)
        return {
            "session_id": session_id,
            "total_runs": len(recent),
            "recent": recent,
        }
    return _task_memory.tool_reliability_report(limit=limit)


@router.get("/trust/receipts")
def trust_receipts(limit: int = Query(default=20, ge=1, le=100), session_id: Optional[str] = Query(default=None)):
    return _trust_receipts_payload(limit=limit, session_id=session_id)


@router.get("/next-action")
def next_action(
    workspace_id: Optional[int] = Query(default=None, ge=1),
    limit: int = Query(default=5, ge=1, le=12),
):
    return _task_memory.next_best_actions(workspace_id=workspace_id, limit=limit)


@router.post("/next-action/execute")
def execute_next_action(payload: NextActionExecuteRequest):
    action = payload.action if isinstance(payload.action, dict) else {}
    execution = action.get("execution") if isinstance(action.get("execution"), dict) else {}
    kind = str(execution.get("kind") or "").strip().lower()
    session_id = payload.session_id or str(uuid4())
    if kind == "activate_workspace":
        workspace_id = execution.get("workspace_id")
        if not isinstance(workspace_id, int):
            raise HTTPException(status_code=400, detail="workspace_id is required")
        result = _task_memory.set_active_workspace(workspace_id)
        return {"ok": True, "kind": kind, "session_id": session_id, "result": result}
    if kind == "reminder_done":
        reminder_id = execution.get("reminder_id")
        if not isinstance(reminder_id, int):
            raise HTTPException(status_code=400, detail="reminder_id is required")
        ok = _task_memory.update_proactive_reminder(reminder_id, status="done", delivered=True)
        reminder = _task_memory.get_proactive_reminder(reminder_id)
        return {"ok": ok, "kind": kind, "session_id": session_id, "result": reminder}
    if kind == "workspace_policy":
        workspace_id = execution.get("workspace_id")
        policy = _task_memory.get_workspace_policy(workspace_id if isinstance(workspace_id, int) else None)
        workspace = _task_memory.get_project_workspace(workspace_id) if isinstance(workspace_id, int) else None
        return {"ok": True, "kind": kind, "session_id": session_id, "result": {"workspace": workspace, "policy": policy}}
    message = str(execution.get("message") or action.get("action") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="action is not executable")
    response = agent_chat(AgentChatRequest(message=message, session_id=session_id))
    return {
        "ok": True,
        "kind": "chat",
        "session_id": session_id,
        "result": {
            "reply": response.reply,
            "tool_result": response.tool_result,
            "plan": response.plan,
        },
    }


@router.post("/autonomy/mission/run")
def run_autonomy_mission(payload: AutonomousMissionRequest):
    return _run_and_store_mission(
        workspace_id=payload.workspace_id,
        session_id=payload.session_id,
        limit=payload.limit,
        auto_approve=payload.auto_approve,
        retry_limit=payload.retry_limit,
        goal=None,
    )


@router.post("/autonomy/missions/start")
def autonomy_mission_start(payload: MissionStartRequest):
    return _run_and_store_mission(
        workspace_id=payload.workspace_id,
        session_id=payload.session_id,
        limit=payload.limit,
        auto_approve=payload.auto_approve,
        retry_limit=payload.retry_limit,
        goal=payload.goal,
    )


@router.get("/autonomy/missions")
def autonomy_mission_list(limit: int = Query(default=20, ge=1, le=100), workspace_id: Optional[int] = Query(default=None, ge=1)):
    return {"items": _task_memory.list_mission_runs(limit=limit, workspace_id=workspace_id)}


@router.get("/autonomy/missions/{mission_id}")
def autonomy_mission_get(mission_id: int):
    item = _task_memory.get_mission_run(mission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="mission not found")
    return {"mission": item}


@router.post("/autonomy/missions/{mission_id}/resume")
def autonomy_mission_resume(mission_id: int, approve: bool = Query(default=False)):
    item = _task_memory.get_mission_run(mission_id)
    if item is None:
        raise HTTPException(status_code=404, detail="mission not found")
    workspace_id = item.get("workspace_id")
    session_id = item.get("session_id")
    limit_count = int(item.get("limit_count") or 3)
    auto_approve = bool(item.get("auto_approve")) or bool(approve)
    goal = item.get("goal")
    prior_result = item.get("result") if isinstance(item.get("result"), dict) else {}
    retry_limit = int(prior_result.get("retry_limit") or prior_result.get("checkpoint", {}).get("retry_limit") or 1)
    return _run_and_store_mission(
        workspace_id=workspace_id if isinstance(workspace_id, int) else None,
        session_id=session_id if isinstance(session_id, str) else None,
        limit=limit_count,
        auto_approve=auto_approve,
        retry_limit=retry_limit,
        goal=str(goal).strip() if isinstance(goal, str) and goal.strip() else None,
        mission_id=mission_id,
    )


@router.get("/integrations/config")
def integrations_config_get():
    return _get_integration_config()


@router.post("/integrations/config")
def integrations_config_set(payload: IntegrationConfigRequest):
    return _set_integration_config(payload.model_dump())


@router.get("/integrations/summary")
def integrations_summary():
    cfg = _get_integration_config()
    connected = sum(1 for item in cfg.get("connections", {}).values() if bool(item.get("connected")))
    enabled = sum(1 for item in cfg.get("connections", {}).values() if bool(item.get("enabled")))
    return {
        "connections": cfg.get("connections", {}),
        "enabled_count": enabled,
        "connected_count": connected,
        "summary": f"{connected} connected of {enabled} enabled elite integrations.",
    }


@router.get("/integrations/intelligence")
def integrations_intelligence():
    return _integration_intelligence()


@router.post("/integrations/github/issues")
def integrations_github_issue_create(payload: GitHubIssueCreateRequest):
    return _github_issue_create(payload)


@router.post("/integrations/github/pulls/review")
def integrations_github_pull_review(payload: GitHubPullReviewRequest):
    return _github_pull_review(payload)


@router.post("/integrations/github/pulls/summary")
def integrations_github_pull_summary(payload: GitHubPullSummaryRequest):
    return _github_pull_summary(payload)


@router.get("/integrations/calendar/events")
def integrations_calendar_events(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": _calendar_events_get()[:limit]}


@router.post("/integrations/calendar/events")
def integrations_calendar_event_create(payload: CalendarEventCreateRequest):
    return _calendar_event_create(payload)


@router.post("/integrations/email/send")
def integrations_email_send(payload: EmailSendRequest):
    return _email_action_send(payload)


@router.get("/desktop/presence")
def desktop_presence_get(workspace_id: Optional[int] = Query(default=None, ge=1)):
    return _desktop_presence_payload(workspace_id=workspace_id)


@router.get("/desktop/awareness")
def desktop_awareness_get(workspace_id: Optional[int] = Query(default=None, ge=1)):
    return _desktop_awareness_payload(workspace_id=workspace_id)


@router.post("/desktop/presence")
def desktop_presence_set(payload: DesktopPresenceSnapshotRequest):
    workspace_id = payload.workspace_id if isinstance(payload.workspace_id, int) else None
    details = dict(payload.details or {})
    if payload.app_name:
        details.setdefault("app_name", payload.app_name)
    if payload.window_title:
        details.setdefault("window_title", payload.window_title)
    snapshot = _task_memory.save_desktop_presence_snapshot(
        workspace_id=workspace_id,
        app_name=payload.app_name,
        window_title=payload.window_title,
        summary=payload.summary or "Desktop presence captured",
        details=details,
    )
    return {"ok": True, "snapshot": snapshot, "presence": _desktop_presence_payload(workspace_id=workspace_id)}


@router.get("/voice/wake")
def voice_wakeword_get():
    return _get_wakeword_config()


@router.post("/voice/wake")
def voice_wakeword_set(payload: VoiceWakeConfigRequest):
    return _set_wakeword_config(payload.model_dump())


@router.post("/voice/wake/detect")
def voice_wakeword_detect(payload: VoiceWakeDetectRequest):
    return _wakeword_detect(payload)


@router.get("/voice/local")
def voice_local_get():
    return _get_local_voice_config()


@router.get("/voice/local/presets")
def voice_local_presets():
    return {"items": _local_voice_presets(), "config": _get_local_voice_config()}


@router.post("/voice/local")
def voice_local_set(payload: LocalVoiceConfigRequest):
    return _set_local_voice_config(payload.model_dump())


@router.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="audio file is required")
    return _transcribe_audio_local(data, filename=file.filename or "voice.wav")


@router.post("/voice/speak")
def voice_speak(payload: VoiceSpeakRequest):
    audio = _synthesize_speech_local(payload.text)
    return Response(content=audio, media_type="audio/wav")


@router.get("/control/browser/templates")
def browser_workflow_templates():
    return {"items": _get_browser_workflow_templates()}


@router.get("/control/browser/templates/auth")
def browser_workflow_auth_templates():
    return {"items": _auth_browser_workflow_templates()}


@router.get("/control/browser/templates/library")
def browser_workflow_library_templates(provider: Optional[str] = Query(default=None, max_length=80)):
    return {"items": _workflow_library_templates(provider=provider)}


@router.get("/control/browser/sessions")
def browser_sessions(limit: int = Query(default=20, ge=1, le=100), workspace_id: Optional[int] = Query(default=None, ge=1)):
    return {"items": _task_memory.list_browser_sessions(limit=limit, workspace_id=workspace_id)}


@router.post("/control/browser/sessions")
def browser_sessions_save(payload: BrowserSessionCreateRequest):
    try:
        item = _task_memory.save_browser_session(
            name=payload.name,
            storage_state=payload.storage_state,
            workspace_id=payload.workspace_id,
            notes=payload.notes,
            provider=payload.provider,
            template_name=payload.template_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "session": item}


@router.delete("/control/browser/sessions")
def browser_sessions_delete(name: Optional[str] = Query(default=None), session_id: Optional[int] = Query(default=None, ge=1)):
    target = session_id if session_id is not None else name
    if target in {None, ""}:
        raise HTTPException(status_code=400, detail="name or session_id is required")
    if not _task_memory.delete_browser_session(target):
        raise HTTPException(status_code=404, detail="browser session not found")
    return {"ok": True, "items": _task_memory.list_browser_sessions(limit=50)}


@router.post("/control/browser/sessions/health")
def browser_sessions_health(payload: BrowserSessionHealthCheckRequest):
    sessions: List[Dict[str, Any]] = []
    if payload.session_id or payload.session_name:
        target = payload.session_id if payload.session_id is not None else payload.session_name
        session = _task_memory.get_browser_session(target)
        if session is None:
            raise HTTPException(status_code=404, detail="browser session not found")
        sessions = [session]
    else:
        sessions = _task_memory.list_browser_sessions(limit=payload.limit, workspace_id=payload.workspace_id)
    items: List[Dict[str, Any]] = []
    for session in sessions:
        try:
            template = _browser_template_for_session(session)
            result = _session_health_from_run(session, template)
            updated = _task_memory.update_browser_session_health(session["id"], status=result["status"], details=result["details"])
            items.append(updated or session)
        except Exception as exc:
            updated = _task_memory.update_browser_session_health(
                session["id"],
                status="error",
                details={"error": str(exc), "template_name": (template or {}).get("name") if 'template' in locals() else None},
            )
            items.append(updated or session)
    return {"ok": True, "items": items}


@router.post("/control/browser/templates")
def browser_workflow_templates_save(payload: BrowserWorkflowTemplateRequest):
    items = [item for item in _get_browser_workflow_templates() if str(item.get("name") or "").strip().lower() != payload.name.strip().lower()]
    items.insert(
        0,
        {
            "name": payload.name.strip(),
            "description": payload.description.strip(),
            "start_url": payload.start_url,
            "category": payload.category.strip() or "custom",
            "auth_template": bool(payload.auth_template),
            "recommended_session_name": (payload.recommended_session_name.strip() if isinstance(payload.recommended_session_name, str) else None) or None,
            "provider": (payload.provider.strip() if isinstance(payload.provider, str) else None) or None,
            "healthcheck_url": (payload.healthcheck_url.strip() if isinstance(payload.healthcheck_url, str) else None) or None,
            "healthcheck_selector": (payload.healthcheck_selector.strip() if isinstance(payload.healthcheck_selector, str) else None) or None,
            "logged_out_markers": [str(v).strip() for v in payload.logged_out_markers if str(v).strip()][:12],
            "healthy_markers": [str(v).strip() for v in payload.healthy_markers if str(v).strip()][:12],
            "steps": [step.model_dump() for step in payload.steps],
        },
    )
    return {"ok": True, "items": _set_browser_workflow_templates(items)}


@router.delete("/control/browser/templates")
def browser_workflow_templates_delete(name: str = Query(..., min_length=1, max_length=80)):
    items = [item for item in _get_browser_workflow_templates() if str(item.get("name") or "").strip().lower() != name.strip().lower()]
    return {"ok": True, "items": _set_browser_workflow_templates(items)}


@router.post("/memory/reminders/{reminder_id}")
def update_proactive_reminder(reminder_id: int, payload: ReminderUpdateRequest):
    ok = _task_memory.update_proactive_reminder(
        reminder_id,
        status=payload.status,
        due_at=payload.due_at,
        delivered=payload.delivered,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="reminder not found")
    return {"ok": True, "reminder": _task_memory.get_proactive_reminder(reminder_id)}


@router.get("/memory/briefing")
def memory_briefing(
    period: str = Query(default="morning", pattern="^(morning|evening)$"),
    recent_project_hours: Optional[int] = Query(default=None, ge=1, le=24 * 30),
):
    return _task_memory.memory_briefing(period=period, recent_project_hours=recent_project_hours)


@router.post("/memory/consolidate")
def consolidate_long_term_memories(limit: int = Query(default=200, ge=10, le=1000)):
    return _task_memory.consolidate_long_term_memories(limit=limit)


@router.post("/memory/{memory_id}")
def update_long_term_memory(memory_id: int, payload: LongTermMemoryUpdateRequest):
    ok = _task_memory.update_long_term_memory(
        memory_id,
        content=payload.content,
        tags=payload.tags,
        importance=payload.importance,
        memory_type=payload.memory_type,
        subject=payload.subject,
        pinned=payload.pinned,
        archived=payload.archived,
        lane=payload.lane,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True, "id": memory_id}


@router.delete("/memory/{memory_id}")
def delete_long_term_memory(memory_id: int):
    ok = _task_memory.archive_long_term_memory(memory_id, archived=True)
    if not ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True, "id": memory_id, "archived": True}


@router.post("/memory/{memory_id}/pin")
def pin_long_term_memory(memory_id: int, pinned: bool = Query(default=True)):
    ok = _task_memory.update_long_term_memory(memory_id, pinned=pinned)
    if not ok:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True, "id": memory_id, "pinned": pinned}


@router.post("/memory/actions/bulk")
def bulk_memory_action(payload: LongTermMemoryBulkActionRequest):
    return _task_memory.bulk_update_long_term_memories(payload.ids, action=payload.action)


@router.post("/memory/pinned/reorder")
def reorder_pinned_memories(payload: LongTermMemoryPinOrderRequest):
    return _task_memory.reorder_pinned_memories(payload.ids)


@router.post("/memory/projects/reorder")
def reorder_project_memories(payload: LongTermMemoryProjectOrderRequest):
    return _task_memory.reorder_project_memories(payload.ids)


@router.get("/memory/filters/saved")
def memory_saved_filters_get():
    return {"items": _memory_saved_filters_get()}


@router.post("/memory/filters/saved")
def memory_saved_filters_add(payload: MemorySavedFilterRequest):
    items = [item for item in _memory_saved_filters_get() if str(item.get("name") or "").strip().lower() != payload.name.strip().lower()]
    items.insert(
        0,
        {
            "name": payload.name.strip(),
            "query": payload.query.strip(),
            "tag": payload.tag.strip(),
            "memory_type": payload.memory_type.strip().lower(),
        },
    )
    return {"ok": True, "items": _memory_saved_filters_set(items)}


@router.delete("/memory/filters/saved")
def memory_saved_filters_delete(name: str = Query(..., min_length=1, max_length=60)):
    items = [item for item in _memory_saved_filters_get() if str(item.get("name") or "").strip().lower() != name.strip().lower()]
    return {"ok": True, "items": _memory_saved_filters_set(items)}


@router.get("/memory/briefings/automations")
def list_memory_briefing_automations():
    return {"items": _task_memory.list_autonomous_jobs(limit=50, mode="briefing")}


@router.post("/memory/briefings/automations")
def ensure_memory_briefing_automations(payload: MemoryBriefingAutomationRequest):
    items = _task_memory.ensure_daily_briefing_automations(
        session_id=payload.session_id,
        timezone_name=payload.timezone_name,
        morning_hour=payload.morning_hour,
        evening_hour=payload.evening_hour,
    )
    return {"ok": True, "items": items}


@router.get("/memory/briefings/delivery")
def memory_briefing_delivery_get():
    return _get_briefing_delivery_config()


@router.post("/memory/briefings/delivery")
def memory_briefing_delivery_set(payload: MemoryBriefingDeliveryConfigRequest):
    cfg = _set_briefing_delivery_config(payload.model_dump())
    return cfg


@router.post("/memory/briefings/delivery/test")
def memory_briefing_delivery_test(period: str = Query(default="morning", pattern="^(morning|evening)$")):
    briefing = _task_memory.memory_briefing(period=period)
    delivery = _dispatch_briefing_deliveries(briefing)
    return {"ok": True, "briefing": briefing, "delivery": delivery}


@router.get("/control/config")
def agent_control_config_get():
    cfg = _get_control_config()
    cfg["browser_runtime"] = _browser_workflow_runtime()
    return cfg


@router.post("/control/config")
def agent_control_config_set(payload: AgentControlConfigRequest):
    cfg = _set_control_config(payload.model_dump())
    cfg["browser_runtime"] = _browser_workflow_runtime()
    return cfg


@router.post("/control/browser/open")
def control_browser_open(payload: BrowserOpenRequest):
    resolved_workspace_id, workspace, policy = _resolve_workspace_context(payload.workspace_id)
    result = _browser_open(payload.url, workspace_id=payload.workspace_id, confirm=payload.confirm)
    verification = _record_verified_tool_run(tool_name="browser_open", result=result, args=payload.model_dump())
    return {"ok": True, "workspace": workspace, "workspace_id": resolved_workspace_id, "policy": policy, "result": result, "verification": verification}


@router.post("/control/browser/search")
def control_browser_search(payload: BrowserSearchRequest):
    resolved_workspace_id, workspace, policy = _resolve_workspace_context(payload.workspace_id)
    result = _browser_search(payload.query, workspace_id=payload.workspace_id, confirm=payload.confirm)
    verification = _record_verified_tool_run(tool_name="browser_search", result=result, args=payload.model_dump())
    return {"ok": True, "workspace": workspace, "workspace_id": resolved_workspace_id, "policy": policy, "result": result, "verification": verification}


@router.post("/control/browser/workflow")
def control_browser_workflow(payload: BrowserWorkflowRequest):
    resolved_workspace_id, workspace, policy = _enforce_workspace_capability("browser", workspace_id=payload.workspace_id)
    loaded_session = None
    storage_state = None
    template = _browser_template_by_name(payload.template_name)
    if payload.session_name:
        loaded_session = _task_memory.get_browser_session(payload.session_name)
        if loaded_session is None and not payload.save_session:
            raise HTTPException(status_code=404, detail="browser session not found")
        storage_state = loaded_session.get("storage_state") if loaded_session else None
    if payload.save_session and not payload.session_name:
        raise HTTPException(status_code=400, detail="session_name is required when save_session=true")
    steps_payload = [step.model_dump() for step in payload.steps]
    if not steps_payload and template:
        steps_payload = [dict(step) for step in list(template.get("steps") or []) if isinstance(step, dict)]
    if not steps_payload:
        raise HTTPException(status_code=400, detail="steps or template_name with stored steps is required")
    start_url = payload.start_url or (template.get("start_url") if isinstance(template, dict) else None)
    if _policy_confirmation_required(policy, confirm=payload.confirm):
        result = {
            "ok": True,
            "executed": False,
            "approval_required": True,
            "reason": "Workspace policy requires confirm=true before browser workflows execute",
            "workspace_policy": policy,
            "results": [],
            "extracted": [],
            "runtime": _browser_workflow_runtime(),
        }
        verification = _record_verified_tool_run(tool_name="browser_workflow", result=result, args=payload.model_dump())
        return {
            "ok": True,
            "workspace": workspace,
            "workspace_id": resolved_workspace_id,
            "policy": policy,
            "loaded_session": loaded_session,
            "result": result,
            "verification": verification,
        }
    result = _run_browser_workflow(
        start_url=start_url,
        steps=steps_payload,
        headless=payload.headless,
        storage_state=storage_state if isinstance(storage_state, dict) else None,
        capture_storage_state=bool(payload.save_session),
    )
    saved_session = None
    if payload.save_session and payload.session_name:
        provider = str((template or {}).get("provider") or (loaded_session or {}).get("provider") or "").strip() or None
        template_name = str((template or {}).get("name") or payload.template_name or (loaded_session or {}).get("template_name") or "").strip() or None
        saved_session = _task_memory.save_browser_session(
            name=payload.session_name,
            storage_state=result.get("storage_state") if isinstance(result.get("storage_state"), dict) else {},
            workspace_id=resolved_workspace_id,
            notes=payload.session_notes or (loaded_session.get("notes") if loaded_session else None),
            provider=provider,
            template_name=template_name,
        )
    if payload.session_name:
        _task_memory.touch_browser_session(payload.session_name)
    verification = _record_verified_tool_run(
        tool_name="browser_workflow",
        result=result,
        args={
            "start_url": start_url,
            "steps": steps_payload,
            "headless": payload.headless,
            "workspace_id": payload.workspace_id,
            "session_name": payload.session_name,
            "save_session": payload.save_session,
            "template_name": payload.template_name,
        },
    )
    return {
        "ok": True,
        "workspace": workspace,
        "workspace_id": resolved_workspace_id,
        "policy": policy,
        "loaded_session": loaded_session,
        "saved_session": saved_session,
        "result": result,
        "verification": verification,
    }


@router.post("/control/desktop/launch")
def control_desktop_launch(payload: DesktopLaunchRequest):
    resolved_workspace_id, workspace, policy = _resolve_workspace_context(payload.workspace_id)
    result = _desktop_launch(payload.app, workspace_id=payload.workspace_id, confirm=payload.confirm)
    verification = _record_verified_tool_run(tool_name="desktop_launch", result=result, args=payload.model_dump())
    return {"ok": True, "workspace": workspace, "workspace_id": resolved_workspace_id, "policy": policy, "result": result, "verification": verification}


@router.post("/control/desktop/action")
def control_desktop_action(payload: DesktopControlRequest):
    resolved_workspace_id, workspace, policy = _resolve_workspace_context(payload.workspace_id)
    result = _desktop_control(
        action=payload.action,
        target=payload.target,
        text=payload.text,
        keys=payload.keys,
        workspace_id=payload.workspace_id,
        confirm=payload.confirm,
    )
    verification = _record_verified_tool_run(tool_name="desktop_control", result=result, args=payload.model_dump())
    return {"ok": True, "workspace": workspace, "workspace_id": resolved_workspace_id, "policy": policy, "result": result, "verification": verification}


@router.post("/vision/analyze")
async def vision_analyze(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    source: str = Form(default="screen_capture"),
    prompt: Optional[str] = Form(default=None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="image file is empty")
    analysis = _vision_summary_from_image(data, prompt=prompt)
    obs_id = _task_memory.add_vision_observation(
        summary=analysis["summary"],
        details=analysis["details"],
        source=source,
        session_id=session_id,
    )
    return {"ok": True, "id": obs_id, **analysis}


@router.get("/vision/observations")
def list_vision_observations(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": _task_memory.list_vision_observations(limit=limit)}


@router.post("/autonomy/jobs")
def create_autonomous_job(payload: AutonomousJobCreateRequest):
    job_id = _task_memory.create_autonomous_job(
        name=payload.name,
        goal=payload.goal,
        mode=payload.mode,
        interval_minutes=payload.interval_minutes,
        session_id=payload.session_id,
        auto_approve=payload.auto_approve,
        enabled=payload.enabled,
        schedule_type=payload.schedule_type,
        run_hour=payload.run_hour,
        run_minute=payload.run_minute,
        timezone_name=payload.timezone_name,
        metadata=payload.metadata,
    )
    return {"ok": True, "id": job_id}


@router.get("/autonomy/jobs")
def list_autonomy_jobs(limit: int = Query(default=100, ge=1, le=200), mode: Optional[str] = Query(default=None)):
    return {"items": _task_memory.list_autonomous_jobs(limit=limit, mode=mode)}


@router.get("/autonomy/watchers")
def list_project_watchers(limit: int = Query(default=50, ge=1, le=200)):
    return {"items": _task_memory.list_autonomous_jobs(limit=limit, mode="watcher")}


@router.get("/autonomy/watchers/network")
def watcher_network():
    return _watcher_network_payload()


@router.post("/autonomy/watchers")
def ensure_project_watcher(payload: ProjectWatcherRequest):
    watcher_type = str(payload.watcher_type or "project").strip().lower() or "project"
    workspace = _task_memory.get_project_workspace(payload.workspace_id) if payload.workspace_id else None
    if watcher_type == "project" and workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    scope_name = workspace.get("name") if workspace else watcher_type.title()
    name = f"Watcher: {scope_name} [{watcher_type}]"
    existing = None
    for job in _task_memory.list_autonomous_jobs(limit=200, mode="watcher"):
        meta = job.get("metadata") or {}
        same_workspace = int(meta.get("workspace_id") or 0) == int(payload.workspace_id or 0)
        same_type = str(meta.get("watcher_type") or "project").strip().lower() == watcher_type
        if same_workspace and same_type:
            existing = job
            break
    metadata = {
        "workspace_id": int(payload.workspace_id) if payload.workspace_id else None,
        "min_score": float(payload.min_score),
        "limit": 3,
        "watcher_type": watcher_type,
    }
    goal = (
        f"Monitor workspace {scope_name} and trigger missions when {watcher_type} signals need action."
        if workspace else
        f"Monitor {watcher_type} signals and escalate when they need action."
    )
    if existing:
        _task_memory.update_autonomous_job(
            int(existing["id"]),
            name=name,
            goal=goal,
            enabled=payload.enabled,
            interval_minutes=payload.interval_minutes,
            auto_approve=payload.auto_approve,
            mode="watcher",
            metadata=metadata,
        )
        job_id = int(existing["id"])
    else:
        job_id = _task_memory.create_autonomous_job(
            name=name,
            goal=goal,
            mode="watcher",
            interval_minutes=payload.interval_minutes,
            session_id=payload.session_id,
            auto_approve=payload.auto_approve,
            enabled=payload.enabled,
            metadata=metadata,
        )
    job = next((item for item in _task_memory.list_autonomous_jobs(limit=200, mode="watcher") if int(item["id"]) == job_id), None)
    return {"ok": True, "job": job}


@router.post("/autonomy/jobs/{job_id}")
def update_autonomy_job(job_id: int, payload: AutonomousJobUpdateRequest):
    ok = _task_memory.update_autonomous_job(
        job_id,
        enabled=payload.enabled,
        interval_minutes=payload.interval_minutes,
        auto_approve=payload.auto_approve,
        goal=payload.goal.strip() if isinstance(payload.goal, str) else None,
        name=payload.name.strip() if isinstance(payload.name, str) else None,
        mode=payload.mode,
        schedule_type=payload.schedule_type,
        run_hour=payload.run_hour,
        run_minute=payload.run_minute,
        timezone_name=payload.timezone_name,
        metadata=payload.metadata,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="autonomous job not found")
    return {"ok": True, "id": job_id}


@router.post("/autonomy/jobs/{job_id}/run")
def run_autonomy_job(job_id: int):
    jobs = _task_memory.list_autonomous_jobs(limit=500)
    target = next((j for j in jobs if int(j["id"]) == int(job_id)), None)
    if target is None:
        raise HTTPException(status_code=404, detail="autonomous job not found")
    result = _execute_autonomous_job(target)
    _task_memory.mark_autonomous_job_result(job_id, ok=bool(result.get("ok", True)), error=None, result=result)
    return {"ok": True, "job_id": job_id, "result": result}


@router.post("/benchmark/llm")
def benchmark_llm(payload: AgentLlmBenchmarkRequest):
    runtime_before = _effective_ollama_runtime()
    if payload.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "provider": (os.getenv("LLM_PROVIDER", "").strip().lower() or "auto"),
            "prompts": [p[:160] for p in payload.prompts],
            "runs_per_prompt": payload.runs_per_prompt,
            "runtime": runtime_before,
        }

    mode = _set_llm_mode(payload.mode)
    if payload.model is not None:
        _set_llm_model_override(payload.model)
    runtime = _effective_ollama_runtime()
    model_name = str(runtime.get("model") or ollama_model())
    timings_ms: List[int] = []
    per_prompt: List[Dict[str, Any]] = []

    for prompt in payload.prompts:
        prompt = str(prompt or "").strip()
        if not prompt:
            continue
        runs: List[Dict[str, Any]] = []
        for _ in range(payload.runs_per_prompt):
            t0 = time.perf_counter()
            ok = True
            err = None
            out = ""
            try:
                out = ollama_chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system="You are concise and factual.",
                    temperature=float(runtime.get("temperature") or 0.2),
                    timeout_s=payload.timeout_s,
                )
            except Exception as exc:
                ok = False
                err = str(exc)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            timings_ms.append(elapsed_ms)
            runs.append(
                {
                    "ok": ok,
                    "latency_ms": elapsed_ms,
                    "output_tokens_est": _token_estimate(out) if ok else 0,
                    "error": err,
                }
            )
        per_prompt.append({"prompt": prompt[:300], "runs": runs})

    successes = [r for pp in per_prompt for r in pp["runs"] if r.get("ok")]
    success_rate = (len(successes) / max(1, sum(len(pp["runs"]) for pp in per_prompt))) * 100.0
    avg_latency = (sum(timings_ms) / max(1, len(timings_ms))) if timings_ms else 0.0
    p95 = sorted(timings_ms)[max(0, int(0.95 * max(1, len(timings_ms))) - 1)] if timings_ms else 0
    return {
        "ok": True,
        "mode": mode,
        "model": model_name,
        "runs_total": sum(len(pp["runs"]) for pp in per_prompt),
        "success_rate_pct": round(success_rate, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": int(p95),
        "details": per_prompt,
        "timestamp": _now_iso(),
    }


@router.post("/evals/multi-agent")
def eval_multi_agent(payload: AgentEvalRequest):
    if payload.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "task": payload.task,
            "runs": payload.runs,
            "mode": payload.mode,
            "model": payload.model,
            "timestamp": _now_iso(),
        }

    _set_llm_mode(payload.mode)
    if payload.model is not None:
        _set_llm_model_override(payload.model)

    per_run: List[Dict[str, Any]] = []
    for idx in range(payload.runs):
        t0 = time.perf_counter()
        result = run_multi_agent(
            MultiAgentRunRequest(
                task=payload.task,
                session_id=f"eval-{idx + 1}-{uuid4()}",
            )
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        score = _score_multi_agent_run(result=result, latency_ms=elapsed_ms)
        per_run.append({"run": idx + 1, "score": score, "result": result})

    overall_scores = [float(r["score"]["overall_score_pct"]) for r in per_run]
    avg_overall = sum(overall_scores) / max(1, len(overall_scores))
    avg_latency = sum(int(r["score"]["latency_ms"]) for r in per_run) / max(1, len(per_run))
    return {
        "ok": True,
        "runs": payload.runs,
        "task": payload.task,
        "average_overall_score_pct": round(avg_overall, 1),
        "average_latency_ms": round(avg_latency, 1),
        "results": per_run,
        "timestamp": _now_iso(),
    }


@router.post("/chat/stream")
async def agent_chat_stream(payload: AgentChatRequest):
    session_id = payload.session_id or str(uuid4())
    plan = _build_plan(payload.message)
    _memory.append(session_id, StoredMessage(role="user", text=payload.message))

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if is_openai_configured():
            provider = "openai"
        else:
            provider = "basic"

    async def event_gen() -> AsyncIterator[str]:
        full_text = ""
        try:
            if provider == "ollama":
                runtime = _effective_ollama_runtime()
                history = _memory.load(session_id, max_messages=12)
                system = (
                    "You are Jarvis, concise and helpful.\n"
                    "For streaming chat mode, answer directly in plain text.\n"
                )
                chunks = ollama_chat_stream(
                    model=str(runtime.get("model") or ollama_model()),
                    messages=[{"role": m.role, "content": m.text} for m in history],
                    system=system,
                    temperature=float(runtime.get("temperature") or 0.2),
                    timeout_s=int(os.getenv("OLLAMA_TIMEOUT_S", "120")),
                )
                for chunk in chunks:
                    full_text += chunk
                    yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0)
            else:
                full_text = _basic_brain(payload.message)
                for part in full_text.split(" "):
                    chunk = part + " "
                    yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0)

            final_reply = full_text.strip() or _basic_brain(payload.message)
            _memory.append(session_id, StoredMessage(role="assistant", text=final_reply))
            done = {
                "session_id": session_id,
                "reply": final_reply,
                "timestamp": _now_iso(),
                "plan": plan,
            }
            yield f"event: done\ndata: {json.dumps(done)}\n\n"
        except Exception as exc:
            detail = str(exc).strip()
            if provider == "ollama":
                if detail.startswith("Ollama "):
                    fallback_reply = f"{detail} Falling back to basic mode.\n\n{_basic_brain(payload.message)}"
                else:
                    fallback_reply = (
                        f"Ollama streaming failed; falling back to basic mode. Error: {detail}\n\n"
                        f"{_basic_brain(payload.message)}"
                    )
                emitted = fallback_reply if not full_text else ("\n\n" + fallback_reply)
                full_text += emitted
                for part in emitted.split(" "):
                    chunk = part + " "
                    yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
                    await asyncio.sleep(0)
                final_reply = full_text.strip()
                _memory.append(session_id, StoredMessage(role="assistant", text=final_reply))
                done = {
                    "session_id": session_id,
                    "reply": final_reply,
                    "timestamp": _now_iso(),
                    "plan": plan,
                }
                yield f"event: done\ndata: {json.dumps(done)}\n\n"
                return

            err_text = f"Streaming failed: {detail}"
            _memory.append(session_id, StoredMessage(role="assistant", text=err_text))
            yield f"event: error\ndata: {json.dumps({'error': err_text, 'session_id': session_id, 'plan': plan})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(payload: AgentChatRequest):
    session_id = payload.session_id or str(uuid4())
    plan = _build_plan(payload.message)

    # Always store the raw user message first.
    _memory.append(session_id, StoredMessage(role="user", text=payload.message))

    msg_trimmed = payload.message.strip()
    msg_lower = msg_trimmed.lower()

    if _message_is_approval(msg_trimmed):
        reply, tool_result = _execute_pending_approval(session_id)
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=tool_result,
            timestamp=_now_iso(),
            plan=plan,
        )

    if _message_is_rejection(msg_trimmed):
        cleared = _clear_pending_approval(session_id)
        reply = "Pending approval cleared." if cleared else "There was no pending approval to clear."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"cleared": cleared},
            timestamp=_now_iso(),
            plan=plan,
        )

    memory_candidate = _extract_memory_candidate(msg_trimmed)
    if memory_candidate is not None and memory_candidate.get("content"):
        memory_id = _task_memory.create_long_term_memory(
            content=str(memory_candidate["content"]),
            tags=list(memory_candidate.get("tags") or []),
            importance=int(memory_candidate.get("importance") or 3),
            memory_type=str(memory_candidate.get("memory_type") or "general"),
            subject=(str(memory_candidate.get("subject")).strip() if memory_candidate.get("subject") else None),
            source=str(memory_candidate.get("source") or "chat"),
            session_id=session_id,
        )
        reply = f"I'll remember that. Saved as memory #{memory_id}."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"memory_id": memory_id},
            timestamp=_now_iso(),
            plan=plan,
        )

    if "who am i" in msg_lower:
        overview = _task_memory.memory_overview(limit_per_group=6)
        profile_items = overview.get("profile") or []
        if not profile_items:
            reply = "I don't have enough profile memory saved yet to answer that confidently."
            tool_result = {"items": []}
        else:
            reply = _format_memory_group("What I know about you", profile_items, limit=6)
            tool_result = {"items": profile_items[:6]}
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=tool_result,
            timestamp=_now_iso(),
            plan=plan,
        )

    workspace_switch = re.search(r"\b(?:switch|set|activate)\s+workspace(?:\s+to)?\s+(.+)$", msg_trimmed, flags=re.IGNORECASE)
    if workspace_switch:
        requested = workspace_switch.group(1).strip().lower()
        workspaces = _task_memory.list_project_workspaces(limit=100, include_archived=False)
        target = next((w for w in workspaces if str(w.get("name") or "").strip().lower() == requested), None)
        if target is None:
            target = next((w for w in workspaces if requested in str(w.get("name") or "").strip().lower()), None)
        if target is None:
            reply = f"I couldn't find a workspace named {workspace_switch.group(1).strip()}."
            tool_result = {"items": workspaces}
        else:
            result = _task_memory.set_active_workspace(int(target["id"]))
            workspace = result.get("workspace")
            reply = f"Active workspace switched to {workspace.get('name')}." + (f" Focus: {workspace.get('focus')}." if workspace and workspace.get("focus") else "")
            tool_result = result
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=tool_result,
            timestamp=_now_iso(),
            plan=plan,
        )

    if "what workspace" in msg_lower or "active workspace" in msg_lower or "current workspace" in msg_lower:
        active_workspace_id = _task_memory.get_active_workspace_id()
        workspace = _task_memory.get_project_workspace(active_workspace_id) if active_workspace_id else None
        if not workspace:
            reply = "No active workspace is selected yet."
            tool_result = {"workspace": None}
        else:
            reply = f"Active workspace: {workspace.get('name')}." + (f" Focus: {workspace.get('focus')}." if workspace.get("focus") else "")
            tool_result = {"workspace": workspace}
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=tool_result,
            timestamp=_now_iso(),
            plan=plan,
        )

    if "what project" in msg_lower or "what are my projects" in msg_lower or "what am i working on" in msg_lower:
        overview = _task_memory.memory_overview(limit_per_group=6)
        project_items = overview.get("projects") or []
        active_workspace_id = _task_memory.get_active_workspace_id()
        active_workspace = _task_memory.get_project_workspace(active_workspace_id) if active_workspace_id else None
        if not project_items:
            if active_workspace:
                reply = (
                    f"Your active workspace is {active_workspace.get('name')}."
                    + (f" Focus: {active_workspace.get('focus')}." if active_workspace.get("focus") else "")
                )
                tool_result = {"items": [], "workspace": active_workspace}
            else:
                reply = "I don't have any active project memories saved yet."
                tool_result = {"items": []}
        else:
            reply = _format_memory_group("Active project memory", project_items, limit=6)
            if active_workspace:
                reply += (
                    "\n\nActive workspace:\n"
                    + f"- {active_workspace.get('name')}"
                    + (f"\n- Focus: {active_workspace.get('focus')}" if active_workspace.get("focus") else "")
                )
            tool_result = {"items": project_items[:6], "workspace": active_workspace}
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=tool_result,
            timestamp=_now_iso(),
            plan=plan,
        )

    if "what do you remember" in msg_lower or "what do you know about me" in msg_lower:
        overview = _task_memory.memory_overview(limit_per_group=6)
        matches = _task_memory.list_long_term_memories(limit=8)
        if not matches:
            reply = "I don't have any long-term memories saved yet."
        else:
            reply = "\n\n".join(
                [
                    _format_memory_group("Profile memory", overview.get("profile") or [], limit=4),
                    _format_memory_group("Project memory", overview.get("projects") or [], limit=4),
                    _format_memory_group("Most important memories", overview.get("important") or [], limit=4),
                ]
            )
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"items": matches[:8], "overview": overview},
            timestamp=_now_iso(),
            plan=plan,
        )
    if msg_lower.startswith("todo:") or msg_lower.startswith("task:"):
        task_text = msg_trimmed.split(":", 1)[1].strip()
        if task_text:
            task_id = _task_memory.create_task(task_text, session_id=session_id)
            reply = f"Task captured as #{task_id}: {task_text}"
            _memory.append(session_id, StoredMessage(role="assistant", text=reply))
            return AgentChatResponse(
                session_id=session_id,
                reply=reply,
                tool_result={"task_id": task_id, "status": "open"},
                timestamp=_now_iso(),
                plan=plan,
            )

    parsed = _parse_tool_message(payload.message)
    if parsed is not None:
        tool_name, args = parsed
        tool = _TOOLS.get(tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        result, verification, pending = _execute_tool_with_approval(
            tool_name=tool_name,
            args=args,
            session_id=session_id,
            label=f"Tool `{tool_name}`",
        )
        reply = _approval_prompt_text(pending) if pending else f"Tool `{tool_name}` executed."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"tool": tool_name, "result": result, "verification": verification, "pending_approval": pending},
            timestamp=_now_iso(),
            plan=plan,
        )

    # Heuristic tool routing for common requests (helps local models that ignore tool directives).
    if ("database" in msg_lower and ("connect" in msg_lower or "connectivity" in msg_lower)) or (
        "db" in msg_lower and "ping" in msg_lower
    ):
        result = _TOOLS["db_ping"].handler({})
        verification = _record_verified_tool_run(tool_name="db_ping", result=result, args={}, session_id=session_id)
        reply = "Database connectivity check complete."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"tool": "db_ping", "result": result, "verification": verification},
            timestamp=_now_iso(),
            plan=plan,
        )

    if "time" in msg_lower and "tool" not in msg_lower:
        result = _TOOLS["get_time"].handler({})
        verification = _record_verified_tool_run(tool_name="get_time", result=result, args={}, session_id=session_id)
        reply = "Current time fetched."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"tool": "get_time", "result": result, "verification": verification},
            timestamp=_now_iso(),
            plan=plan,
        )

    if ("list files" in msg_lower or "show files" in msg_lower or "list the files" in msg_lower) and "tool" not in msg_lower:
        # Try to extract a target path like: "list files under data" or "list files in models"
        m = re.search(r"\b(?:in|under|from)\s+([^\s]+)", msg_lower)
        target = m.group(1) if m else "data"
        result = _TOOLS["list_files"].handler({"path": target, "recursive": False, "max_entries": 200})
        verification = _record_verified_tool_run(tool_name="list_files", result=result, args={"path": target, "recursive": False, "max_entries": 200}, session_id=session_id)
        reply = f"Listing files under `{target}`."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"tool": "list_files", "result": result, "verification": verification},
            timestamp=_now_iso(),
            plan=plan,
        )

    if ("system info" in msg_lower or "system status" in msg_lower) and "tool" not in msg_lower:
        result = _TOOLS["system_info"].handler({})
        verification = _record_verified_tool_run(tool_name="system_info", result=result, args={}, session_id=session_id)
        reply = "System info fetched."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result={"tool": "system_info", "result": result, "verification": verification},
            timestamp=_now_iso(),
            plan=plan,
        )

    if (
        "what tools" in msg_lower
        or "available tools" in msg_lower
        or msg_lower.strip() in {"tools", "tool list", "help tools"}
    ):
        names = ", ".join(sorted(_TOOLS.keys()))
        reply = (
            "Available tools: "
            + names
            + ".\n"
            + "Note: `shell_run` requires JARVIS_ALLOW_SHELL=true; `repo_write_file` requires JARVIS_ALLOW_REPO_WRITE=true; `write_file` writes under /data and requires JARVIS_ALLOW_WRITE=true."
        )
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=None,
            timestamp=_now_iso(),
            plan=plan,
        )

    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if is_openai_configured():
            provider = "openai"
        else:
            provider = "basic"

    if provider == "ollama":
        history = _memory.load(session_id, max_messages=12)
        memory_context = _memory_context_for_prompt(payload.message)
        max_steps = int(os.getenv("AGENT_MAX_TOOL_STEPS", "3"))
        max_steps = max(0, min(max_steps, 8))
        profile = _current_profile()
        runtime = _effective_ollama_runtime()
        system = (
            "You are Jarvis, running locally inside a Dockerized platform.\n"
            "You may ONLY use the tools explicitly listed below; do not invent capabilities.\n"
            "If you need a tool, output exactly ONE line in this format (and nothing else):\n"
            "TOOL: <tool_name> <json_args>\n"
            "Otherwise, answer normally.\n"
            "If the user asks what you can do, describe ONLY these tools and the API/dashboard, briefly.\n"
            "If a tool is disabled, you should say it is disabled by configuration.\n"
            f"Active execution profile: {profile}. In safe/dev profiles, risky commands or sensitive file writes require confirm=true.\n"
            "Available tools: "
            + ", ".join(sorted(_TOOLS.keys()))
            + "\n"
        )
        if memory_context:
            system += "Relevant long-term memory:\n" + memory_context + "\n"
        try:
            tool_calls: List[Dict[str, Any]] = []
            working = [{"role": m.role, "content": m.text} for m in history]

            for _ in range(max_steps + 1):
                raw = ollama_chat(
                    model=str(runtime.get("model") or ollama_model()),
                    messages=working,
                    system=system,
                    temperature=float(runtime.get("temperature") or 0.2),
                    timeout_s=int(os.getenv("OLLAMA_TIMEOUT_S", "120")),
                )

                directive = parse_tool_directive(raw)
                if not directive or directive.get("tool") not in _TOOLS:
                    reply = strip_final_answer(raw) or _basic_brain(payload.message)
                    _memory.append(session_id, StoredMessage(role="assistant", text=reply))
                    return AgentChatResponse(
                        session_id=session_id,
                        reply=reply,
                        tool_result={"calls": tool_calls} if tool_calls else None,
                        timestamp=_now_iso(),
                        plan=plan,
                    )

                tool_name = str(directive["tool"])
                args = (
                    directive.get("args")
                    if isinstance(directive.get("args"), dict)
                    else {}
                )
                result, verification, pending = _execute_tool_with_approval(
                    tool_name=tool_name,
                    args=args,
                    session_id=session_id,
                    label=f"Tool `{tool_name}`",
                )
                tool_calls.append({"tool": tool_name, "args": args, "result": result, "verification": verification, "pending_approval": pending})
                if pending:
                    reply = _approval_prompt_text(pending)
                    _memory.append(session_id, StoredMessage(role="assistant", text=reply))
                    return AgentChatResponse(
                        session_id=session_id,
                        reply=reply,
                        tool_result={"calls": tool_calls},
                        timestamp=_now_iso(),
                        plan=plan,
                    )

                working.append({"role": "assistant", "content": raw})
                working.append(
                    {
                        "role": "user",
                        "content": "Tool result: "
                        + json.dumps({"tool": tool_name, "result": result}),
                    }
                )

            reply = "Tool calls complete."
            _memory.append(session_id, StoredMessage(role="assistant", text=reply))
            return AgentChatResponse(
                session_id=session_id,
                reply=reply,
                tool_result={"calls": tool_calls} if tool_calls else None,
                timestamp=_now_iso(),
                plan=plan,
            )
        except Exception as exc:
            detail = str(exc).strip()
            if detail.startswith("Ollama "):
                reply = f"{detail} Falling back to basic mode."
            else:
                reply = f"Ollama request failed; falling back to basic mode. Error: {detail}"
            _memory.append(session_id, StoredMessage(role="assistant", text=reply))
            return AgentChatResponse(
                session_id=session_id,
                reply=reply,
                tool_result=None,
                timestamp=_now_iso(),
                plan=plan,
            )

    if provider != "openai":
        reply = _basic_brain(payload.message)
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(
            session_id=session_id,
            reply=reply,
            tool_result=None,
            timestamp=_now_iso(),
            plan=plan,
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip()
    if not api_key:
        reply = "OPENAI_API_KEY is not set; falling back to basic mode."
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(session_id=session_id, reply=reply, tool_result=None, timestamp=_now_iso(), plan=plan)

    history = _memory.load(session_id, max_messages=12)
    messages = [m.to_responses_input() for m in history]
    memory_context = _memory_context_for_prompt(payload.message)

    tools = [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema,
            "strict": True,
        }
        for t in sorted(_TOOLS.values(), key=lambda x: x.name)
    ]

    instructions = (
        "You are Jarvis, an assistant running inside a local platform.\n"
        "Use tools when helpful. If you call a tool, wait for the tool result and then answer.\n"
        "Be concise.\n"
    )
    if memory_context:
        instructions += "Relevant long-term memory:\n" + memory_context + "\n"

    try:
        first = chat_with_tools(
            api_key=api_key,
            model=model,
            instructions=instructions,
            messages=messages,
            tools=tools,
            timeout_s=int(os.getenv("OPENAI_TIMEOUT_S", "60")),
        )
    except Exception as exc:
        reply = f"OpenAI request failed; falling back to basic mode. Error: {exc}"
        _memory.append(session_id, StoredMessage(role="assistant", text=reply))
        return AgentChatResponse(session_id=session_id, reply=reply, tool_result=None, timestamp=_now_iso(), plan=plan)

    call = _extract_first_function_call(first)
    if call is not None:
        tool_name = call.get("name")
        call_id = call.get("call_id")
        raw_args = call.get("arguments", "{}")
        if not isinstance(tool_name, str) or tool_name not in _TOOLS or not isinstance(call_id, str):
            reply = "Model requested an unknown tool."
            _memory.append(session_id, StoredMessage(role="assistant", text=reply))
            return AgentChatResponse(session_id=session_id, reply=reply, tool_result={"tool": tool_name, "error": True}, timestamp=_now_iso(), plan=plan)

        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            if not isinstance(parsed_args, dict):
                parsed_args = {}
        except json.JSONDecodeError:
            parsed_args = {}

        result, verification, pending = _execute_tool_with_approval(
            tool_name=tool_name,
            args=parsed_args,
            session_id=session_id,
            label=f"Tool `{tool_name}`",
        )

        if pending:
            final_text = _approval_prompt_text(pending)
            _memory.append(session_id, StoredMessage(role="assistant", text=final_text))
            return AgentChatResponse(
                session_id=session_id,
                reply=final_text,
                tool_result={"tool": tool_name, "result": result, "verification": verification, "pending_approval": pending},
                timestamp=_now_iso(),
                plan=plan,
            )

        try:
            followup = chat_with_tools(
                api_key=api_key,
                model=model,
                instructions=instructions,
                messages=[],
                tools=tools,
                tool_output=(call_id, json.dumps(result)),
                previous_response_id=first.get("id"),
                timeout_s=int(os.getenv("OPENAI_TIMEOUT_S", "60")),
            )
            final_text = _extract_output_text(followup) or f"Tool `{tool_name}` executed."
        except Exception as exc:
            final_text = f"Tool `{tool_name}` executed, but follow-up failed: {exc}"

        _memory.append(session_id, StoredMessage(role="assistant", text=final_text))
        return AgentChatResponse(
            session_id=session_id,
            reply=final_text,
            tool_result={"tool": tool_name, "result": result, "verification": verification},
            timestamp=_now_iso(),
            plan=plan,
        )

    reply = _extract_output_text(first) or _basic_brain(payload.message)
    _memory.append(session_id, StoredMessage(role="assistant", text=reply))
    return AgentChatResponse(session_id=session_id, reply=reply, tool_result=None, timestamp=_now_iso(), plan=plan)
