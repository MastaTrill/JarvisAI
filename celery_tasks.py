"""
JarvisAI Celery Tasks
Example background tasks for model training and data processing
"""

from celery_app import celery_app
from celery.signals import worker_ready, worker_shutdown
import time
import os
import threading
from datetime import datetime, timezone

import requests

from agent_task_memory import AgentTaskMemory

@celery_app.task(bind=True)
def train_model(self, model_name: str, epochs: int = 5):
    """Simulate model training as a background task"""
    for epoch in range(1, epochs + 1):
        time.sleep(1)  # Simulate training time
        self.update_state(state='PROGRESS', meta={'epoch': epoch, 'total_epochs': epochs})
    return {"status": "completed", "model": model_name, "epochs": epochs}

@celery_app.task
def process_data(data_id: int):
    """Simulate data processing task"""
    time.sleep(2)
    return {"status": "processed", "data_id": data_id}


_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()


def _scheduler_loop() -> None:
    memory = AgentTaskMemory()
    api_url = (os.getenv("JARVIS_API_URL", "http://jarvis-api:8000") or "").rstrip("/")
    poll_s = max(5, int(os.getenv("JARVIS_SCHEDULER_POLL_S", "15")))

    while not _scheduler_stop.is_set():
        now_iso = datetime.now(timezone.utc).isoformat()
        due = memory.claim_due_goal_schedules(now_iso=now_iso, limit=10)

        for sched in due:
            sid = int(sched["id"])
            try:
                payload = {
                    "goal": sched["goal"],
                    "session_id": sched.get("session_id"),
                    "auto_approve": bool(sched.get("auto_approve", False)),
                }
                resp = requests.post(f"{api_url}/agent/goals/run", json=payload, timeout=120)
                if resp.ok:
                    memory.mark_goal_schedule_result(sid, ok=True, error=None)
                else:
                    memory.mark_goal_schedule_result(sid, ok=False, error=f"http {resp.status_code}")
            except Exception as exc:
                memory.mark_goal_schedule_result(sid, ok=False, error=str(exc))

        due_jobs = memory.claim_due_autonomous_jobs(now_iso=now_iso, limit=10)
        for job in due_jobs:
            jid = int(job["id"])
            try:
                resp = requests.post(f"{api_url}/agent/autonomy/jobs/{jid}/run", timeout=180)
                if resp.ok:
                    payload = resp.json() if "application/json" in (resp.headers.get("content-type") or "") else {"ok": True}
                    memory.mark_autonomous_job_result(jid, ok=True, error=None, result=payload)
                else:
                    memory.mark_autonomous_job_result(jid, ok=False, error=f"http {resp.status_code}", result=None)
            except Exception as exc:
                memory.mark_autonomous_job_result(jid, ok=False, error=str(exc), result=None)

        try:
            requests.post(f"{api_url}/agent/memory/reminders/dispatch-due?limit=10", timeout=60)
        except Exception:
            pass

        _scheduler_stop.wait(poll_s)


@worker_ready.connect
def _start_goal_scheduler(**_kwargs):
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, name="jarvis-goal-scheduler", daemon=True)
    _scheduler_thread.start()


@worker_shutdown.connect
def _stop_goal_scheduler(**_kwargs):
    _scheduler_stop.set()
