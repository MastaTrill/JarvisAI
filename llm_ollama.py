from __future__ import annotations

import base64
import os
import re
import json
from typing import Any, Dict, Iterator, List, Optional

import requests


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()


def is_ollama_configured() -> bool:
    return bool(ollama_model())


def _to_ollama_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert our stored format to Ollama chat messages: [{"role":"user","content":"..."}].
    """
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        text = m.get("text")
        if isinstance(role, str) and isinstance(text, str):
            out.append({"role": role, "content": text})
    return out


def ollama_chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    format_json: bool = False,
    options: Optional[Dict[str, Any]] = None,
    timeout_s: int = 60,
) -> str:
    url = f"{ollama_base_url()}/api/chat"
    ollama_messages: List[Dict[str, str]] = list(messages or [])
    if system:
        # Some models follow instructions more reliably when the system prompt is
        # provided as an explicit system message (rather than the top-level field).
        ollama_messages = [{"role": "system", "content": system}] + ollama_messages
    payload: Dict[str, Any] = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
    }
    opts: Dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        opts["temperature"] = float(temperature)
    if isinstance(options, dict):
        for k, v in options.items():
            opts[k] = v
    if opts:
        payload["options"] = opts
    if format_json:
        payload["format"] = "json"
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def ollama_chat_with_images(
    *,
    model: str,
    prompt: str,
    image_bytes: List[bytes],
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    format_json: bool = False,
    options: Optional[Dict[str, Any]] = None,
    timeout_s: int = 120,
) -> str:
    url = f"{ollama_base_url()}/api/chat"
    user_message: Dict[str, Any] = {
        "role": "user",
        "content": prompt,
        "images": [base64.b64encode(b).decode("ascii") for b in image_bytes if isinstance(b, (bytes, bytearray))],
    }
    ollama_messages: List[Dict[str, Any]] = [user_message]
    if system:
        ollama_messages = [{"role": "system", "content": system}] + ollama_messages
    payload: Dict[str, Any] = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
    }
    opts: Dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        opts["temperature"] = float(temperature)
    if isinstance(options, dict):
        for k, v in options.items():
            opts[k] = v
    if opts:
        payload["options"] = opts
    if format_json:
        payload["format"] = "json"
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def ollama_chat_stream(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout_s: int = 120,
) -> Iterator[str]:
    url = f"{ollama_base_url()}/api/chat"
    ollama_messages: List[Dict[str, str]] = list(messages or [])
    if system:
        ollama_messages = [{"role": "system", "content": system}] + ollama_messages
    payload: Dict[str, Any] = {"model": model, "messages": ollama_messages, "stream": True}
    if isinstance(temperature, (int, float)):
        payload["options"] = {"temperature": float(temperature)}

    with requests.post(url, json=payload, timeout=timeout_s, stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            message = data.get("message") if isinstance(data, dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield content


def ollama_list_models(timeout_s: int = 10) -> List[str]:
    url = f"{ollama_base_url()}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    models = data.get("models") if isinstance(data, dict) else []
    if not isinstance(models, list):
        return []
    names: List[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


_TOOL_LINE_RE = re.compile(r"^\s*TOOL\s*:\s*(\w+)\s*(\{.*\})?\s*$", re.IGNORECASE)


def parse_tool_directive(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single tool directive of the form:
      TOOL: tool_name {"arg":"value"}

    Returns {"tool": str, "args": dict}.
    """
    for line in (text or "").splitlines():
        m = _TOOL_LINE_RE.match(line)
        if not m:
            continue
        tool = m.group(1)
        args_raw = m.group(2) or "{}"
        import json

        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        return {"tool": tool, "args": args}
    return None


def strip_final_answer(text: str) -> str:
    """
    Remove any TOOL directive line and any leading 'FINAL:' marker.
    """
    lines = []
    for line in (text or "").splitlines():
        if _TOOL_LINE_RE.match(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("final:"):
        cleaned = cleaned[6:].strip()
    return cleaned
