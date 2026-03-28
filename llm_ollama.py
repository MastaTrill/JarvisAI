from __future__ import annotations

import base64
import os
import re
import json
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests


class OllamaUnavailableError(RuntimeError):
    pass


def _build_base_url(parts: SplitResult, host: str) -> str:
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


def _candidate_ollama_base_urls() -> List[str]:
    configured = (os.getenv("OLLAMA_BASE_URL") or "http://ollama:11434").strip().rstrip("/")
    parts = urlsplit(configured)
    candidates: List[str] = [configured]
    host = (parts.hostname or "").strip().lower()

    # Docker-style hostnames are convenient in containers but fail on many
    # local Windows runs. Retry localhost variants automatically.
    if host in {"ollama", "host.docker.internal", "docker.internal"}:
        candidates.append(_build_base_url(parts, "127.0.0.1"))

    seen = set()
    deduped: List[str] = []
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def ollama_base_url() -> str:
    return _candidate_ollama_base_urls()[0]


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()


def is_ollama_configured() -> bool:
    return bool(ollama_model())


def _summarize_ollama_error(attempted: List[str], exc: Exception) -> str:
    attempted_list = ", ".join(attempted)
    detail = str(exc or "").strip()
    preferred = next((url for url in attempted if "127.0.0.1" in url or "localhost" in url), attempted[-1] if attempted else ollama_base_url())
    lower = detail.lower()

    if "failed to resolve" in lower or "nameresolutionerror" in lower or "getaddrinfo failed" in lower:
        return (
            f"Ollama is not reachable. Tried {attempted_list}. "
            "Start Ollama locally or update OLLAMA_BASE_URL."
        )
    if "failed to establish a new connection" in lower or "actively refused" in lower or "connection refused" in lower:
        return (
            f"Ollama is not running at {preferred}. "
            "Start Ollama locally or update OLLAMA_BASE_URL."
        )
    if isinstance(exc, requests.Timeout) or "read timed out" in lower or "connect timeout" in lower:
        return (
            f"Ollama timed out at {preferred}. "
            "Start Ollama locally or update OLLAMA_BASE_URL."
        )
    return (
        f"Ollama request failed after trying {attempted_list}. "
        "Start Ollama locally or update OLLAMA_BASE_URL."
    )


def _ollama_request(method: str, path: str, *, timeout_s: int, **kwargs: Any) -> requests.Response:
    attempted: List[str] = []
    last_exc: Optional[Exception] = None
    connect_timeout = 1
    for base_url in _candidate_ollama_base_urls():
        attempted.append(base_url)
        try:
            resp = requests.request(
                method,
                f"{base_url}{path}",
                timeout=(connect_timeout, timeout_s),
                **kwargs,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            continue
    raise OllamaUnavailableError(_summarize_ollama_error(attempted, last_exc or RuntimeError("unknown ollama error")))


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
    resp = _ollama_request("POST", "/api/chat", json=payload, timeout_s=timeout_s)
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
    resp = _ollama_request("POST", "/api/chat", json=payload, timeout_s=timeout_s)
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
    ollama_messages: List[Dict[str, str]] = list(messages or [])
    if system:
        ollama_messages = [{"role": "system", "content": system}] + ollama_messages
    payload: Dict[str, Any] = {"model": model, "messages": ollama_messages, "stream": True}
    if isinstance(temperature, (int, float)):
        payload["options"] = {"temperature": float(temperature)}

    with _ollama_request("POST", "/api/chat", json=payload, timeout_s=timeout_s, stream=True) as resp:
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
    try:
        resp = _ollama_request("GET", "/api/tags", timeout_s=timeout_s)
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
