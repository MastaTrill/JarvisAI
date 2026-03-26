from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class OpenAIRequestError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"OpenAI API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: int = 60,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def create_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        base_backoff_s = float(os.getenv("OPENAI_RETRY_BACKOFF_S", "1.5"))
        retry_statuses = {429, 500, 502, 503, 504}

        for attempt in range(max_retries + 1):
            resp = requests.post(
                url, headers=headers, json=payload, timeout=self._timeout_s
            )
            if resp.status_code < 400:
                return resp.json()

            # Retry only on transient errors.
            if resp.status_code in retry_statuses and attempt < max_retries:
                retry_after = resp.headers.get("retry-after") or resp.headers.get(
                    "Retry-After"
                )
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = base_backoff_s * (2**attempt)
                else:
                    sleep_s = base_backoff_s * (2**attempt)
                time.sleep(min(sleep_s, 20.0))
                continue

            # Surface server response body for debugging (no secrets included).
            body = resp.text.strip()
            if len(body) > 500:
                body = body[:500] + "..."
            raise OpenAIRequestError(resp.status_code, body or resp.reason)

        raise OpenAIRequestError(500, "Exhausted retries")


def _extract_output_text(response: Dict[str, Any]) -> str:
    # Responses API has convenience `output_text` on the response in many cases.
    text = response.get("output_text")
    if isinstance(text, str) and text.strip():
        return text

    chunks: List[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") in {"output_text", "text"}:
                t = c.get("text")
                if isinstance(t, str) and t:
                    chunks.append(t)
    return "\n".join(chunks).strip()


def _extract_first_function_call(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            return item
    return None


def chat_with_tools(
    *,
    api_key: str,
    model: str,
    instructions: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tool_output: Optional[Tuple[str, str]] = None,
    previous_response_id: Optional[str] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    `messages` should be Responses-style input items:
      [{"role":"user","content":[{"type":"input_text","text":"..."}]}, ...]

    If `tool_output` is provided, it must be (call_id, output_json_string).
    """
    client = OpenAIResponsesClient(api_key=api_key, timeout_s=timeout_s)

    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
    if tool_output is not None:
        call_id, output_str = tool_output
        payload["input"] = [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_str,
            }
        ]

    return client.create_response(payload)


def is_openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def vision_analyze(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_bytes: bytes,
    timeout_s: int = 90,
) -> str:
    client = OpenAIResponsesClient(api_key=api_key, timeout_s=timeout_s)
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}",
                    },
                ],
            }
        ],
    }
    response = client.create_response(payload)
    return _extract_output_text(response)
