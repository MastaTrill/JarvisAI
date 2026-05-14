"""Groq API integration for Jarvis AI."""

import os
from typing import Any, Dict, List, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None


def is_groq_configured() -> bool:
    """Check if Groq API is configured."""
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def groq_chat(
    model: str = "mixtral-8x7b-32768",
    messages: List[Dict[str, str]] = None,
    system: str = "",
    temperature: float = 0.7,
    timeout_s: int = 60,
) -> str:
    """Call Groq API and return text response."""
    if not is_groq_configured():
        raise ValueError("GROQ_API_KEY not set")

    if Groq is None:
        raise ImportError("groq library not installed")

    if messages is None:
        messages = []

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Build message list with system prompt
    all_messages = messages.copy()
    if system:
        all_messages.insert(0, {"role": "system", "content": system})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=all_messages,
            temperature=temperature,
            timeout=timeout_s,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Groq request failed: {str(e)}")


def groq_extract_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from Groq response (TOOL: name args format)."""
    import json
    import re

    # Look for TOOL: <name> <json_args> format
    match = re.search(r"TOOL:\s*(\w+)\s*({.*})", response_text, re.DOTALL)
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2)

    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        args = {}

    return {
        "name": tool_name,
        "arguments": args,
    }
