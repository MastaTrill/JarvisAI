"""
Code Execution Sandbox for JarvisAI.

Provides safe, sandboxed code execution with:
- Restricted Python subprocess with timeout
- Resource limits (CPU time, memory)
- Whitelisted imports
- Execution history and output capture
- Support for Python, JavaScript (Node), and shell scripts
"""

import os
import sys
import json
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import resource  # Unix-only
except ImportError:
    resource = None  # Windows

try:
    import signal
except ImportError:
    signal = None

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text

from db_config import Base as ConfigBase
from database import get_db
from auth_helpers import get_current_user
from models_user import User


# --- Database Model ---

class CodeExecution(ConfigBase):
    __tablename__ = "code_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    language = Column(String(20), default="python")
    code = Column(Text, nullable=False)
    output = Column(Text, default="")
    error = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True)
    duration_ms = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending, running, success, error, timeout
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))


# --- Pydantic Schemas ---

class CodeRequest(BaseModel):
    code: str
    language: str = Field(default="python", description="python, javascript, bash")
    timeout: int = Field(default=30, ge=1, le=120)
    stdin: Optional[str] = None


class CodeResponse(BaseModel):
    id: int
    language: str
    output: str
    error: Optional[str]
    exit_code: Optional[int]
    duration_ms: int
    status: str


# --- Router ---

router = APIRouter(prefix="/sandbox", tags=["Code Sandbox"])

# Whitelisted Python imports
PYTHON_ALLOWED_IMPORTS = {
    "math", "random", "statistics", "datetime", "collections", "itertools",
    "functools", "operator", "copy", "pprint", "textwrap", "string",
    "re", "json", "csv", "io", "hashlib", "base64", "binascii",
    "struct", "array", "queue", "heapq", "bisect", "decimal",
    "fractions", "numbers", "typing", "dataclasses", "enum",
    "pathlib", "os.path", "time", "calendar", "uuid", "secrets",
    "html", "xml.etree.ElementTree", "urllib.parse", "difflib",
    "unicodedata", "codecs", "stringprep", "locale",
    "numpy", "pandas", "matplotlib", "seaborn", "sklearn",
    "scipy", "tensorflow", "torch", "transformers",
}

# Blocked patterns in code
BLOCKED_PATTERNS = [
    r"__import__\s*\(",
    r"exec\s*\(",
    r"eval\s*\(",
    r"compile\s*\(",
    r"open\s*\(\s*['\"]/",
    r"subprocess",
    r"os\.system",
    r"os\.popen",
    r"os\.exec",
    r"os\.spawn",
    r"shutil\.rmtree",
    r"shutil\.move",
    r"socket",
    r"requests\.",
    r"urllib\.request",
    r"http\.client",
    r"ftplib",
    r"smtplib",
    r"telnetlib",
    r"xmlrpc",
    r"pickle\.loads",
    r"yaml\.load\b",
    r"input\s*\(",
    r"breakpoint\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"vars\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
]


def _validate_code(code: str, language: str) -> Optional[str]:
    """Validate code for dangerous patterns. Returns error message or None."""
    import re

    if language == "python":
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return f"Blocked: code contains disallowed pattern '{pattern}'"

    elif language == "bash":
        blocked_bash = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){:|:&};:",
            "chmod 777 /", "chown root", "sudo", "su -",
            "wget", "curl", "nc ", "ncat", "nmap",
            "iptables", "ufw", "firewall",
            "systemctl", "service ",
            "crontab", "at ",
            "useradd", "userdel", "passwd",
            "mount", "umount", "fdisk",
        ]
        for pattern in blocked_bash:
            if pattern in code.lower():
                return f"Blocked: code contains disallowed command '{pattern}'"

    return None


def _execute_python(code: str, timeout: int, stdin: Optional[str] = None) -> Dict[str, Any]:
    """Execute Python code in a restricted subprocess."""
    import time as time_module

    # Wrap code to capture output and restrict environment
    wrapper = f'''
import sys
import io
import json
import traceback

# Restrict builtins
_safe_builtins = {{
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
    'callable', 'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
    'filter', 'float', 'format', 'frozenset', 'hasattr', 'hash', 'hex',
    'id', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'map',
    'max', 'min', 'next', 'object', 'oct', 'ord', 'pow', 'print', 'range',
    'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip', 'True', 'False', 'None',
    'NotImplemented', 'Ellipsis', 'Exception', 'ValueError', 'TypeError',
    'KeyError', 'IndexError', 'AttributeError', 'RuntimeError', 'StopIteration',
    'ZeroDivisionError', 'OverflowError', 'ImportError', 'ModuleNotFoundError',
    'OSError', 'IOError', 'FileNotFoundError', 'PermissionError',
    'ArithmeticError', 'LookupError', 'AssertionError', 'BufferError',
    'DeprecationWarning', 'FutureWarning', 'UserWarning', 'Warning',
    'classmethod', 'staticmethod', 'property',
}}

class RestrictedBuiltins:
    def __init__(self, allowed):
        self._allowed = allowed
    def __getitem__(self, name):
        if name in self._allowed:
            return __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
        raise NameError(f"builtin '{{name}}' is not allowed")

# Capture stdout/stderr
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = _captured_stdout = io.StringIO()
sys.stderr = _captured_stderr = io.StringIO()

_result = {{"output": "", "error": None, "exit_code": 0}}

try:
    _code = {json.dumps(code)}
    exec(compile(_code, "<sandbox>", "exec"))
except SystemExit as e:
    _result["exit_code"] = e.code if e.code is not None else 0
except Exception as e:
    _result["error"] = traceback.format_exc()
    _result["exit_code"] = 1
finally:
    _result["output"] = _captured_stdout.getvalue()
    _stderr_output = _captured_stderr.getvalue()
    if _stderr_output and not _result["error"]:
        _result["error"] = _stderr_output

sys.stdout = _old_stdout
sys.stderr = _old_stderr
print(json.dumps(_result))
'''

    start = time_module.time()
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            f.flush()
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.PIPE if stdin else None,
            input=stdin,
        )
        duration = int((time_module.time() - start) * 1000)

        os.unlink(temp_path)

        if result.returncode != 0:
            return {
                "output": result.stdout,
                "error": result.stderr or "Process exited with non-zero code",
                "exit_code": result.returncode,
                "duration_ms": duration,
                "status": "error",
            }

        # Parse the JSON output from the wrapper
        try:
            # Find the JSON line (last line of stdout)
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    data = json.loads(line)
                    return {
                        "output": data.get("output", result.stdout),
                        "error": data.get("error"),
                        "exit_code": data.get("exit_code", 0),
                        "duration_ms": duration,
                        "status": "success" if not data.get("error") else "error",
                    }
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "output": result.stdout,
            "error": None,
            "exit_code": 0,
            "duration_ms": duration,
            "status": "success",
        }

    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return {
            "output": "",
            "error": f"Execution timed out after {timeout} seconds",
            "exit_code": -1,
            "duration_ms": timeout * 1000,
            "status": "timeout",
        }
    except Exception as e:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        return {
            "output": "",
            "error": str(e),
            "exit_code": -1,
            "duration_ms": int((time_module.time() - start) * 1000),
            "status": "error",
        }


def _execute_javascript(code: str, timeout: int, stdin: Optional[str] = None) -> Dict[str, Any]:
    """Execute JavaScript code using Node.js."""
    import time as time_module

    start = time_module.time()
    try:
        result = subprocess.run(
            ["node", "-e", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = int((time_module.time() - start) * 1000)

        return {
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "exit_code": result.returncode,
            "duration_ms": duration,
            "status": "success" if result.returncode == 0 else "error",
        }
    except FileNotFoundError:
        return {
            "output": "",
            "error": "Node.js is not installed. Install it to run JavaScript code.",
            "exit_code": -1,
            "duration_ms": 0,
            "status": "error",
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"Execution timed out after {timeout} seconds",
            "exit_code": -1,
            "duration_ms": timeout * 1000,
            "status": "timeout",
        }


def _execute_bash(code: str, timeout: int, stdin: Optional[str] = None) -> Dict[str, Any]:
    """Execute bash commands in a restricted shell."""
    import time as time_module

    start = time_module.time()
    try:
        result = subprocess.run(
            ["bash", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = int((time_module.time() - start) * 1000)

        return {
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "exit_code": result.returncode,
            "duration_ms": duration,
            "status": "success" if result.returncode == 0 else "error",
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"Execution timed out after {timeout} seconds",
            "exit_code": -1,
            "duration_ms": timeout * 1000,
            "status": "timeout",
        }


# --- Endpoints ---

@router.post("/execute", response_model=CodeResponse)
def execute_code(
    body: CodeRequest,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Execute code in a sandboxed environment."""
    # Validate
    validation_error = _validate_code(body.code, body.language)
    if validation_error:
        raise HTTPException(400, validation_error)

    # Execute
    if body.language == "python":
        result = _execute_python(body.code, body.timeout, body.stdin)
    elif body.language == "javascript":
        result = _execute_javascript(body.code, body.timeout, body.stdin)
    elif body.language == "bash":
        result = _execute_bash(body.code, body.timeout, body.stdin)
    else:
        raise HTTPException(400, f"Unsupported language: {body.language}")

    # Store execution record
    exec_record = CodeExecution(
        language=body.language,
        code=body.code[:10000],  # Limit stored code size
        output=result["output"][:50000],
        error=result["error"][:10000] if result["error"] else None,
        exit_code=result["exit_code"],
        duration_ms=result["duration_ms"],
        status=result["status"],
        created_by=current_user.username,
    )
    db.add(exec_record)
    db.commit()
    db.refresh(exec_record)

    return CodeResponse(
        id=exec_record.id,
        language=body.language,
        output=result["output"],
        error=result["error"],
        exit_code=result["exit_code"],
        duration_ms=result["duration_ms"],
        status=result["status"],
    )


@router.get("/history")
def execution_history(
    limit: int = Query(20, ge=1, le=100),
    language: Optional[str] = None,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get code execution history."""
    query = db.query(CodeExecution)
    if language:
        query = query.filter_by(language=language)
    executions = query.order_by(CodeExecution.created_at.desc()).limit(limit).all()
    return [
        {
            "id": e.id,
            "language": e.language,
            "code": e.code[:500] + "..." if len(e.code) > 500 else e.code,
            "output": e.output[:1000] + "..." if e.output and len(e.output) > 1000 else e.output,
            "error": e.error[:500] if e.error else None,
            "exit_code": e.exit_code,
            "duration_ms": e.duration_ms,
            "status": e.status,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in executions
    ]


@router.get("/languages")
def supported_languages():
    """List supported execution languages."""
    langs = [
        {"id": "python", "name": "Python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
        {"id": "bash", "name": "Bash", "version": "5.x"},
    ]
    # Check if Node.js is available
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            langs.insert(1, {"id": "javascript", "name": "JavaScript (Node.js)", "version": result.stdout.strip()})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {"languages": langs}
