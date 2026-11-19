"""The audit log: a record of every tool call, so we can check later what the agents did.

Each call adds one line of JSON to reports/audit.log with: when it happened, which run it
belonged to, the tool name, its arguments, a short preview of the result, a SHA-256 hash
of the full result (to prove later that a result was not changed), and any error.

Two things are never written:
- API keys and other secrets (they are replaced with "[REDACTED]").
- The text of news snippets (the preview shows only how long each snippet was).

Use it by wrapping a tool function:  @audited(run_id)
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import typing
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"  # backend/reports
LOG_NAME = "audit.log"
PREVIEW_CHARS = 200
REDACTED = "[REDACTED]"

# Argument names and environment variable names that hold secrets.
_SECRET_NAME = re.compile(r"key|token|secret|password", re.IGNORECASE)
# Secrets that can show up inside text, e.g. in an error message.
_SECRET_TEXT = [
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|key|token|secret|password)=)[^&\s\"']+", re.IGNORECASE),
]

_lock = threading.Lock()  # two tools finishing at once must not mix up their lines
_log = logging.getLogger(__name__)


def new_run_id() -> str:
    """A short random ID that groups all the tool calls of one run."""
    return uuid.uuid4().hex[:12]


# ---------- turning results into text ----------


def _plain(value):
    """Turn pydantic models (and lists of them) into plain dicts and lists."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _text(value) -> str:
    """The full result as text. The same result always gives the same text."""
    if isinstance(value, str):
        return value
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, default=str)


def _hide_snippets(value):
    """A copy with every news snippet replaced by its length, e.g. "[snippet, 300 chars]"."""
    if isinstance(value, dict):
        return {
            k: f"[snippet, {len(v)} chars]" if k == "snippet" and isinstance(v, str)
            else _hide_snippets(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_hide_snippets(v) for v in value]
    return value


# ---------- hiding secrets ----------


def _secret_values() -> list[str]:
    """Values of environment variables whose names look secret, like TAVILY_API_KEY."""
    return [
        v for name, v in os.environ.items()
        if _SECRET_NAME.search(name) and len(v.strip()) >= 6
    ]


def scrub(text: str) -> str:
    """Replace any secret found in the text with [REDACTED]."""
    for value in _secret_values():
        text = text.replace(value.strip(), REDACTED)
    for pattern in _SECRET_TEXT:
        text = pattern.sub(rf"\1{REDACTED}", text)
    return text


def _safe_args(args: dict) -> dict:
    """The arguments, with secret-named ones hidden and secrets removed from the rest."""
    safe = {}
    for name, value in args.items():
        if _SECRET_NAME.search(name):
            safe[name] = REDACTED
        else:
            safe[name] = json.loads(scrub(json.dumps(_plain(value), default=str)))
    return safe


# ---------- writing ----------


def record(run_id: str, tool: str, args: dict, result=None, error: str | None = None) -> dict:
    """Write one line to the audit log and return it."""
    preview_source = result if isinstance(result, str) else _text(_hide_snippets(_plain(result)))
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": run_id,
        "tool": tool,
        "args": _safe_args(args),
        "result_preview": None if error else scrub(preview_source)[:PREVIEW_CHARS],
        "result_sha256": None if error else hashlib.sha256(
            _text(result).encode("utf-8")).hexdigest(),
        "error": scrub(error) if error else None,
    }
    line = json.dumps(entry, ensure_ascii=False)
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with _lock, open(REPORTS_DIR / LOG_NAME, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:  # a full disk should not stop the research
        _log.warning("Could not write the audit log: %s", e)
    return entry


def audited(run_id: str | None = None):
    """Wrap a tool so every call is written to the audit log. Errors are logged, then re-raised.

    run_id can be a fixed string, or None to make a new one when the wrapper is created."""
    rid = run_id or new_run_id()

    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs).arguments
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                record(rid, fn.__name__, dict(bound), error=f"{type(e).__name__}: {e}")
                raise
            record(rid, fn.__name__, dict(bound), result=result)
            return result

        # Give tool frameworks (MCP, LangChain) the real types, e.g. FinancialReport.
        # As plain strings they would be looked up in this file, where they don't exist.
        try:
            wrapper.__annotations__ = typing.get_type_hints(fn)
        except Exception:  # unusual hints: keep the copied ones
            pass
        return wrapper

    return wrap
