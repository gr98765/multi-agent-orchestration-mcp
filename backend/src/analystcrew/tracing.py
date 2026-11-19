"""Optional LangSmith tracing: a timeline of every model call, tool call and agent step.

Tracing is ON only when both are set (in the environment or the .env file):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
Otherwise it is OFF and the program runs normally and prints nothing about it.
LANGSMITH_PROJECT (optional) picks the project the traces go to.
Tracing is always OFF while tests run (pytest), so test data never reaches LangSmith.

Model and tool calls made through LangChain are traced automatically. Our own plain-Python
steps (Analyst, Writer, Reviewer, Coordinator) are traced with step() below.
The API key is only read by the LangSmith library. We never print, log or save it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from dotenv import load_dotenv
from langsmith import traceable
from langsmith import utils as ls_utils

# Every variable name LangSmith or LangChain checks to decide whether to trace.
_SWITCHES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2",
             "LANGCHAIN_TRACING")


def running_tests() -> bool:
    """True while pytest is running."""
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def tracing_enabled() -> bool:
    """True when LANGSMITH_TRACING is "true" and an API key is present."""
    switched_on = os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
    return switched_on and bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def setup_tracing() -> bool:
    """Load .env (never overwriting variables that are already set), then turn tracing
    fully on or fully off. Returns True if tracing is on. Prints nothing."""
    load_dotenv(override=False)
    on = tracing_enabled() and not running_tests()  # never trace test runs
    for name in _SWITCHES:
        if on and name == "LANGSMITH_TRACING":
            os.environ[name] = "true"  # LangSmith only accepts lowercase "true"
        elif not on and name in os.environ:
            os.environ[name] = "false"  # e.g. switched on but no key: stay quiet, stay off
    ls_utils.get_env_var.cache_clear()  # LangSmith remembers old values; make it re-read
    return on


def step(name: str, metadata: dict, fn: Callable, *args):
    """Run fn(*args) as a named step in the trace, e.g. step("writer", meta, write, ...).

    When tracing is off, this just calls fn(*args)."""
    return traceable(name=name, run_type="chain", metadata=metadata)(fn)(*args)
