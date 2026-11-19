"""Connects to the AI model. Swap models by changing ANALYSTCREW_MODEL in .env.

Model calls fail fast: each request gives up after 60 seconds and is tried at most twice.

If the model is busy (HTTP 503 "unavailable", HTTP 429 "too many requests", a timeout, or
the server hanging up without an answer),
we try backup models, listed in .env as ANALYSTCREW_FALLBACK_MODELS (comma-separated), e.g.
    ANALYSTCREW_FALLBACK_MODELS=google_genai:gemini-3.5-flash,google_genai:gemini-2.5-flash
Every call starts with the main model; backups are only used when it is busy.
Each backup gets the same 60-second timeout and 2 tries, so even a full failure stays short.

If no model answers, you get a ModelError that names every model tried and says what to do,
instead of a long hang or a confusing error from deep inside the model library.
"""

from __future__ import annotations

import os
import sys

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .audit import scrub
from .tracing import setup_tracing

setup_tracing()  # loads .env (never overwriting set variables) and turns tracing on or off

DEFAULT_MODEL = "google_genai:gemini-3.6-flash"
TIMEOUT_SECONDS = 60  # give up on one request after this long
MAX_RETRIES = 2  # passed to the model library: how many times it may try a request

# Words in an error that mean "busy, try another model": overloaded, rate-limited, too slow,
# or the overloaded server hung up without answering.
BUSY_SIGNS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "Timeout", "timed out",
              "Server disconnected")


class ModelError(Exception):
    """The model can't be used. The message says how to fix it."""


def _reason(error: Exception) -> str:
    return scrub(f"{type(error).__name__}: {error}")[:300]  # never show a key


def _failure_message(model_name: str, error: Exception) -> str:
    return (
        f"The model '{model_name}' did not answer ({_reason(error)}). The service may be "
        "overloaded. Try again later, or set ANALYSTCREW_MODEL in your .env file to "
        "another model."
    )


def _all_failed_message(failures: list[tuple[str, Exception]]) -> str:
    tried = "; ".join(f"'{name}' ({_reason(error)})" for name, error in failures)
    return (
        f"None of the {len(failures)} models answered. Tried: {tried}. The service may be "
        "overloaded. Try again later, or change ANALYSTCREW_MODEL or "
        "ANALYSTCREW_FALLBACK_MODELS in your .env file."
    )


def is_busy(error: Exception) -> bool:
    """True if the error means the model is busy (503, 429 or a timeout)."""
    if isinstance(error, TimeoutError):
        return True
    text = f"{type(error).__name__}: {error}"
    return any(sign in text for sign in BUSY_SIGNS)


class Model:
    """Wraps one or more chat models (main first, then backups).

    A busy model hands the call to the next one. Any other failure raises ModelError with
    clear advice. bind_tools() and with_structured_output() apply to every model, so the
    Researcher's and the Writer's calls get the same backups and clear errors.
    `answered_by` is the name of the model that answered the most recent call."""

    def __init__(self, models: list[tuple[str, object]]):
        self._models = models  # [(name, chat model), ...] main model first
        self.answered_by: str | None = None

    def bind_tools(self, *args, **kwargs) -> Model:
        return Model([(n, m.bind_tools(*args, **kwargs)) for n, m in self._models])

    def with_structured_output(self, *args, **kwargs) -> Model:
        return Model([(n, m.with_structured_output(*args, **kwargs)) for n, m in self._models])

    def invoke(self, *args, **kwargs):
        failures: list[tuple[str, Exception]] = []
        for i, (name, model) in enumerate(self._models):
            try:
                result = model.invoke(*args, **kwargs)
            except ModelError:
                raise
            except Exception as e:
                failures.append((name, e))
                if not is_busy(e):  # a backup won't fix this; stop, naming every model tried
                    if len(failures) == 1:
                        raise ModelError(_failure_message(name, e)) from e
                    raise ModelError(_all_failed_message(failures)) from e
                if i + 1 < len(self._models):
                    print(f"Model '{name}' is busy ({_reason(e)[:80]}); "
                          f"trying '{self._models[i + 1][0]}'.", file=sys.stderr)
                continue
            self.answered_by = name
            if failures:
                print(f"Model '{name}' answered.", file=sys.stderr)
            return result
        last = failures[-1][1]
        if len(failures) == 1:
            raise ModelError(_failure_message(*failures[0])) from last
        raise ModelError(_all_failed_message(failures)) from last

    def __getattr__(self, name):  # anything else goes straight to the main model
        return getattr(self._models[0][1], name)


def model_names() -> list[str]:
    """The main model, then the backups from ANALYSTCREW_FALLBACK_MODELS (no repeats)."""
    names = [os.getenv("ANALYSTCREW_MODEL", DEFAULT_MODEL)]
    for name in os.getenv("ANALYSTCREW_FALLBACK_MODELS", "").split(","):
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return names


def _build(model_name: str) -> BaseChatModel:
    if model_name.startswith("google_genai") and not os.getenv("GOOGLE_API_KEY"):
        raise ModelError(
            "GOOGLE_API_KEY is missing from your .env file. Get a free key at "
            "https://aistudio.google.com and add GOOGLE_API_KEY=... to it."
        )
    # Free tiers allow only a few requests per minute. Pace ourselves to stay under the limit.
    limiter = InMemoryRateLimiter(requests_per_second=0.1, max_bucket_size=2)
    return init_chat_model(
        model_name,
        temperature=0,
        rate_limiter=limiter,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def get_model() -> Model:
    return Model([(name, _build(name)) for name in model_names()])
