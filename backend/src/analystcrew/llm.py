"""Connects to the AI model. Swap models by changing ANALYSTCREW_MODEL in .env.

Model calls fail fast: each request gives up after 60 seconds and is tried at most twice.
If it still fails, you get a ModelError that names the model and says what to do,
instead of a long hang or a confusing error from deep inside the model library.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .audit import scrub
from .tracing import setup_tracing

setup_tracing()  # loads .env (never overwriting set variables) and turns tracing on or off

DEFAULT_MODEL = "google_genai:gemini-3.6-flash"
TIMEOUT_SECONDS = 60  # give up on one request after this long
MAX_RETRIES = 2  # passed to the model library: how many times it may try a request


class ModelError(Exception):
    """The model can't be used. The message says how to fix it."""


def _failure_message(model_name: str, error: Exception) -> str:
    reason = scrub(f"{type(error).__name__}: {error}")[:300]  # never show a key
    return (
        f"The model '{model_name}' did not answer ({reason}). The service may be "
        "overloaded. Try again later, or set ANALYSTCREW_MODEL in your .env file to "
        "another model."
    )


class Model:
    """Wraps a chat model so a failed call raises ModelError with clear advice.

    bind_tools() and with_structured_output() return wrapped models too, so the
    Researcher's and the Writer's calls get the same clear errors."""

    def __init__(self, model, model_name: str):
        self._model = model
        self._name = model_name

    def bind_tools(self, *args, **kwargs) -> Model:
        return Model(self._model.bind_tools(*args, **kwargs), self._name)

    def with_structured_output(self, *args, **kwargs) -> Model:
        return Model(self._model.with_structured_output(*args, **kwargs), self._name)

    def invoke(self, *args, **kwargs):
        try:
            return self._model.invoke(*args, **kwargs)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(_failure_message(self._name, e)) from e

    def __getattr__(self, name):  # anything else goes straight to the real model
        return getattr(self._model, name)


def get_model() -> Model:
    model_name = os.getenv("ANALYSTCREW_MODEL", DEFAULT_MODEL)

    if model_name.startswith("google_genai") and not os.getenv("GOOGLE_API_KEY"):
        raise ModelError(
            "GOOGLE_API_KEY is missing from your .env file. Get a free key at "
            "https://aistudio.google.com and add GOOGLE_API_KEY=... to it."
        )

    # Free tiers allow only a few requests per minute. Pace ourselves to stay under the limit.
    limiter = InMemoryRateLimiter(requests_per_second=0.1, max_bucket_size=2)
    model: BaseChatModel = init_chat_model(
        model_name,
        temperature=0,
        rate_limiter=limiter,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    return Model(model, model_name)
