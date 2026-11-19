"""Tests for llm.py: model calls fail fast with a clear error, never a long hang.

A fake model stands in for Gemini (no network, no key). Like the real library, it tries
a request at most `max_retries` times, so we can count the attempts."""

import pytest

from analystcrew import llm

MODEL = "google_genai:test-model"


class AlwaysFailingModel:
    """A fake chat model whose every request fails with an 'overloaded' error."""

    def __init__(self, max_retries: int):
        self.max_retries = max_retries
        self.attempts = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        # The real library turns max_retries into this many attempts in total.
        for _ in range(self.max_retries):
            self.attempts += 1
        raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")


@pytest.fixture
def fake_model(monkeypatch):
    """get_model() builds an AlwaysFailingModel; we keep the settings it was given."""
    made = {}

    def fake_init_chat_model(name, **settings):
        made["name"] = name
        made["settings"] = settings
        made["model"] = AlwaysFailingModel(settings["max_retries"])
        return made["model"]

    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-FAKE-KEY-123456")
    monkeypatch.setenv("ANALYSTCREW_MODEL", MODEL)
    monkeypatch.setattr(llm, "init_chat_model", fake_init_chat_model)
    return made


def test_model_gets_timeout_and_retry_limit(fake_model):
    llm.get_model()

    assert fake_model["name"] == MODEL
    assert fake_model["settings"]["timeout"] == 60
    assert fake_model["settings"]["max_retries"] == 2


@pytest.mark.parametrize("use", ["plain", "bind_tools", "with_structured_output"])
def test_failing_model_raises_clear_error_without_extra_retries(fake_model, use):
    model = llm.get_model()
    if use == "bind_tools":  # how the Researcher calls it
        model = model.bind_tools([])
    elif use == "with_structured_output":  # how the Writer calls it
        model = model.with_structured_output(dict)

    with pytest.raises(llm.ModelError) as caught:
        model.invoke("hello")

    message = str(caught.value)
    assert f"'{MODEL}'" in message
    assert "overloaded" in message
    assert "Try again later" in message
    assert "ANALYSTCREW_MODEL" in message
    assert "503 UNAVAILABLE" in message  # the real reason is kept
    assert "AIza-FAKE-KEY-123456" not in message
    assert fake_model["model"].attempts <= 3  # first try + at most 2 retries
    assert fake_model["model"].attempts == 2  # exactly what max_retries allows, nothing extra


def test_missing_key_error_is_unchanged(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("ANALYSTCREW_MODEL", MODEL)

    with pytest.raises(llm.ModelError, match="GOOGLE_API_KEY is missing"):
        llm.get_model()
