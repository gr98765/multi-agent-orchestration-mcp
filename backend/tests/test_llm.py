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
    monkeypatch.delenv("ANALYSTCREW_FALLBACK_MODELS", raising=False)
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


# ---------- backup models (ANALYSTCREW_FALLBACK_MODELS) ----------

BACKUP = "google_genai:backup-model"


class ScriptedModel:
    """A fake chat model that fails with `error` (counting attempts like the real library),
    or answers "answer from <name>" if error is None."""

    def __init__(self, name: str, error: Exception | None, max_retries: int):
        self.name = name
        self.error = error
        self.max_retries = max_retries
        self.attempts = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if self.error is None:
            self.attempts += 1
            return f"answer from {self.name}"
        self.attempts += self.max_retries  # the library tries max_retries times, then gives up
        raise self.error


BUSY = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")


@pytest.fixture
def scripted(monkeypatch):
    """get_model() builds ScriptedModels; tests say how each model behaves, by name."""
    behavior: dict[str, Exception | None] = {}
    made: dict[str, ScriptedModel] = {}

    def fake_init_chat_model(name, **settings):
        made[name] = ScriptedModel(name, behavior[name], settings["max_retries"])
        made[name].settings = settings
        return made[name]

    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-FAKE-KEY-123456")
    monkeypatch.setenv("ANALYSTCREW_MODEL", MODEL)
    monkeypatch.setenv("ANALYSTCREW_FALLBACK_MODELS", f" {BACKUP} , {MODEL},")
    monkeypatch.setattr(llm, "init_chat_model", fake_init_chat_model)
    return behavior, made


def test_fallback_list_is_read_from_env(scripted):
    assert llm.model_names() == [MODEL, BACKUP]  # spaces, repeats and blanks are ignored


@pytest.mark.parametrize("use", ["plain", "bind_tools", "with_structured_output"])
def test_busy_main_model_hands_over_to_backup(scripted, use):
    behavior, made = scripted
    behavior.update({MODEL: BUSY, BACKUP: None})
    model = llm.get_model()
    if use == "bind_tools":
        model = model.bind_tools([])
    elif use == "with_structured_output":
        model = model.with_structured_output(dict)

    assert model.invoke("hello") == f"answer from {BACKUP}"
    assert model.answered_by == BACKUP
    assert made[MODEL].attempts == 2 and made[BACKUP].attempts == 1


@pytest.mark.parametrize("error", [
    RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded."),
    TimeoutError("The read operation timed out"),
    ConnectionError("Server disconnected without sending a response."),
])
def test_rate_limit_and_timeout_also_use_backup(scripted, error):
    behavior, _ = scripted
    behavior.update({MODEL: error, BACKUP: None})

    assert llm.get_model().invoke("hello") == f"answer from {BACKUP}"


def test_main_model_used_first_when_it_works(scripted):
    behavior, made = scripted
    behavior.update({MODEL: None, BACKUP: None})
    model = llm.get_model()

    assert model.invoke("hello") == f"answer from {MODEL}"
    assert model.answered_by == MODEL
    assert made[BACKUP].attempts == 0


def test_all_models_busy_error_names_every_model(scripted):
    behavior, made = scripted
    behavior.update({MODEL: BUSY, BACKUP: TimeoutError("The read operation timed out")})

    with pytest.raises(llm.ModelError) as caught:
        llm.get_model().invoke("hello")

    message = str(caught.value)
    assert f"'{MODEL}'" in message and f"'{BACKUP}'" in message
    assert "503 UNAVAILABLE" in message and "timed out" in message
    assert "ANALYSTCREW_FALLBACK_MODELS" in message
    assert "AIza-FAKE-KEY-123456" not in message
    # every model keeps the 60-second timeout and 2-try limit: at most 2 x 2 tries in total
    assert all(m.settings["timeout"] == 60 and m.settings["max_retries"] == 2
               for m in made.values())
    assert made[MODEL].attempts == 2 and made[BACKUP].attempts == 2


def test_other_errors_do_not_use_backup(scripted):
    """A wrong request (not 'busy') fails right away, as before: a backup won't fix it."""
    behavior, made = scripted
    behavior.update({MODEL: ValueError("400 INVALID_ARGUMENT: bad request"), BACKUP: None})

    with pytest.raises(llm.ModelError, match=f"'{MODEL}' did not answer"):
        llm.get_model().invoke("hello")
    assert made[BACKUP].attempts == 0


def test_backup_with_other_error_still_names_every_model(scripted):
    """Main model busy, backup gone (404): the error lists both, then stops."""
    behavior, _ = scripted
    behavior.update({MODEL: BUSY, BACKUP: ValueError("404 NOT_FOUND: model no longer available")})

    with pytest.raises(llm.ModelError) as caught:
        llm.get_model().invoke("hello")

    message = str(caught.value)
    assert f"'{MODEL}'" in message and f"'{BACKUP}'" in message
    assert "503 UNAVAILABLE" in message and "404 NOT_FOUND" in message
