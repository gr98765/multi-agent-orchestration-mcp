"""Tests for optional LangSmith tracing, fully offline.

1. With no key, a full run (with fake agents) completes and nothing about tracing is printed.
2. Even with tracing variables set inside a test, setup_tracing() refuses under pytest:
   a full run makes no network request and creates no LangSmith client.
3. Our named steps carry the right names and metadata. This is checked with tracing turned
   on explicitly for one block, sending runs to an in-memory client with the network blocked.

conftest.py makes sure the real .env file is never read and all tracing variables start off.
"""

import os

import langsmith
import pytest
import requests
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import run_trees
from test_graph import FakeAgents

from analystcrew import tracing
from analystcrew.agents.graph import run_report

FAKE_KEY = "lsv2_pt_FAKE-KEY-for-tests-only-123456"


@pytest.fixture
def network(monkeypatch) -> list[str]:
    """Records (and blocks) every HTTP request LangSmith could make."""
    calls: list[str] = []

    def blocked(self, method, url, *args, **kwargs):
        calls.append(f"{method} {url}")
        raise AssertionError("tests must not call the network")

    monkeypatch.setattr(requests.Session, "request", blocked)
    return calls


@pytest.fixture
def clients_made(monkeypatch) -> list[dict]:
    """Records every LangSmith client that gets created."""
    made: list[dict] = []
    original = langsmith.Client.__init__

    def recording_init(self, *args, **kwargs):
        made.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(langsmith.Client, "__init__", recording_init)
    return made


def test_no_key_runs_normally_with_tracing_off(monkeypatch, capsys):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # switched on, but no key

    assert tracing.setup_tracing() is False
    report = run_report("NVDA", FakeAgents(fails=1).agents())

    assert report.passed is True
    assert [t.step for t in report.trace][-2:] == ["write", "review"]
    assert capsys.readouterr().out == ""  # nothing printed about tracing


def test_tracing_variables_in_a_test_never_send_anything(monkeypatch, network, clients_made):
    # The worst case: a test (or a leaked .env) sets real-looking tracing variables.
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", "should-never-be-used")

    assert tracing.running_tests() is True
    assert tracing.setup_tracing() is False  # refuses under pytest...
    assert os.environ["LANGSMITH_TRACING"] == "false"  # ...and switches LangChain off too

    report = run_report("NVDA", FakeAgents(fails=1).agents())  # runs the traced steps
    wait_for_all_tracers()

    assert report.passed is True
    assert network == []
    assert clients_made == []
    assert run_trees._CLIENT is None


def test_steps_have_names_and_metadata(monkeypatch, network):
    runs: list[dict] = []

    class MemoryClient(langsmith.Client):
        """A LangSmith client that keeps runs in memory instead of sending them."""

        def create_run(self, *args, **run):
            runs.append(run)

        def update_run(self, run_id, **run):
            runs.append(run)

    client = MemoryClient(
        api_key=FAKE_KEY, api_url="http://langsmith.invalid", auto_batch_tracing=False)
    monkeypatch.setattr(run_trees, "_CLIENT", client)

    with langsmith.tracing_context(enabled=True, client=client):  # only for this block
        report = run_report("NVDA", FakeAgents(fails=1).agents())
    wait_for_all_tracers()

    assert report.passed is True
    assert network == []
    names = {r.get("name") for r in runs}
    assert {"analystcrew", "analyst", "writer", "reviewer", "coordinator"} <= names

    top = next(r for r in runs if r.get("name") == "analystcrew")
    run_id = top["extra"]["metadata"]["run_id"]
    for run in runs:
        if run.get("name") in {"analyst", "writer", "reviewer", "coordinator"}:
            assert run["extra"]["metadata"]["ticker"] == "NVDA"
            assert run["extra"]["metadata"]["run_id"] == run_id
    assert FAKE_KEY not in str(runs)  # the key is never put into a trace


@pytest.mark.parametrize("value", ["", "false", "TRUE-ish"])
def test_tracing_needs_true_and_a_key(monkeypatch, value):
    monkeypatch.setenv("LANGSMITH_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LANGSMITH_TRACING", value)

    assert tracing.tracing_enabled() is False
    assert tracing.setup_tracing() is False


def test_tracing_enabled_reads_switch_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "True")
    monkeypatch.setenv("LANGSMITH_API_KEY", FAKE_KEY)

    assert tracing.tracing_enabled() is True  # the settings ask for tracing...
    assert tracing.setup_tracing() is False  # ...but never during tests
