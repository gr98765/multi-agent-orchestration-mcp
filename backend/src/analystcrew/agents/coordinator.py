"""The Coordinator: runs each agent as one step and decides what happens after a review.

The steps are: research -> analyze -> write -> review. If the review fails, its issues go
back to the Writer for another try, at most MAX_RETRIES extra rounds. Then we finish,
pass or fail, so a run can never loop forever.

Every step is timed and written to a trace (step, seconds, round, review issues), and the
trace is stored on the final report. graph.py connects these steps with LangGraph.

When LangSmith tracing is on (see tracing.py), the Analyst, Writer, Reviewer and Coordinator
also appear as named steps there, tagged with the ticker and run ID. The Researcher needs
nothing extra: its model and tool calls go through LangChain, which traces them itself.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from ..models import Claim, Report, ResearchNotes, TraceStep
from ..tracing import step
from .analyst import analyze
from .researcher import run_research
from .reviewer import Review, review
from .writer import write_report

MAX_RETRIES = 2  # rewrites after the first draft, so at most 3 drafts in total


@dataclass
class Agents:
    """The four agents. Tests swap in fakes; normal runs use the real ones."""

    research: Callable[[str], ResearchNotes] = run_research
    analyze: Callable[[ResearchNotes], list[Claim]] = analyze
    write: Callable[[ResearchNotes, list[Claim], list[str]], Report] = write_report
    review: Callable[[list[Claim], ResearchNotes], Review] = review


class State(TypedDict, total=False):
    """Everything the steps share. Each step returns only the parts it changes."""

    company: str
    run_id: str  # groups this run's steps in LangSmith
    notes: ResearchNotes
    claims: list[Claim]
    report: Report
    round: int  # how many drafts the Writer has made so far
    passed: bool  # did the latest review pass?
    issues: list[str]  # problems from the latest review
    feedback: list[str]  # what the Writer must fix in its next draft
    trace: Annotated[list[TraceStep], operator.add]  # each step adds to the list


def _timed(step: str, round_no: int, work: Callable[[], object]) -> tuple[object, TraceStep]:
    """Run work(), and return its result with a trace entry saying how long it took."""
    start = time.perf_counter()
    result = work()
    seconds = round(time.perf_counter() - start, 3)
    return result, TraceStep(step=step, seconds=seconds, round=round_no)


def _meta(state: State) -> dict:
    """Tags added to each LangSmith step, so one run's steps are easy to find."""
    return {"ticker": state["company"], "run_id": state.get("run_id", "")}


def make_steps(agents: Agents) -> dict[str, Callable[[State], dict]]:
    """The graph's steps, each using the given agents."""

    def research_step(state: State) -> dict:
        notes, trace = _timed("research", 0, lambda: agents.research(state["company"]))
        return {"notes": notes, "trace": [trace]}

    def analyze_step(state: State) -> dict:
        claims, trace = _timed(
            "analyze", 0, lambda: step("analyst", _meta(state), agents.analyze, state["notes"]))
        return {"claims": claims, "trace": [trace]}

    def write_step(state: State) -> dict:
        round_no = state.get("round", 0) + 1
        report, trace = _timed(
            "write", round_no,
            lambda: step("writer", _meta(state), agents.write,
                         state["notes"], state["claims"], state.get("feedback", [])),
        )
        return {"report": report, "round": round_no, "trace": [trace]}

    def review_step(state: State) -> dict:
        report = state["report"]
        result, trace = _timed(
            "review", state["round"],
            lambda: step("reviewer", _meta(state), agents.review, report.sentences, state["notes"]),
        )
        issues = [i.message for i in result.issues]
        trace.issues = issues
        # Sentences the Writer itself dropped are worth fixing too, so they don't come back.
        dropped = [f"This sentence was dropped ({r})" for r in report.rejected]
        return {
            "passed": result.passed,
            "issues": issues,
            "feedback": issues + dropped,
            "trace": [trace],
        }

    def finish_step(state: State) -> dict:
        report = state["report"].model_copy(update={
            "evidence": state["notes"].evidence,
            "passed": state["passed"],
            "issues": state["issues"],
            "trace": state["trace"],
        })
        return {"report": report}

    return {
        "research": research_step,
        "analyze": analyze_step,
        "write": write_step,
        "review": review_step,
        "finish": finish_step,
    }


def _decide(passed: bool, round_no: int) -> Literal["write", "finish"]:
    if passed or round_no >= 1 + MAX_RETRIES:
        return "finish"
    return "write"


def after_review(state: State) -> Literal["write", "finish"]:
    """Send the draft back to the Writer if it failed and rewrites are left. Otherwise finish."""
    return step("coordinator", _meta(state), _decide, state["passed"], state["round"])
