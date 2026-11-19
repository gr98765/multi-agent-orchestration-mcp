"""The Reviewer: checks a report against the evidence, using plain Python rules (no model).

A report fails if any sentence:
1. cites no evidence, or cites an evidence ID that does not exist. One exception: a sentence
   that only says something was "Not Found" (and has no numbers) needs no evidence, because
   there is nothing to cite for missing data;
2. contains a number that cannot be found in the evidence it cites. A number counts as
   found if it is written in that evidence, or if it is a percentage calculated from two
   cited numbers (a / b * 100, like a margin, or (a - b) / b * 100, like growth);
3. relies on news older than 180 days, or news with no date (we can't tell how old it is).

Years, dates and form names like "10-K" are numbers too, so they must match the evidence.
"""

from __future__ import annotations

import re
from datetime import date
from email.utils import parsedate_to_datetime
from itertools import permutations
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Claim, Evidence, ResearchNotes

NEWS_MAX_AGE_DAYS = 180

_ID = re.compile(r"\bE\d+\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

Check = Literal["no_evidence", "unknown_id", "unsupported_number", "stale_news", "undated_news"]


class Issue(BaseModel):
    sentence: int  # position of the sentence in the report, starting at 0
    check: Check
    message: str


class Review(BaseModel):
    passed: bool
    issues: list[Issue] = Field(default_factory=list)


def states_not_found(text: str) -> bool:
    """True for a sentence like "Operating margin: Not Found." that has no numbers in it."""
    return "Not Found" in text and not _NUMBER.search(_ID.sub(" ", text))


# ---------- numbers ----------


def _numbers(text: str) -> list[tuple[float, int]]:
    """Every number in the text as (value, decimal places). Evidence IDs are skipped."""
    found = []
    for raw in _NUMBER.findall(_ID.sub(" ", text)):
        raw = raw.replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        found.append((float(raw), decimals))
    return found


def _allowed(cited: list[Evidence]) -> list[float]:
    """Numbers a sentence may use: those written in the evidence, plus percentages from it."""
    allowed = []
    for ev in cited:
        for text in (ev.summary, ev.value or ""):
            allowed += [value for value, _ in _numbers(text)]
        if ev.number is not None:
            allowed.append(abs(ev.number))
    raw = [ev.number for ev in cited if ev.number is not None]
    for a, b in permutations(raw, 2):
        if b != 0:
            allowed.append(abs(a / b * 100))  # a margin: part of a whole
            allowed.append(abs((a - b) / b * 100))  # growth: change from b to a
    return allowed


def _matches(value: float, decimals: int, allowed: list[float]) -> bool:
    """True if some allowed number rounds to this value at the same number of decimals."""
    return any(abs(round(a, decimals) - value) < 1e-9 for a in allowed)


# ---------- dates ----------


def _parse_date(text: str | None) -> date | None:
    """Read "2026-08-26" or "Wed, 26 Aug 2026 10:00:00 GMT". None if missing or unreadable."""
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        return None


# ---------- the review ----------


def review(sentences: list[Claim], notes: ResearchNotes, today: date | None = None) -> Review:
    """Check every sentence. The report passes only if there are no issues."""
    today = today or date.today()
    by_id = {ev.id: ev for ev in notes.evidence}
    issues: list[Issue] = []

    for i, s in enumerate(sentences):
        def flag(check: Check, message: str, i: int = i) -> None:
            issues.append(Issue(sentence=i, check=check, message=message))

        if not s.evidence_ids:
            if not states_not_found(s.text):
                flag("no_evidence", f"No evidence ID: {s.text}")
            continue
        for eid in s.evidence_ids:
            if eid not in by_id:
                flag("unknown_id", f"{eid} does not exist: {s.text}")
        cited = [by_id[e] for e in s.evidence_ids if e in by_id]

        allowed = _allowed(cited)
        for value, decimals in _numbers(s.text):
            if not _matches(value, decimals, allowed):
                shown = f"{value:.{decimals}f}"
                flag("unsupported_number", f"{shown} is not in the cited evidence: {s.text}")

        for ev in cited:
            if ev.kind != "news":
                continue
            published = _parse_date(ev.published_date)
            if published is None:
                flag("undated_news", f"{ev.id} has no date, so its age is unknown: {s.text}")
            elif (today - published).days > NEWS_MAX_AGE_DAYS:
                age = (today - published).days
                flag("stale_news", f"{ev.id} is {age} days old (limit {NEWS_MAX_AGE_DAYS})")

    return Review(passed=not issues, issues=issues)
