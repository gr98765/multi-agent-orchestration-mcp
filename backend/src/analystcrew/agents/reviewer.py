"""The Reviewer: checks a report against the evidence, using plain Python rules (no model).

A report fails if any sentence:
1. cites no evidence, or cites an evidence ID that does not exist. One exception: a sentence
   that only says something was "Not Found" (and has no numbers) needs no evidence, because
   there is nothing to cite for missing data;
2. contains a number that cannot be found in the evidence it cites. A number counts as
   found if it is written in that evidence, or if it is a percentage calculated from two
   cited numbers (a / b * 100, like a margin, or (a - b) / b * 100, like growth);
3. relies on news older than 180 days, or news with no date (we can't tell how old it is).

The whole report also fails if it is out of date: the filings or news show a newer quarter
than the one the report's financials cover (see newer_quarter below).

Years, dates and form names like "10-K" are numbers too, so they must match the evidence.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from itertools import permutations
from typing import Literal

from pydantic import BaseModel, Field

from ..models import Claim, Evidence, ResearchNotes

NEWS_MAX_AGE_DAYS = 180
QUARTER_DAYS = 91  # about how long a fiscal quarter is
EARNINGS_LAG_DAYS = 100  # an earnings release comes out within ~100 days of its quarter's end
TOLERANCE_DAYS = 7  # period end dates can differ by a few days and still be the same quarter

_ID = re.compile(r"\bE\d+\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

Check = Literal[
    "no_evidence", "unknown_id", "unsupported_number", "stale_news", "undated_news",
    "out_of_date",
]


class Issue(BaseModel):
    sentence: int | None  # position of the sentence, starting at 0; None = the whole report
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


# ---------- is the report out of date? ----------

_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_Q = r"(?:\bQ([1-4])\b|\b(first|second|third|fourth)[\s-]+quarter)"
_FY = r"(?:\bFY\s*'?|\bfiscal\s+(?:year\s+)?'?)(\d{4}|\d{2})\b"
_QUARTER_THEN_YEAR = re.compile(_Q + r"\s*(?:of\s+)?(?:the\s+)?" + _FY, re.I)  # "Q3 FY27"
_YEAR_THEN_QUARTER = re.compile(_FY + r"\s+" + _Q, re.I)  # "fiscal 2027 third quarter"


def quarters_named(text: str) -> list[tuple[int, int]]:
    """Fiscal quarters a text names, as (year, quarter): "Q3 FY27" -> (2027, 3).

    News is untrusted: we only pick out quarter names with a pattern, nothing else."""
    found = []
    for m in _QUARTER_THEN_YEAR.finditer(text):
        q, word, year = m.groups()
        found.append((year, q, word))
    for m in _YEAR_THEN_QUARTER.finditer(text):
        year, q, word = m.groups()
        found.append((year, q, word))
    out = []
    for year, q, word in found:
        y = int(year) + (2000 if len(year) == 2 else 0)
        out.append((y, int(q) if q else _QUARTER_WORDS[word.lower()]))
    return out


def newer_quarter(notes: ResearchNotes) -> str | None:
    """Describe a newer quarter that the filings or news show, or None if the report is current.

    - a 10-K/10-Q covering a later period than the report;
    - an earnings release (8-K) filed more than EARNINGS_LAG_DAYS after the report's quarter
      ended: too late to be about that quarter, so it announces a later one;
    - news naming a later fiscal quarter that had already ended when the article came out
      (so "Q4 guidance" before Q4 is over doesn't count: that is a forecast, not results).
    """
    if notes.period_end is None or notes.fiscal_year is None or notes.fiscal_quarter is None:
        return None  # we don't know which quarter the report covers
    end = date.fromisoformat(notes.period_end)
    label = f"FY{notes.fiscal_year} Q{notes.fiscal_quarter}"

    for ev in notes.evidence:
        if ev.kind == "filing":
            period = _parse_date(ev.period_end)
            if period and (period - end).days > TOLERANCE_DAYS:
                return f"the quarter ended {period} ({ev.form} filed {ev.published_date})"
            filed = _parse_date(ev.published_date)
            if ev.earnings_release and filed and (filed - end).days > EARNINGS_LAG_DAYS:
                return f"the quarter after {label} (earnings release filed {filed})"
        elif ev.kind == "news":
            published = _parse_date(ev.published_date)
            if published is None:
                continue
            for year, quarter in quarters_named(f"{ev.summary} {ev.value or ''}"):
                steps = (year - notes.fiscal_year) * 4 + (quarter - notes.fiscal_quarter)
                if steps > 0 and published > end + timedelta(QUARTER_DAYS * steps):
                    return f"FY{year} Q{quarter} (named in news {ev.id}, published {published})"
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

    newer = newer_quarter(notes)
    if newer:
        issues.append(Issue(
            sentence=None, check="out_of_date",
            message=f"Report may be out of date: it covers FY{notes.fiscal_year} "
                    f"Q{notes.fiscal_quarter}, but {newer} is newer.",
        ))

    return Review(passed=not issues, issues=issues)
