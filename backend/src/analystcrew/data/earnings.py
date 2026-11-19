"""Reads a company's newest earnings press release, for a quarter the 10-Q hasn't covered yet.

Why: a company announces its quarter in an 8-K (an "earnings release") weeks before it files
the 10-Q. The SEC's number database (XBRL) only fills in once the 10-Q arrives, so without
this step a report written in that gap would quietly use the quarter before.

How:
1. In the filing list, the newest 8-K with item code "2.02" is the earnings release.
2. Its exhibit EX-99.1 is the press release itself (an HTML page).
3. We find the income statement table in it ("In millions", "Three Months Ended") and read
   the first number on each row we need: that column is the quarter just reported.
4. Safety check: the revenue in the table must match the "$X billion" in the headline.
   If anything looks wrong, we return None and the caller keeps the 10-Q figures.
   We never guess a number.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime

from pydantic import BaseModel

from .sec import Filing, _download_text, exhibit_url, list_filings

EARNINGS_ITEM = "2.02"  # the SEC's 8-K item code for "Results of Operations"

# Rows of the income statement we read, as (metric name, row label pattern).
# The first match after the table heading wins, so order inside the table doesn't matter.
ROWS = {
    "revenue": r"(?:Total\s+)?(?:net\s+)?revenues?",
    "gross_profit": r"Gross\s+(?:profit|margin)",
    "operating_income": r"(?:Operating\s+income|Income\s+from\s+operations)(?:\s*\(loss\))?",
    "net_income": r"Net\s+income(?:\s*\(loss\))?",
}
_AMOUNT = r"\s*\$?\s*(\(?\d[\d,]*(?:\.\d+)?\)?)"
_SCALES = {"millions": 1e6, "thousands": 1e3}


class EarningsRelease(BaseModel):
    """Figures from one earnings press release. Values are in USD (EPS in USD per share)."""

    filing: Filing  # the 8-K the release came from
    url: str  # link to the press release itself
    period_end: str  # last day of the quarter it reports, e.g. "2026-10-25"
    values: dict[str, float]  # metric name -> value; a metric we couldn't read is left out


def latest_earnings_filing(cik: int) -> Filing | None:
    """The newest 8-K that announces earnings (item 2.02), or None."""
    for f in list_filings(cik, forms=("8-K",), limit=40):
        if EARNINGS_ITEM in f.items.split(","):
            return f
    return None


def plain_text(page: str) -> str:
    """HTML -> one line of plain text: tags removed, &amp; etc. decoded, spaces collapsed."""
    text = re.sub(r"<[^>]+>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _number(raw: str) -> float:
    """'1,234' -> 1234.0 and '(1,234)' -> -1234.0 (brackets mean a loss)."""
    value = float(raw.strip("()").replace(",", ""))
    return -value if raw.startswith("(") else value


def _period_end(text: str) -> str | None:
    """The quarter-end date from the headline: "...quarter ended October 25, 2026..."."""
    found = re.search(r"quarter ended ([A-Z][a-z]+\.? \d{1,2}, \d{4})", text)
    if not found:
        return None
    for fmt in ("%B %d, %Y", "%b. %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(found.group(1), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _income_statement(text: str) -> tuple[str, float] | None:
    """The income statement part of the text, and its scale (1e6 for "In millions")."""
    for heading in re.finditer(r"STATEMENTS? OF (?:INCOME|OPERATIONS|EARNINGS)", text, re.I):
        table = text[heading.start(): heading.start() + 5000]
        top = table[:400]
        scale = re.search(r"In (millions|thousands)", top, re.I)
        if scale and re.search(r"Three Months Ended", top, re.I):
            return table, _SCALES[scale.group(1).lower()]
    return None


def _headline_revenue(text: str) -> float | None:
    """Revenue from the opening paragraph: "...revenue ... of $115.8 billion" -> 115.8e9."""
    found = re.search(r"revenue[^.$]{0,150}?of \$(\d+(?:\.\d+)?) billion", text[:3000], re.I)
    return float(found.group(1)) * 1e9 if found else None


def parse_release(page: str) -> tuple[str, dict[str, float]] | None:
    """Read the quarter-end date and the quarter's figures from a press release page.

    Returns (period_end, values), or None if the page can't be read safely."""
    text = plain_text(page)
    period_end = _period_end(text)
    statement = _income_statement(text)
    if period_end is None or statement is None:
        return None
    table, scale = statement

    values: dict[str, float] = {}
    for metric, label in ROWS.items():
        row = re.search(rf"\b{label}{_AMOUNT}", table, re.I)
        if row:
            values[metric] = _number(row.group(1)) * scale
    per_share = re.search(r"per share", table, re.I)
    if per_share:
        eps = re.search(rf"\bDiluted{_AMOUNT}", table[per_share.end():], re.I)
        if eps:
            values["eps_diluted"] = _number(eps.group(1))

    if "revenue" not in values:
        return None
    headline = _headline_revenue(text)
    if headline is not None and abs(values["revenue"] - headline) > 0.051e9:
        return None  # the table and the headline disagree: don't trust either
    return period_end, values


def latest_release(cik: int) -> EarningsRelease | None:
    """The newest earnings release with its figures, or None if there is none we can read."""
    filing = latest_earnings_filing(cik)
    if filing is None:
        return None
    url = exhibit_url(cik, filing.accession)
    page = _download_text(url) if url else None
    parsed = parse_release(page) if page else None
    if parsed is None:
        return None
    period_end, values = parsed
    return EarningsRelease(filing=filing, url=url, period_end=period_end, values=values)


def is_newer(release: EarningsRelease, period_end: str) -> bool:
    """True if the release covers a later quarter than the one ending on period_end."""
    return date.fromisoformat(release.period_end) > date.fromisoformat(period_end)
