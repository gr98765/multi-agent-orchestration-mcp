"""Turn SEC XBRL "company facts" into clean, cited metrics for a fiscal year or quarter.

Real-world wrinkles handled here (good interview material):
* Companies don't file a Q4 report. Q4 lives inside the annual 10-K, so Q4 flow metrics are
  derived as (full year) - (nine-month year-to-date), or full year minus Q1..Q3.
* 10-Qs report both 3-month and year-to-date figures; cash-flow items are often YTD only.
* A 10-K's comparative (prior-year) figures carry the *filing's* fiscal-year tag, so the
  `fy` field alone can't tell you which year a number belongs to. We use period end dates.
* Companies switch XBRL concepts over time (e.g. "Revenues" -> "RevenueFromContract..."),
  so every metric has an ordered list of fallback concepts.
* Some metrics can't be derived by subtraction (EPS: share counts change). We never guess:
  anything we can't source is returned as status="not_found" with an explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel

from .sec import filing_folder_url, find_company, get_company_facts

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)
TOLERANCE_DAYS = 7


@dataclass(frozen=True)
class MetricSpec:
    concepts: tuple[str, ...]
    unit: str
    kind: Literal["flow", "instant"]
    quarter_derivable: bool = True


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
        "USD",
        "flow",
    ),
    "gross_profit": MetricSpec(("GrossProfit",), "USD", "flow"),
    "operating_income": MetricSpec(("OperatingIncomeLoss",), "USD", "flow"),
    "net_income": MetricSpec(("NetIncomeLoss",), "USD", "flow"),
    "eps_diluted": MetricSpec(
        ("EarningsPerShareDiluted",), "USD/shares", "flow", quarter_derivable=False
    ),
    "operating_cash_flow": MetricSpec(
        ("NetCashProvidedByUsedInOperatingActivities",), "USD", "flow"
    ),
    "cash_and_equivalents": MetricSpec(
        ("CashAndCashEquivalentsAtCarryingValue",), "USD", "instant"
    ),
}

EPS_NOTE = (
    "EPS is only used when directly reported; it can't be derived by subtraction "
    "because share counts change between periods."
)


class Source(BaseModel):
    form: str
    accession: str
    filed: str
    url: str


class MetricValue(BaseModel):
    metric: str
    status: Literal["reported", "derived", "not_found"]
    value: float | None = None
    unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    concept: str | None = None
    sources: list[Source] = []
    note: str | None = None


class FinancialReport(BaseModel):
    ticker: str = ""
    company: str
    cik: int
    fiscal_year: int
    fiscal_quarter: int | None
    period_start: str
    period_end: str
    metrics: list[MetricValue]
    notes: list[str] = []


@dataclass
class FiscalYear:
    year: int
    start: str
    end: str
    quarter_ends: list[str]
    complete: bool = True  # False = year in progress: only 10-Qs filed so far


# ---------- helpers ----------


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _close(a: str, b: str) -> bool:
    return abs((_d(a) - _d(b)).days) <= TOLERANCE_DAYS


def _days(f: dict) -> int:
    return (_d(f["end"]) - _d(f["start"])).days


def _in(days: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= days <= bounds[1]


def _facts(cf: dict, concept: str, unit: str) -> list[dict]:
    """Facts for one concept from 10-K/10-Q filings, one per period (latest filing wins)."""
    rows = cf.get("facts", {}).get("us-gaap", {}).get(concept,
                                                      {}).get("units", {}).get(unit, [])
    best: dict[tuple, dict] = {}
    for r in rows:
        if not str(r.get("form", "")).startswith(("10-K", "10-Q")):
            continue
        key = (r.get("start"), r["end"])
        if key not in best or r["filed"] > best[key]["filed"]:
            best[key] = r
    return list(best.values())


def _raw_facts(cf: dict, concept: str, unit: str) -> list[dict]:
    rows = cf.get("facts", {}).get("us-gaap", {}).get(concept,
                                                      {}).get("units", {}).get(unit, [])
    return [r for r in rows if str(r.get("form", "")).startswith(("10-K", "10-Q"))]


def _flow_anchors() -> list[tuple[str, str]]:
    return [(c, s.unit) for s in METRICS.values() if s.kind == "flow" for c in s.concepts]


def _source(cf: dict, f: dict) -> Source:
    return Source(
        form=f["form"],
        accession=f["accn"],
        filed=f["filed"],
        url=filing_folder_url(int(cf["cik"]), f["accn"]),
    )


def _find_duration(facts: list[dict], start: str, end: str) -> dict | None:
    for f in facts:
        if f.get("start") and _close(f["start"], start) and _close(f["end"], end):
            return f
    return None


# ---------- fiscal calendar ----------


def available_years(cf: dict) -> list[int]:
    """Fiscal years with a 10-K, plus the in-progress year if it has 10-Qs."""
    years: set[int] = set()
    for concept, unit in _flow_anchors():
        for f in _raw_facts(cf, concept, unit):
            if not f.get("fy") or not f.get("start"):
                continue
            if f["form"].startswith("10-K") and f.get("fp") == "FY" and _in(_days(f), ANNUAL_DAYS):
                years.add(int(f["fy"]))
            elif f["form"].startswith("10-Q") and f.get("fp") == "Q1":
                years.add(int(f["fy"]))
    return sorted(years)


def _cluster(ends: list[str]) -> list[str]:
    out: list[str] = []
    for e in sorted(set(ends)):
        if not out or not _close(out[-1], e):
            out.append(e)
    return out


def _quarter_ends(cf: dict, start: str, limit: date, forms: tuple[str, ...]) -> list[str]:
    ends: list[str] = []
    for concept, unit in _flow_anchors():
        for f in _facts(cf, concept, unit):
            if not f.get("start") or not f["form"].startswith(forms):
                continue
            if not (_d(start) < _d(f["end"]) <= limit):
                continue
            if _close(f["start"], start) or _in(_days(f), QUARTER_DAYS):
                ends.append(f["end"])
    return _cluster(ends)


def fiscal_calendar(cf: dict, year: int) -> FiscalYear | None:
    # The current-year window is the annual fact tagged fy=year with the LATEST end date;
    # earlier ones with the same tag are prior-year comparatives.
    candidates = []
    for concept, unit in _flow_anchors():
        for f in _raw_facts(cf, concept, unit):
            if (
                f["form"].startswith("10-K")
                and f.get("fy") == year
                and f.get("start")
                and _in(_days(f), ANNUAL_DAYS)
            ):
                candidates.append((f["end"], f["start"]))
    if not candidates:
        return _in_progress_calendar(cf, year)
    end, start = max(candidates)
    limit = _d(end) + timedelta(TOLERANCE_DAYS)
    return FiscalYear(
        year=year, start=start, end=end, quarter_ends=_quarter_ends(
            cf, start, limit, ("10-",))
    )


def _in_progress_calendar(cf: dict, year: int) -> FiscalYear | None:
    """A fiscal year with no 10-K yet. Its Q1 is the fy=year, fp=Q1 fact with the latest end
    date (older ones in the same filing are prior-year comparatives)."""
    q1 = []
    for concept, unit in _flow_anchors():
        for f in _raw_facts(cf, concept, unit):
            if (
                f["form"].startswith("10-Q")
                and f.get("fy") == year
                and f.get("fp") == "Q1"
                and f.get("start")
                and _in(_days(f), QUARTER_DAYS)
            ):
                q1.append((f["end"], f["start"]))
    if not q1:
        return None
    _, start = max(q1)
    ends = _quarter_ends(cf, start, _d(start) +
                         timedelta(ANNUAL_DAYS[1]), ("10-Q",))[:3]
    if not ends:
        return None
    return FiscalYear(year=year, start=start, end=ends[-1], quarter_ends=ends, complete=False)


# ---------- metric extraction ----------


def _metric(
    cf: dict, name: str, spec: MetricSpec, fy: FiscalYear, quarter: int | None
) -> MetricValue:
    if quarter is not None and not fy.complete and quarter > len(fy.quarter_ends):
        return MetricValue(
            metric=name,
            status="not_found",
            note=f"Q{quarter} FY{fy.year} has not been filed yet "
            f"(latest filed: Q{len(fy.quarter_ends)}).",
        )
    if quarter is not None and fy.complete and len(fy.quarter_ends) != 4:
        return MetricValue(
            metric=name,
            status="not_found",
            note=f"Could not map fiscal {fy.year} into 4 quarters from the filings.",
        )
    if quarter is None:
        p_start, p_end, prev_end = fy.start, fy.end, None
    else:
        p_end = fy.quarter_ends[quarter - 1]
        prev_end = fy.quarter_ends[quarter - 2] if quarter > 1 else None
        p_start = (_d(prev_end) + timedelta(1)
                   ).isoformat() if prev_end else fy.start

    base = {"metric": name, "unit": spec.unit,
            "period_start": p_start, "period_end": p_end}

    for concept in spec.concepts:
        facts = _facts(cf, concept, spec.unit)
        if not facts:
            continue

        if spec.kind == "instant":
            hits = [f for f in facts if not f.get(
                "start") and _close(f["end"], p_end)]
            if hits:
                f = min(hits, key=lambda x: abs(
                    (_d(x["end"]) - _d(p_end)).days))
                return MetricValue(
                    **{**base, "period_start": None, "period_end": f["end"]},
                    status="reported",
                    value=f["val"],
                    concept=concept,
                    sources=[_source(cf, f)],
                )
            continue

        if quarter is None:
            f = _find_duration(facts, fy.start, fy.end)
            if f and _in(_days(f), ANNUAL_DAYS):
                return MetricValue(
                    **base,
                    status="reported",
                    value=f["val"],
                    concept=concept,
                    sources=[_source(cf, f)],
                )
            continue

        # 1) directly reported 3-month figure
        direct = [
            f
            for f in facts
            if f.get("start") and _close(f["end"], p_end) and _in(_days(f), QUARTER_DAYS)
        ]
        if direct:
            f = direct[0]
            return MetricValue(
                **base,
                status="reported",
                value=f["val"],
                concept=concept,
                sources=[_source(cf, f)],
            )
        if not spec.quarter_derivable or prev_end is None:
            continue

        ytd_now = _find_duration(facts, fy.start, p_end)
        if not ytd_now:
            continue
        # 2) year-to-date difference, e.g. Q4 = full year - nine months YTD
        ytd_prev = _find_duration(facts, fy.start, prev_end)
        if ytd_prev:
            return MetricValue(
                **base,
                status="derived",
                value=ytd_now["val"] - ytd_prev["val"],
                concept=concept,
                sources=[_source(cf, ytd_now), _source(cf, ytd_prev)],
                note=(
                    f"Derived: year-to-date through {p_end} minus year-to-date through {prev_end}."
                ),
            )
        # 3) full period minus each earlier quarter's 3-month figure
        earlier = []
        for i in range(quarter - 1):
            qe = fy.quarter_ends[i]
            hit = [
                f
                for f in facts
                if f.get("start") and _close(f["end"], qe) and _in(_days(f), QUARTER_DAYS)
            ]
            if not hit:
                break
            earlier.append(hit[0])
        if len(earlier) == quarter - 1:
            return MetricValue(
                **base,
                status="derived",
                value=ytd_now["val"] - sum(f["val"] for f in earlier),
                concept=concept,
                sources=[_source(cf, ytd_now)] + [_source(cf, f)
                                                  for f in earlier],
                note=f"Derived: year-to-date through {p_end} minus Q1-Q{quarter - 1} figures.",
            )

    note = (
        EPS_NOTE
        if name == "eps_diluted" and quarter
        else "Not reported in SEC XBRL data for this period."
    )
    return MetricValue(**base, status="not_found", note=note)


def _has_quarter(fy: FiscalYear | None, quarter: int) -> bool:
    if fy is None:
        return False
    return len(fy.quarter_ends) == 4 if fy.complete else quarter <= len(fy.quarter_ends)


def build_report(
    cf: dict,
    fiscal_year: int | None = None,
    quarter: int | None = None,
    latest_quarter: bool = False,
) -> FinancialReport:
    if quarter is not None and quarter not in (1, 2, 3, 4):
        raise ValueError(
            "fiscal_quarter must be 1, 2, 3, 4, or omitted for the full year.")
    years = available_years(cf)
    if not years:
        raise ValueError(
            "No 10-K or 10-Q financial data found for this company.")
    calendars = {y: fiscal_calendar(cf, y) for y in years}

    if latest_quarter:  # the most recently filed quarter, whichever year it's in
        year = next(y for y in reversed(years) if calendars[y])
        c = calendars[year]
        quarter = len(c.quarter_ends) if not c.complete else 4
    elif fiscal_year is not None:
        year = fiscal_year
    elif quarter is None:  # latest full year
        year = next(
            (y for y in reversed(years)
             if calendars[y] and calendars[y].complete), years[-1]
        )
    else:  # latest year in which this quarter has been filed
        year = next((y for y in reversed(years) if _has_quarter(
            calendars[y], quarter)), years[-1])

    fy = calendars.get(year) or fiscal_calendar(cf, year)
    if fy is None:
        raise ValueError(
            f"No 10-K or 10-Q data for fiscal year {year}. Available: {years[-10:]}")
    if quarter is None and not fy.complete:
        raise ValueError(
            f"Fiscal {year} is still in progress (Q1-Q{len(fy.quarter_ends)} filed). "
            "Ask for a specific quarter, or use latest_quarter."
        )

    metrics = [_metric(cf, name, spec, fy, quarter)
               for name, spec in METRICS.items()]
    span = (
        f"runs {fy.start} to {fy.end}"
        if fy.complete
        else f"began {fy.start}; filed so far: Q1-Q{len(fy.quarter_ends)}"
    )
    notes = [
        f"Fiscal {year} {span}.",
        "Values are in the listed unit (USD, not millions). status='not_found' means no "
        "official figure exists in the data; never substitute an estimate.",
    ]
    if quarter == 4:
        notes.append(
            "Companies don't file a Q4 report; Q4 figures come from the annual 10-K.")
    ref = next((m for m in metrics if m.period_start), None)
    return FinancialReport(
        company=cf.get("entityName", ""),
        cik=int(cf["cik"]),
        fiscal_year=year,
        fiscal_quarter=quarter,
        period_start=ref.period_start if ref else fy.start,
        period_end=ref.period_end if ref else fy.end,
        metrics=metrics,
        notes=notes,
    )


# ---------- the one function most code will call ----------


def get_financials(
    company: str,
    fiscal_year: int | None = None,
    quarter: int | None = None,
    latest_quarter: bool = False,
) -> FinancialReport:
    """Official numbers for a company and period, e.g. get_financials("NVDA", latest_quarter=True).

    company: a ticker ("NVDA") or name ("nvidia")
    fiscal_year + quarter: a specific period; quarter=None means the full year
    latest_quarter: the most recently filed quarter (no need to know the fiscal calendar)
    """
    found = find_company(company)
    if found is None:
        raise ValueError(
            f"Not Found: no SEC-registered company matches '{company}'.")
    facts = get_company_facts(found.cik)
    if not facts:
        raise ValueError(
            f"Not Found: the SEC has no financial data for {found.name}.")
    report = build_report(facts, fiscal_year, quarter, latest_quarter)
    return report.model_copy(update={"ticker": found.ticker, "company": found.name})
