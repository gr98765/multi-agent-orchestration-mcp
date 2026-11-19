"""The Analyst: turns the Researcher's numbers into claims, such as profit margins.

There is no AI model here. All math is done in plain Python, so the numbers are exact.
Every claim lists the evidence IDs (E1, E2, ...) of the facts it was calculated from.
"""

from __future__ import annotations

from ..models import Claim, Evidence, ResearchNotes

# (name used in the report, metric the Researcher recorded)
MARGINS = [
    ("Gross margin", "gross_profit"),
    ("Operating margin", "operating_income"),
    ("Net margin", "net_income"),
]


def _find(notes: ResearchNotes, metric: str) -> Evidence | None:
    """The first financial fact for this metric that has a number, or None."""
    for ev in notes.evidence:
        if ev.kind != "financial" or ev.number is None:
            continue
        if ev.summary.startswith(f"{metric} ("):
            return ev
    return None


def _growth(revenue: Evidence | None, prior: Evidence | None) -> Claim:
    """Year-over-year revenue growth: (this year - last year) / last year * 100."""
    if revenue is None or prior is None or prior.number == 0:
        return Claim(text="Year-over-year revenue growth: Not Found.")
    pct = (revenue.number - prior.number) / prior.number * 100
    return Claim(
        text=f"Revenue grew {pct:+.1f}% year over year ({prior.value} to {revenue.value}).",
        evidence_ids=[revenue.id, prior.id],
    )


def analyze(notes: ResearchNotes) -> list[Claim]:
    """Turn the notes into claims: revenue, growth, margins and price moves."""
    claims: list[Claim] = []

    revenue = _find(notes, "revenue")
    if revenue is None or revenue.number == 0:
        claims.append(Claim(text="Revenue: Not Found, so margins could not be calculated."))
        claims.append(_growth(None, None))  # no usable revenue, so no growth either
    else:
        claims.append(Claim(text=f"Revenue was {revenue.value}.", evidence_ids=[revenue.id]))
        claims.append(_growth(revenue, _find(notes, "revenue_prior_year")))
        for label, metric in MARGINS:
            part = _find(notes, metric)
            if part is None:
                claims.append(Claim(text=f"{label}: Not Found."))
                continue
            pct = part.number / revenue.number * 100
            claims.append(
                Claim(text=f"{label} was {pct:.1f}%.", evidence_ids=[part.id, revenue.id])
            )

    for ev in notes.evidence:
        if ev.kind == "price" and ev.number is not None:
            claims.append(
                Claim(text=f"The stock moved {ev.value} over the period.", evidence_ids=[ev.id])
            )

    return claims
