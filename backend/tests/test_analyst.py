"""Tests for the Analyst: margins are calculated in Python and cite the right evidence IDs.

Each test builds small ResearchNotes by hand, so nothing touches the internet or a model."""

from analystcrew.agents.analyst import analyze
from analystcrew.models import ResearchNotes

B = 1_000_000_000


def notes_with(**metrics) -> ResearchNotes:
    """Notes holding one financial fact per metric, e.g. revenue=100 * B."""
    notes = ResearchNotes(company="NVDA")
    for metric, number in metrics.items():
        notes.add("financial", f"{metric} (reported)", value=f"${number / B:.2f}B", number=number)
    return notes


def texts(claims) -> list[str]:
    return [c.text for c in claims]


def test_normal_case_margins_and_ids():
    notes = notes_with(
        revenue=35_082_000_000,  # E1
        gross_profit=26_156_000_000,  # E2
        operating_income=21_869_000_000,  # E3
        net_income=19_309_000_000,  # E4
        revenue_prior_year=18_120_000_000,  # E5
    )
    notes.add("filing", "10-Q filed 2024-11-20", url="https://www.sec.gov/fake")  # E6, ignored
    notes.add("price", "NVDA move", value="+12.3% (100.00 to 112.30)", number=12.3)  # E7

    claims = analyze(notes)

    assert texts(claims) == [
        "Revenue was $35.08B.",
        "Revenue grew +93.6% year over year ($18.12B to $35.08B).",
        "Gross margin was 74.6%.",
        "Operating margin was 62.3%.",
        "Net margin was 55.0%.",
        "The stock moved +12.3% (100.00 to 112.30) over the period.",
    ]
    assert [c.evidence_ids for c in claims] == [
        ["E1"],
        ["E1", "E5"],
        ["E2", "E1"],
        ["E3", "E1"],
        ["E4", "E1"],
        ["E7"],
    ]


def test_missing_revenue_says_not_found():
    claims = analyze(notes_with(gross_profit=10 * B, revenue_prior_year=5 * B))

    assert texts(claims) == [
        "Revenue: Not Found, so margins could not be calculated.",
        "Year-over-year revenue growth: Not Found.",
    ]
    assert [c.evidence_ids for c in claims] == [[], []]


def test_revenue_without_number_counts_as_missing():
    notes = ResearchNotes(company="NVDA")
    notes.add("financial", "revenue (reported)", value="$1.00B")  # no number recorded

    assert "Not Found" in analyze(notes)[0].text


def test_zero_revenue_does_not_divide():
    claims = analyze(notes_with(revenue=0, gross_profit=10 * B, revenue_prior_year=5 * B))

    assert texts(claims) == [
        "Revenue: Not Found, so margins could not be calculated.",
        "Year-over-year revenue growth: Not Found.",
    ]


def test_missing_margin_input_says_not_found():
    claims = analyze(notes_with(revenue=100 * B, gross_profit=40 * B, net_income=10 * B))

    assert texts(claims) == [
        "Revenue was $100.00B.",
        "Year-over-year revenue growth: Not Found.",
        "Gross margin was 40.0%.",
        "Operating margin: Not Found.",
        "Net margin was 10.0%.",
    ]
    assert claims[3].evidence_ids == []


def test_price_fact_without_number_is_skipped():
    notes = notes_with(revenue=100 * B)
    notes.add("price", "NVDA move", value="n/a")

    assert texts(analyze(notes)) == [
        "Revenue was $100.00B.",
        "Year-over-year revenue growth: Not Found.",
        "Gross margin: Not Found.",
        "Operating margin: Not Found.",
        "Net margin: Not Found.",
    ]


def test_missing_prior_year_growth_says_not_found():
    claims = analyze(notes_with(revenue=100 * B))

    assert claims[1].text == "Year-over-year revenue growth: Not Found."
    assert claims[1].evidence_ids == []


def test_revenue_decline_shows_negative_growth():
    claims = analyze(notes_with(revenue=80 * B, revenue_prior_year=100 * B))

    assert claims[1].text == "Revenue grew -20.0% year over year ($100.00B to $80.00B)."
    assert claims[1].evidence_ids == ["E1", "E2"]


def test_zero_prior_year_revenue_does_not_divide():
    claims = analyze(notes_with(revenue=100 * B, revenue_prior_year=0))

    assert claims[1].text == "Year-over-year revenue growth: Not Found."
