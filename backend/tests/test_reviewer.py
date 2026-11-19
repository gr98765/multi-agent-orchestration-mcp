"""Tests for the Reviewer: deliberately bad sentences must fail, honest ones must pass."""

from datetime import date

import pytest

from analystcrew.agents.analyst import analyze
from analystcrew.agents.reviewer import review
from analystcrew.models import Claim, ResearchNotes

TODAY = date(2026, 11, 16)


@pytest.fixture
def notes() -> ResearchNotes:
    n = ResearchNotes(company="NVDA")
    n.add("financial", "revenue (reported)", value="$35.08B", number=35_082_000_000)  # E1
    n.add("financial", "gross_profit (reported)", value="$26.16B", number=26_156_000_000)  # E2
    n.add("financial", "revenue_prior_year (reported)", value="$18.12B",
          number=18_120_000_000)  # E3
    n.add("filing", "10-K filed 2025-02-26", url="https://www.sec.gov/fake")  # E4
    n.add("news", "Nvidia beats estimates", value="Sales rose.", untrusted=True,
          published_date="2026-10-01")  # E5: 46 days old
    n.add("news", "Old chip story", value="...", untrusted=True,
          published_date="2026-03-01")  # E6: 260 days old
    n.add("news", "Undated rumor", value="...", untrusted=True)  # E7: no date
    n.add("news", "Fresh RFC-dated story", value="...", untrusted=True,
          published_date="Mon, 09 Nov 2026 10:00:00 GMT")  # E8: 7 days old
    return n


def checks(result) -> list[str]:
    return [i.check for i in result.issues]


# ---------- honest reports pass ----------


def test_analyst_claims_pass(notes):
    result = review(analyze(notes), notes, today=TODAY)

    assert result.passed, result.issues


def test_honest_sentences_pass(notes):
    sentences = [
        Claim(text="Revenue was $35.08B.", evidence_ids=["E1"]),
        Claim(text="Gross margin was 74.6%.", evidence_ids=["E2", "E1"]),
        Claim(text="Revenue grew 93.6% from a year earlier.", evidence_ids=["E1", "E3"]),
        Claim(text="The annual 10-K was filed on 2025-02-26.", evidence_ids=["E4"]),
        Claim(text="Recent coverage says Nvidia beat estimates.", evidence_ids=["E5"]),
        Claim(text="A story from last week discussed chips.", evidence_ids=["E8"]),
    ]

    result = review(sentences, notes, today=TODAY)

    assert result.passed, result.issues


# ---------- check 1: evidence IDs ----------


def test_sentence_without_evidence_fails(notes):
    result = review([Claim(text="Nvidia is the best company.")], notes, today=TODAY)

    assert not result.passed
    assert checks(result) == ["no_evidence"]
    assert result.issues[0].sentence == 0


def test_unknown_evidence_id_fails(notes):
    sentences = [
        Claim(text="Revenue was $35.08B.", evidence_ids=["E1"]),
        Claim(text="Revenue was $35.08B.", evidence_ids=["E1", "E99"]),
    ]

    result = review(sentences, notes, today=TODAY)

    assert not result.passed
    assert checks(result) == ["unknown_id"]
    assert result.issues[0].sentence == 1
    assert "E99" in result.issues[0].message


def test_not_found_sentence_needs_no_evidence(notes):
    result = review([Claim(text="Operating margin: Not Found.")], notes, today=TODAY)

    assert result.passed


def test_not_found_sentence_with_a_number_still_needs_evidence(notes):
    result = review([Claim(text="Free cash flow: Not Found, but likely $5B.")], notes,
                    today=TODAY)

    assert checks(result) == ["no_evidence"]


# ---------- check 2: numbers ----------


def test_wrong_number_fails(notes):
    result = review([Claim(text="Revenue was $36.00B.", evidence_ids=["E1"])], notes,
                    today=TODAY)

    assert checks(result) == ["unsupported_number"]
    assert "36.00" in result.issues[0].message


def test_wrong_margin_fails(notes):
    result = review([Claim(text="Gross margin was 80.0%.", evidence_ids=["E2", "E1"])], notes,
                    today=TODAY)

    assert checks(result) == ["unsupported_number"]


def test_number_from_evidence_that_is_not_cited_fails(notes):
    # $35.08B is real (E1), but this sentence only cites E2
    result = review([Claim(text="Revenue was $35.08B.", evidence_ids=["E2"])], notes,
                    today=TODAY)

    assert checks(result) == ["unsupported_number"]


def test_made_up_year_fails(notes):
    result = review([Claim(text="The 10-K was filed in 2024.", evidence_ids=["E4"])], notes,
                    today=TODAY)

    assert checks(result) == ["unsupported_number"]


def test_every_bad_number_is_listed(notes):
    sentence = Claim(text="Revenue was $35.08B, up 50.0% to a record 12 quarters.",
                     evidence_ids=["E1", "E3"])

    result = review([sentence], notes, today=TODAY)

    assert [i.message.split(" ")[0] for i in result.issues] == ["50.0", "12"]


# ---------- check 3: news age ----------


def test_old_news_fails(notes):
    result = review([Claim(text="Chips were discussed.", evidence_ids=["E6"])], notes,
                    today=TODAY)

    assert checks(result) == ["stale_news"]
    assert "260 days old" in result.issues[0].message


def test_undated_news_fails(notes):
    result = review([Claim(text="A rumor circulated.", evidence_ids=["E7"])], notes,
                    today=TODAY)

    assert checks(result) == ["undated_news"]


def test_news_exactly_180_days_old_passes(notes):
    notes.evidence[4].published_date = "2026-05-20"  # E5, 180 days before TODAY

    result = review([Claim(text="Coverage was positive.", evidence_ids=["E5"])], notes,
                    today=TODAY)

    assert result.passed


def test_issues_from_several_sentences_are_all_reported(notes):
    sentences = [
        Claim(text="No source here."),
        Claim(text="Revenue was $99.99B.", evidence_ids=["E1"]),
        Claim(text="Old news.", evidence_ids=["E6", "E42"]),
    ]

    result = review(sentences, notes, today=TODAY)

    assert [(i.sentence, i.check) for i in result.issues] == [
        (0, "no_evidence"),
        (1, "unsupported_number"),
        (2, "unknown_id"),
        (2, "stale_news"),
    ]


# ---------- out-of-date reports ----------

HONEST = [Claim(text="Revenue was $35.08B.", evidence_ids=["E1"])]


@pytest.fixture
def q3_notes() -> ResearchNotes:
    """A report on FY2025 Q3 (quarter ended 2024-10-27), with its own 10-Q and 8-K."""
    n = ResearchNotes(company="NVDA", fiscal_year=2025, fiscal_quarter=3,
                      period_end="2024-10-27")
    n.add("financial", "revenue (reported)", value="$35.08B", number=35_082_000_000)  # E1
    n.add("filing", "10-Q filed 2024-11-20", published_date="2024-11-20", form="10-Q",
          period_end="2024-10-27")  # E2: the quarter the report covers
    n.add("filing", "8-K filed 2024-11-20 (earnings release)", published_date="2024-11-20",
          form="8-K", earnings_release=True)  # E3: this quarter's own earnings release
    return n


def out_of_date(result) -> list[str]:
    return [i.message for i in result.issues if i.check == "out_of_date"]


def test_current_report_is_not_out_of_date(q3_notes):
    assert review(HONEST, q3_notes, today=date(2024, 12, 1)).passed


def test_newer_10k_flags_stale_period(q3_notes):
    q3_notes.add("filing", "10-K filed 2025-02-26", published_date="2025-02-26", form="10-K",
                 period_end="2025-01-26")

    result = review(HONEST, q3_notes, today=date(2025, 3, 1))

    assert not result.passed
    [message] = out_of_date(result)
    assert message.startswith("Report may be out of date")
    assert "FY2025 Q3" in message and "quarter ended 2025-01-26" in message
    assert result.issues[-1].sentence is None  # about the whole report, not one sentence


def test_later_earnings_release_flags_stale_period(q3_notes):
    """An earnings 8-K filed 122 days after Q3 ended must be about a later quarter."""
    q3_notes.add("filing", "8-K filed 2025-02-26 (earnings release)",
                 published_date="2025-02-26", form="8-K", earnings_release=True)

    [message] = out_of_date(review(HONEST, q3_notes, today=date(2025, 3, 1)))

    assert "the quarter after FY2025 Q3 (earnings release filed 2025-02-26)" in message


def test_non_earnings_8k_is_ignored(q3_notes):
    q3_notes.add("filing", "8-K filed 2025-06-30", published_date="2025-06-30", form="8-K")

    assert review(HONEST, q3_notes, today=date(2025, 7, 1)).passed


def test_news_about_newer_quarters_results_flags_stale_period(q3_notes):
    q3_notes.add("news", "Nvidia fourth-quarter fiscal 2025 results beat estimates",
                 value="...", untrusted=True, published_date="2025-02-27")  # E4

    [message] = out_of_date(review(HONEST, q3_notes, today=date(2025, 3, 1)))

    assert "FY2025 Q4 (named in news E4, published 2025-02-27)" in message


def test_news_forecasting_a_future_quarter_is_not_stale(q3_notes):
    """Q4 FY25 guidance written before Q4 ended is a forecast, not newer results."""
    q3_notes.add("news", "Nvidia guides Q4 FY25 revenue higher", value="Outlook for Q4 FY2025.",
                 untrusted=True, published_date="2024-11-21")

    assert review(HONEST, q3_notes, today=date(2024, 12, 1)).passed


def test_unknown_report_period_skips_the_check(notes):
    notes.add("filing", "10-K filed 2026-02-25", published_date="2026-02-25", form="10-K",
              period_end="2026-01-25")

    assert out_of_date(review(HONEST, notes, today=TODAY)) == []
