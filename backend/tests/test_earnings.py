"""Tests for reading earnings releases (8-K), fully offline with a fake SEC.

The fake company's latest 10-Q is FY2026 Q1 (quarter ended 2025-04-27). We add an earnings
8-K and check when its figures replace the 10-Q's, and when they must not."""

import httpx
import pytest
from conftest import CIK, SUBMISSIONS, fake_sec_handler

from analystcrew.data import sec
from analystcrew.data.earnings import parse_release
from analystcrew.data.financials import get_financials

ACCN = "0001045810-25-000200"
FOLDER = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCN.replace('-', '')}/"


def press_release(quarter_end="July 27, 2025", revenue="46,743", headline="46.7") -> str:
    """A short press release shaped like Nvidia's: headline, then the income statement."""
    return f"""<html><body>
    <p>NVIDIA Announces Financial Results for Second Quarter Fiscal 2026</p>
    <p>NVIDIA today reported revenue for the second quarter ended {quarter_end},
    of ${headline} billion, up 6% from the previous quarter.</p>
    <table><tr><td>Revenue</td><td>$46.7</td><td>$44.1</td></tr></table>
    <p>NVIDIA CORPORATION</p><p>CONDENSED CONSOLIDATED STATEMENTS OF INCOME</p>
    <p>(In millions, except per share data)</p>
    <table>
    <tr><td>Three Months Ended</td><td>Six Months Ended</td></tr>
    <tr><td>Revenue</td><td>$</td><td>{revenue}</td><td>$</td><td>30,040</td></tr>
    <tr><td>Cost of revenue</td><td>12,890</td><td>7,466</td></tr>
    <tr><td>Gross profit</td><td>33,853</td><td>22,574</td></tr>
    <tr><td>Operating income</td><td>28,440</td><td>18,642</td></tr>
    <tr><td>Net income</td><td>$</td><td>26,422</td><td>$</td><td>16,599</td></tr>
    <tr><td>Net income per share:</td></tr>
    <tr><td>Basic</td><td>$</td><td>1.08</td><td>$</td><td>0.68</td></tr>
    <tr><td>Diluted</td><td>$</td><td>1.08</td><td>$</td><td>0.67</td></tr>
    </table></body></html>"""


def index_page() -> str:
    """The filing's index page: lists each document with its type."""
    return f"""<table>
    <tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/{CIK}/x/nvda-8k.htm">8-K</a></td>
        <td>8-K</td></tr>
    <tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/{CIK}/x/pr.htm">pr.htm</a>
        </td><td>EX-99.1</td></tr>
    </table>"""


def use_earnings_8k(monkeypatch, filed: str, page: str) -> None:
    """Add an earnings 8-K (item 2.02) filed on `filed`, whose press release is `page`."""
    recent = {key: list(values) for key, values in SUBMISSIONS["filings"]["recent"].items()}
    recent["items"] = ["2.02,9.01", "", "", ""]  # the existing 8-K is not about earnings
    for key, value in [("accessionNumber", ACCN), ("filingDate", filed), ("reportDate", filed),
                       ("form", "8-K"), ("primaryDocument", "nvda-8k.htm"),
                       ("items", "2.02,9.01")]:
        recent[key].insert(0, value)
    submissions = {**SUBMISSIONS, "filings": {"recent": recent}}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith(f"submissions/CIK{CIK:010d}.json"):
            return httpx.Response(200, json=submissions)
        if url == FOLDER + f"{ACCN}-index.html":
            return httpx.Response(200, text=index_page())
        if url.endswith("/x/pr.htm"):
            return httpx.Response(200, text=page)
        return fake_sec_handler(request)

    monkeypatch.setattr(sec, "_client", httpx.Client(transport=httpx.MockTransport(handler)))


def metric(report, name):
    return next(m for m in report.metrics if m.metric == name)


# ---------- reading a press release ----------


def test_parse_release_reads_the_quarter_column():
    period_end, values = parse_release(press_release())

    assert period_end == "2025-07-27"
    assert values == {
        "revenue": 46_743e6, "gross_profit": 33_853e6, "operating_income": 28_440e6,
        "net_income": 26_422e6, "eps_diluted": 1.08,
    }


def test_parse_release_rejects_table_that_disagrees_with_headline():
    assert parse_release(press_release(headline="99.9")) is None


def test_parse_release_without_income_statement_is_none():
    assert parse_release("<p>revenue for the second quarter ended July 27, 2025</p>") is None


# ---------- choosing between the 10-Q and the 8-K ----------


def test_newer_8k_beats_older_10q(monkeypatch):
    use_earnings_8k(monkeypatch, filed="2025-08-27", page=press_release())

    report = get_financials("NVDA", latest_quarter=True)

    assert (report.fiscal_year, report.fiscal_quarter) == (2026, 2)
    assert (report.period_start, report.period_end) == ("2025-04-28", "2025-07-27")
    revenue = metric(report, "revenue")
    assert revenue.status == "reported" and revenue.value == 46_743e6
    [source] = revenue.sources
    assert source.kind == "earnings release"
    assert source.form == "8-K" and source.filed == "2025-08-27"
    assert source.url == f"https://www.sec.gov/Archives/edgar/data/{CIK}/x/pr.htm"
    assert metric(report, "eps_diluted").value == 1.08
    # not in the release: reported as missing, never estimated
    assert metric(report, "cash_and_equivalents").status == "not_found"
    assert "earnings release" in report.notes[0]


def test_older_8k_does_not_replace_10q(monkeypatch):
    """The 8-K for Q1 (the quarter the 10-Q already covers) changes nothing."""
    page = press_release(quarter_end="April 27, 2025", revenue="44,062", headline="44.1")
    use_earnings_8k(monkeypatch, filed="2025-05-28", page=page)

    report = get_financials("NVDA", latest_quarter=True)

    assert (report.fiscal_year, report.fiscal_quarter) == (2026, 1)
    revenue = metric(report, "revenue")
    assert revenue.value == 44_062e6
    assert revenue.sources[0].kind == "filing" and revenue.sources[0].form == "10-Q"


def test_8k_two_quarters_ahead_is_not_guessed(monkeypatch):
    """If the release isn't exactly the next quarter we can't name it, so keep the 10-Q."""
    page = press_release(quarter_end="October 26, 2025")
    use_earnings_8k(monkeypatch, filed="2025-11-19", page=page)

    report = get_financials("NVDA", latest_quarter=True)

    assert (report.fiscal_year, report.fiscal_quarter) == (2026, 1)
    assert any("not used" in note for note in report.notes)


def test_unreadable_release_keeps_10q(monkeypatch):
    use_earnings_8k(monkeypatch, filed="2025-08-27", page="<p>Nothing useful here.</p>")

    report = get_financials("NVDA", latest_quarter=True)

    assert (report.fiscal_year, report.fiscal_quarter) == (2026, 1)


@pytest.mark.parametrize("kwargs", [{"fiscal_year": 2025, "quarter": 3}, {}])
def test_release_only_checked_for_latest_quarter(monkeypatch, kwargs):
    use_earnings_8k(monkeypatch, filed="2025-08-27", page=press_release())

    report = get_financials("NVDA", **kwargs)

    assert all(s.kind == "filing" for m in report.metrics for s in m.sources)
