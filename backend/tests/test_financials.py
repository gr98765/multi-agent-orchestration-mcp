import pytest

from analystcrew.data.financials import available_years, build_report, fiscal_calendar

B = 1_000_000


def m(report, name):
    return next(x for x in report.metrics if x.metric == name)


def test_available_years_and_latest_default(facts):
    assert available_years(facts) == [2020, 2025, 2026]
    # latest COMPLETE year for annual reports
    assert build_report(facts).fiscal_year == 2025


def test_calendar_ignores_prior_year_comparatives_tagged_with_current_fy(facts):
    fy = fiscal_calendar(facts, 2025)
    assert (fy.start, fy.end) == ("2024-01-29", "2025-01-26")
    assert fy.quarter_ends == ["2024-04-28",
                               "2024-07-28", "2024-10-27", "2025-01-26"]


def test_full_year_values_are_reported_with_sources(facts):
    r = build_report(facts, 2025)
    rev = m(r, "revenue")
    assert rev.status == "reported" and rev.value == 130497 * B
    assert rev.sources[0].form == "10-K"
    assert rev.sources[0].url.startswith(
        "https://www.sec.gov/Archives/edgar/data/1045810/")


def test_q4_revenue_is_derived_from_annual_minus_nine_months(facts):
    rev = m(build_report(facts, 2025, 4), "revenue")
    assert rev.status == "derived"
    # $39,331M, Nvidia's actual Q4 FY25 revenue
    assert rev.value == (130497 - 91166) * B
    assert {s.form for s in rev.sources} == {"10-K", "10-Q"}
    assert rev.period_start == "2024-10-28" and rev.period_end == "2025-01-26"


def test_q4_net_income_derived_from_quarterly_figures_when_no_ytd(facts):
    ni = m(build_report(facts, 2025, 4), "net_income")
    assert ni.status == "derived"
    assert ni.value == (72880 - 14881 - 16599 - 19309) * B


def test_directly_reported_quarter(facts):
    rev = m(build_report(facts, 2025, 2), "revenue")
    assert rev.status == "reported" and rev.value == 30040 * B


def test_cash_flow_quarter_from_ytd_difference(facts):
    ocf = m(build_report(facts, 2025, 2), "operating_cash_flow")
    assert ocf.status == "derived" and ocf.value == (29800 - 15345) * B


def test_eps_is_never_derived(facts):
    eps = m(build_report(facts, 2025, 4), "eps_diluted")
    assert eps.status == "not_found" and eps.value is None
    assert "share counts" in eps.note
    assert m(build_report(facts, 2025, 3), "eps_diluted").value == 0.78


def test_missing_metric_is_explicit_not_found(facts):
    gp = m(build_report(facts, 2025), "gross_profit")
    assert gp.status == "not_found" and gp.value is None


def test_balance_sheet_instant_value(facts):
    cash = m(build_report(facts, 2025, 4), "cash_and_equivalents")
    assert cash.status == "reported" and cash.value == 8589 * B


def test_falls_back_to_older_concept_name(facts):
    rev = m(build_report(facts, 2020), "revenue")
    assert rev.concept == "Revenues" and rev.value == 10918 * B


def test_quarter_unavailable_when_quarters_cannot_be_mapped(facts):
    rev = m(build_report(facts, 2020, 2), "revenue")
    assert rev.status == "not_found" and "4 quarters" in rev.note


def test_invalid_inputs(facts):
    with pytest.raises(ValueError, match="fiscal_quarter"):
        build_report(facts, 2025, 5)
    with pytest.raises(ValueError, match="Available"):
        build_report(facts, 2011)


# ---------- in-progress fiscal year (only 10-Qs filed so far) ----------


def test_in_progress_year_calendar(facts):
    fy = fiscal_calendar(facts, 2026)
    assert not fy.complete and fy.start == "2025-01-27" and fy.quarter_ends == [
        "2025-04-27"]


def test_in_progress_quarter_is_reported(facts):
    r = build_report(facts, 2026, 1)
    rev = m(r, "revenue")
    assert rev.status == "reported" and rev.value == 44062 * B
    assert (rev.period_start, rev.period_end) == ("2025-01-27", "2025-04-27")
    assert "filed so far: Q1-Q1" in r.notes[0]


def test_unfiled_quarter_says_so(facts):
    rev = m(build_report(facts, 2026, 2), "revenue")
    assert rev.status == "not_found" and "not been filed yet" in rev.note


def test_full_year_of_in_progress_year_explains(facts):
    with pytest.raises(ValueError, match="still in progress"):
        build_report(facts, 2026)


def test_latest_quarter_finds_newest_filing(facts):
    r = build_report(facts, latest_quarter=True)
    assert (r.fiscal_year, r.fiscal_quarter) == (2026, 1)


def test_quarter_without_year_uses_latest_year_where_it_exists(facts):
    assert build_report(facts, quarter=1).fiscal_year == 2026
    assert build_report(facts, quarter=4).fiscal_year == 2025


# ---------- get_financials: the full chain (find company -> download -> calculate) ----------


def test_get_financials_by_name():
    from analystcrew.data.financials import get_financials

    r = get_financials("nvidia", 2025, 4)
    assert (r.ticker, r.company) == ("NVDA", "NVIDIA CORP")
    assert m(r, "revenue").value == 39_331_000_000


def test_get_financials_unknown_company():
    from analystcrew.data.financials import get_financials

    with pytest.raises(ValueError, match="Not Found"):
        get_financials("zzzz imaginary")
