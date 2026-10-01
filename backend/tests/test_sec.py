"""Checks for sec.py, the messenger to the SEC."""

import pytest

from analystcrew.data import sec


def test_find_company_by_ticker_and_name():
    assert sec.find_company("nvda").cik == 1045810
    assert sec.find_company("advanced micro").ticker == "AMD"
    assert sec.find_company("no such company zzz") is None


def test_search_companies_lists_matches():
    assert [c.ticker for c in sec.search_companies("corp")] == ["NVDA"]


def test_sends_your_name_and_email(fake_sec):
    sec.find_company("NVDA")
    assert fake_sec[0].headers["User-Agent"] == "Test User test@example.com"


def test_missing_user_agent_gives_clear_message(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "")
    with pytest.raises(sec.SecError, match="SEC_USER_AGENT"):
        sec.find_company("NVDA")


def test_filings_show_main_reports_with_links():
    filings = sec.list_filings(1045810)
    assert [f.form for f in filings] == [
        "8-K", "10-K", "10-Q"]  # the Form 4 is skipped
    assert filings[1].url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )


def test_cache_means_one_download(fake_sec):
    sec.get_company_facts(1045810)
    sec.get_company_facts(1045810)
    assert len(fake_sec) == 1
