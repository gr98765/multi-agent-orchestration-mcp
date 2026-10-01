"""Checks for prices.py, using a fake price table (no internet)."""

import pandas as pd
import pytest

from analystcrew.data import prices


def fake_table(*args):
    days = pd.to_datetime(["2026-04-27", "2026-04-28", "2026-07-24"])
    return pd.DataFrame(
        {"Close": [100.0, 105.0, 120.0], "High": [
            101.0, 108.0, 125.0], "Low": [95.0, 99.0, 118.0]},
        index=days,
    )


def test_price_change(monkeypatch):
    monkeypatch.setattr(prices, "_download_history", fake_table)
    p = prices.get_price_performance("nvda", "2026-04-27", "2026-07-26")
    assert (p.ticker, p.start_close, p.end_close,
            p.pct_change) == ("NVDA", 100.0, 120.0, 20.0)
    assert (p.period_high, p.period_low) == (125.0, 95.0)
    assert p.end_date == "2026-07-24"  # the last trading day, not the weekend


def test_bad_dates_give_clear_messages():
    with pytest.raises(prices.PriceError, match="year-month-day"):
        prices.get_price_performance("NVDA", "July 1", "2026-07-26")
    with pytest.raises(prices.PriceError, match="after"):
        prices.get_price_performance("NVDA", "2026-07-26", "2026-01-01")


def test_no_data(monkeypatch):
    monkeypatch.setattr(prices, "_download_history", lambda *a: pd.DataFrame())
    with pytest.raises(prices.PriceError, match="No price data"):
        prices.get_price_performance("ZZZZ", "2026-01-01", "2026-02-01")
