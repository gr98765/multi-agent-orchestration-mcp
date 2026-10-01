"""Stock prices from Yahoo Finance, through the free yfinance library.

Answers: "How did this stock move between two dates?" Start price, end price, % change,
and the highest and lowest prices in between. yfinance is unofficial, so it can
occasionally break; errors come back as clear messages.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel


class PriceError(Exception):
    """Something went wrong getting prices. The message explains what to do."""


class PricePerformance(BaseModel):
    ticker: str
    start_date: str  # first trading day in the range
    end_date: str  # last trading day in the range
    start_close: float
    end_close: float
    pct_change: float
    period_high: float
    period_low: float
    source: str = "Yahoo Finance via yfinance (unofficial)"


def _download_history(ticker: str, start: str, end_exclusive: str):
    """Daily prices as a table. Tests replace this function with fake data."""
    import yfinance as yf

    return yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=True)


def get_price_performance(ticker: str, start_date: str, end_date: str) -> PricePerformance:
    """How the stock moved between two dates (YYYY-MM-DD)."""
    try:
        start, end = date.fromisoformat(
            start_date), date.fromisoformat(end_date)
    except ValueError as e:
        raise PriceError(
            "Dates must look like 2026-01-31 (year-month-day).") from e
    if end <= start:
        raise PriceError("end_date must be after start_date.")

    try:  # yfinance treats the end date as exclusive, so add a day
        table = _download_history(
            ticker.upper(), start_date, str(end + timedelta(days=1)))
    except Exception as e:  # noqa: BLE001 - yfinance can fail in many ways
        raise PriceError(
            f"Could not get prices from Yahoo Finance: {e}") from e
    if table is None or table.empty:
        raise PriceError(
            f"No price data for {ticker.upper()} between {start_date} and {end_date}.")

    closes = table["Close"]
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    return PricePerformance(
        ticker=ticker.upper(),
        start_date=str(closes.index[0].date()),
        end_date=str(closes.index[-1].date()),
        start_close=round(first, 2),
        end_close=round(last, 2),
        pct_change=round((last / first - 1) * 100, 2),
        period_high=round(float(table["High"].max()), 2),
        period_low=round(float(table["Low"].min()), 2),
    )
