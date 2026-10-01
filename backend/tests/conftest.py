"""Shared test setup: a fake SEC that answers like the real one (Nvidia's real numbers).

Tests never touch the internet: we swap sec.py's web client for this fake."""

import httpx
import pytest

from analystcrew.data import sec

CIK = 1045810
K25 = ("10-K", "0001045810-25-000023", "2025-02-26", 2025, "FY")
Q1 = ("10-Q", "0001045810-24-000124", "2024-05-29", 2025, "Q1")
Q2 = ("10-Q", "0001045810-24-000264", "2024-08-28", 2025, "Q2")
Q3 = ("10-Q", "0001045810-24-000316", "2024-11-20", 2025, "Q3")
K20 = ("10-K", "0001045810-20-000010", "2020-02-20", 2020, "FY")
Q1_26 = ("10-Q", "0001045810-25-000116",
         "2025-05-28", 2026, "Q1")  # year in progress

FY_START, FY_END = "2024-01-29", "2025-01-26"
Q1_END, Q2_START, Q2_END, Q3_START, Q3_END = (
    "2024-04-28",
    "2024-04-29",
    "2024-07-28",
    "2024-07-29",
    "2024-10-27",
)


def f(start, end, val, filing):
    form, accn, filed, fy, fp = filing
    row = {"end": end, "val": val, "accn": accn,
           "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start:
        row["start"] = start
    return row


def concept(unit, rows):
    return {"units": {unit: rows}}


B = 1_000_000


def company_facts() -> dict:
    return {
        "cik": CIK,
        "entityName": "NVIDIA CORP",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": concept(
                    "USD",
                    [
                        f(FY_START, FY_END, 130497 * B, K25),
                        f(
                            "2023-01-30", "2024-01-28", 60922 * B, K25
                        ),  # comparative, tagged fy=2025!
                        f(FY_START, Q1_END, 26044 * B, Q1),
                        # prior-year comparative
                        f("2023-01-30", "2023-04-30", 7192 * B, Q1),
                        f(Q2_START, Q2_END, 30040 * B, Q2),
                        f(FY_START, Q2_END, 56084 * B, Q2),
                        f(Q3_START, Q3_END, 35082 * B, Q3),
                        f(FY_START, Q3_END, 91166 * B, Q3),
                        f("2025-01-27", "2025-04-27", 44062 * B, Q1_26),
                        # comparative, tagged fy=2026
                        f(FY_START, Q1_END, 26044 * B, Q1_26),
                    ],
                ),
                "Revenues": concept("USD", [f("2019-01-28", "2020-01-26", 10918 * B, K20)]),
                "NetIncomeLoss": concept(
                    "USD",
                    [
                        f(FY_START, FY_END, 72880 * B, K25),
                        f("2023-01-30", "2024-01-28", 29760 * B, K25),
                        f(FY_START, Q1_END, 14881 * B, Q1),
                        f(Q2_START, Q2_END, 16599 * B, Q2),
                        f(Q3_START, Q3_END, 19309 * B, Q3),
                        f("2025-01-27", "2025-04-27", 18775 * B, Q1_26),
                    ],
                ),
                "EarningsPerShareDiluted": concept(
                    "USD/shares",
                    [
                        f(FY_START, FY_END, 2.94, K25),
                        f(FY_START, Q1_END, 0.60, Q1),
                        f(Q2_START, Q2_END, 0.67, Q2),
                        f(Q3_START, Q3_END, 0.78, Q3),
                    ],
                ),
                "NetCashProvidedByUsedInOperatingActivities": concept(
                    "USD",
                    [
                        f(FY_START, FY_END, 64089 * B, K25),
                        f(FY_START, Q1_END, 15345 * B, Q1),
                        f(FY_START, Q2_END, 29800 * B, Q2),
                        f(FY_START, Q3_END, 46964 * B, Q3),
                    ],
                ),
                "CashAndCashEquivalentsAtCarryingValue": concept(
                    "USD",
                    [
                        f(None, FY_END, 8589 * B, K25),
                        f(None, Q3_END, 9107 * B, Q3),
                    ],
                ),
            }
        },
    }


TICKERS = {
    "0": {"cik_str": CIK, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
}

SUBMISSIONS = {
    "cik": str(CIK),
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001045810-25-000030", K25[1], "0001045810-25-000011", Q3[1]],
            "filingDate": ["2025-02-26", "2025-02-26", "2025-02-20", "2024-11-20"],
            "reportDate": ["2025-02-26", "2025-01-26", "", "2024-10-27"],
            "form": ["8-K", "10-K", "4", "10-Q"],
            "primaryDocument": [
                "nvda-8k.htm",
                "nvda-20250126.htm",
                "form4.xml",
                "nvda-20241027.htm",
            ],
        }
    },
}


def fake_sec_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("company_tickers.json"):
        return httpx.Response(200, json=TICKERS)
    if url.endswith(f"companyfacts/CIK{CIK:010d}.json"):
        return httpx.Response(200, json=company_facts())
    if url.endswith(f"submissions/CIK{CIK:010d}.json"):
        return httpx.Response(200, json=SUBMISSIONS)
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def fake_sec(monkeypatch, tmp_path):
    """Every test gets the fake SEC, an empty temporary cache, and a test User-Agent."""
    calls = []

    def handler(request):
        calls.append(request)
        return fake_sec_handler(request)

    monkeypatch.setattr(sec, "_client", httpx.Client(
        transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(sec, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sec.time, "sleep", lambda s: None)
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    return calls


@pytest.fixture
def facts():
    return company_facts()
