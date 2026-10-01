"""The messenger to the SEC (the US government's company-filings website).

Three jobs:
  1. find_company / search_companies: "NVDA" or "nvidia" -> Nvidia's official name and ID
  2. get_company_facts: download every financial number a company has reported
  3. list_filings: its recent official reports, with links

The SEC's data is free, but it requires a "User-Agent": your name and email, sent with
every request. We also save each download to disk (a cache), so running the same thing
twice is instant and doesn't put extra load on the SEC.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # reads SEC_USER_AGENT from your .env file

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FILINGS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

CACHE_DIR = Path(os.getenv("ANALYSTCREW_CACHE", ".cache"))
CACHE_HOURS = 24
MAIN_FORMS = ("10-K", "10-Q", "8-K")  # annual, quarterly, and event reports


class SecError(Exception):
    """Something went wrong talking to the SEC. The message explains what to do."""


# ---------- the "forms" this file hands back ----------


class Company(BaseModel):
    ticker: str  # e.g. "NVDA"
    cik: int  # the SEC's ID number for the company, e.g. 1045810
    name: str  # official name, e.g. "NVIDIA CORP"


class Filing(BaseModel):
    # "10-K" (annual), "10-Q" (quarterly), "8-K" (event / press release)
    form: str
    filing_date: str
    report_date: str | None  # the period the report covers
    accession: str  # the SEC's ID for this filing
    url: str  # direct link to the document


# ---------- downloading (with cache) ----------

_client: httpx.Client | None = None  # tests replace this with a fake SEC


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30)
    return _client


def _download_json(url: str) -> dict | None:
    """Fetch a JSON file from the SEC, using the disk cache when possible.
    Returns None if the SEC says the file doesn't exist (HTTP 404)."""
    cache_file = CACHE_DIR / \
        (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_HOURS * 3600:
        return json.loads(cache_file.read_text())

    user_agent = (os.getenv("SEC_USER_AGENT") or "").strip()
    if "@" not in user_agent:
        raise SecError(
            "SEC_USER_AGENT is missing from your .env file. The SEC requires your name and "
            'email, e.g. SEC_USER_AGENT="Jane Doe jane@example.com"'
        )

    try:
        response = _http().get(url, headers={"User-Agent": user_agent})
    except httpx.HTTPError as e:
        raise SecError(f"Could not reach the SEC website: {e}") from e
    # the SEC allows ~10 requests per second; stay well under it
    time.sleep(0.12)

    if response.status_code == 404:
        return None
    if response.status_code == 403:
        raise SecError(
            "The SEC refused the request (HTTP 403). Check that SEC_USER_AGENT "
            "in .env has your real name and email."
        )
    if response.status_code != 200:
        raise SecError(
            f"The SEC returned HTTP {response.status_code}. Try again later.")

    data = response.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


# ---------- job 1: find companies ----------


def _all_companies() -> list[Company]:
    """Every company the SEC lists (about 10,000), from one file it publishes."""
    data = _download_json(TICKERS_URL) or {}
    return [
        Company(ticker=r["ticker"], cik=int(r["cik_str"]), name=r["title"]) for r in data.values()
    ]


def search_companies(query: str, limit: int = 10) -> list[Company]:
    """All companies whose ticker or name contains the search words."""
    q = query.strip().lower()
    companies = _all_companies()
    exact = [c for c in companies if c.ticker.lower() == q]
    partial = [
        c for c in companies if c not in exact and (q in c.name.lower() or q in c.ticker.lower())
    ]
    # shorter names first: usually the main company
    partial.sort(key=lambda c: len(c.name))
    return (exact + partial)[:limit]


def find_company(query: str) -> Company | None:
    """The single best match for a ticker ("NVDA") or name ("nvidia"), or None."""
    matches = search_companies(query, limit=1)
    return matches[0] if matches else None


# ---------- job 2: download all financial numbers ----------


def get_company_facts(cik: int) -> dict | None:
    """Every number the company has reported in its filings (a big file, often 5+ MB).
    financials.py turns this into clean numbers for a specific period."""
    return _download_json(FACTS_URL.format(cik=cik))


# ---------- job 3: list filings ----------


def filing_folder_url(cik: int, accession: str) -> str:
    """Link to the folder holding one filing's documents."""
    return f"{ARCHIVES_URL}/{cik}/{accession.replace('-', '')}/"


def list_filings(cik: int, forms: tuple[str, ...] = MAIN_FORMS, limit: int = 10) -> list[Filing]:
    """The company's most recent filings of the given types, newest first."""
    data = _download_json(FILINGS_URL.format(cik=cik))
    if not data:
        return []
    recent = data["filings"]["recent"]  # the SEC stores this as parallel lists
    filings: list[Filing] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        accession = recent["accessionNumber"][i]
        filings.append(
            Filing(
                form=form,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i] or None,
                accession=accession,
                url=filing_folder_url(cik, accession) +
                recent["primaryDocument"][i],
            )
        )
        if len(filings) >= limit:
            break
    return filings
