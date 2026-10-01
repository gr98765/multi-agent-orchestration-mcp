"""News search through Tavily (free: 1,000 credits a month, no credit card).

Each search returns articles with a title, link, date, and a short snippet. Snippets are
cut short and treated as untrusted: news text could contain hidden instructions aimed at
an AI, so the agents must treat it only as data, never as commands.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from . import sec  # reuses the same cache folder as the SEC downloads

TAVILY_URL = "https://api.tavily.com/search"
SNIPPET_CHARS = 600
CACHE_HOURS = 6  # news changes faster than filings, so the cache expires sooner


class NewsError(Exception):
    """Something went wrong with the news search. The message explains what to do."""


class NewsItem(BaseModel):
    title: str
    url: str
    source: str  # the website, e.g. "reuters.com"
    published_date: str | None = None
    snippet: str  # a short excerpt (untrusted text)


_client: httpx.Client | None = None  # tests replace this with a fake Tavily


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30)
    return _client


def search_news(query: str, days: int = 90, max_results: int = 5) -> list[NewsItem]:
    """Recent news articles matching the query, e.g. "Nvidia data center revenue"."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        raise NewsError(
            "TAVILY_API_KEY is missing from your .env file. Get a free key at https://tavily.com"
        )

    request = {
        "query": query,
        "topic": "news",
        "days": max(1, min(days, 365)),
        "max_results": max(1, min(max_results, 10)),
        "search_depth": "basic",  # costs 1 credit per search
    }
    key = hashlib.sha256(json.dumps(
        request, sort_keys=True).encode()).hexdigest()
    cache_file = sec.CACHE_DIR / f"news-{key}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_HOURS * 3600:
        return [NewsItem(**item) for item in json.loads(cache_file.read_text())]

    try:
        response = _http().post(
            TAVILY_URL, json=request, headers={
                "Authorization": f"Bearer {api_key}"}
        )
    except httpx.HTTPError as e:
        raise NewsError(f"Could not reach Tavily: {e}") from e
    if response.status_code in (401, 403):
        raise NewsError(
            "Tavily rejected the API key. Check TAVILY_API_KEY in your .env file.")
    if response.status_code in (429, 432):
        raise NewsError(
            "Tavily's limit was reached (too many searches, or free credits used up).")
    if response.status_code != 200:
        raise NewsError(
            f"Tavily returned HTTP {response.status_code}. Try again later.")

    items = [
        NewsItem(
            title=r.get("title") or "(no title)",
            url=r["url"],
            source=urlparse(r["url"]).netloc.removeprefix("www."),
            published_date=r.get("published_date"),
            snippet=(r.get("content") or "")[:SNIPPET_CHARS],
        )
        for r in response.json().get("results", [])
    ]
    sec.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps([i.model_dump() for i in items]))
    return items
