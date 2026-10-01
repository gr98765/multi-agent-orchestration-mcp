"""Checks for the MCP server: call each tool the way an AI agent would."""

import pytest
from mcp import Client
from test_news import fake_tavily  # noqa: F401  (reuse the fake Tavily)
from test_prices import fake_table

from analystcrew.data import prices
from analystcrew.mcp_server import mcp

TOOLS = {
    "resolve_company",
    "get_financials",
    "list_filings",
    "search_news",
    "get_price_performance",
}


async def test_offers_five_read_only_tools():
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    assert all(t.annotations.read_only_hint for t in tools)


async def test_get_financials_tool():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_financials", {"company": "NVDA",
                               "fiscal_year": 2025, "fiscal_quarter": 4}
        )
    revenue = result.structured_content["metrics"][0]
    assert (revenue["value"], revenue["status"]) == (39_331_000_000, "derived")


async def test_unknown_company_is_a_readable_error():
    async with Client(mcp) as client:
        result = await client.call_tool("resolve_company", {"query": "zzzz imaginary"})
    assert result.is_error and "Not Found" in result.content[0].text


async def test_list_filings_tool():
    async with Client(mcp) as client:
        result = await client.call_tool("list_filings", {"company": "nvidia"})
    assert [f["form"] for f in result.structured_content["result"]] == [
        "8-K", "10-K", "10-Q"]


@pytest.mark.usefixtures("fake_tavily")
async def test_news_tool_flags_untrusted_text():
    async with Client(mcp) as client:
        result = await client.call_tool("search_news", {"query": "Nvidia"})
    assert "Untrusted" in result.structured_content["warning"]


async def test_news_without_key_explains(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    async with Client(mcp) as client:
        result = await client.call_tool("search_news", {"query": "Nvidia"})
    assert result.is_error and "TAVILY_API_KEY" in result.content[0].text


async def test_price_tool(monkeypatch):
    monkeypatch.setattr(prices, "_download_history", fake_table)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_price_performance",
            {"ticker": "NVDA", "start_date": "2026-04-27", "end_date": "2026-07-26"},
        )
    assert result.structured_content["pct_change"] == 20.0
