"""See Batch 2 working: start the MCP server and call all 5 tools, like an AI agent would.

python backend/scripts/check_tools.py NVDA
"""

import asyncio
import os
import sys

from mcp import Client, StdioServerParameters

COMPANY = sys.argv[1] if len(sys.argv) > 1 else "NVDA"


def show(title: str, result) -> dict | list | None:
    print(f"\n=== {title}")
    if result.is_error:
        print("  Problem:", result.content[0].text)
        return None
    data = result.structured_content
    return data.get("result", data)  # lists come wrapped as {"result": [...]}


async def main() -> None:
    # Start the MCP server as a separate program, exactly as Claude Desktop would.
    # MCP doesn't pass your environment along by default, so we hand over .env values.
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "analystcrew.mcp_server"], env=dict(os.environ)
    )
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print("Tools offered by the server:", ", ".join(t.name for t in tools))

        c = show(
            "1. resolve_company", await client.call_tool("resolve_company", {"query": COMPANY})
        )
        if c:
            print(f"  {c['name']} ({c['ticker']}), SEC ID {c['cik']}")

        r = show(
            "2. get_financials (latest quarter)",
            await client.call_tool("get_financials", {"company": COMPANY, "latest_quarter": True}),
        )
        if r:
            print(
                f"  Q{r['fiscal_quarter']} FY{r['fiscal_year']}: "
                f"{r['period_start']} to {r['period_end']}"
            )
            for m in r["metrics"][:3]:
                value = f"${m['value'] / 1e9:,.2f}B" if m["value"] else "Not Found"
                print(f"  {m['metric']:<18} {value:>12}  {m['status']}")

        filings = show(
            "3. list_filings",
            await client.call_tool("list_filings", {"company": COMPANY, "limit": 3}),
        )
        for f in filings or []:
            print(f"  {f['form']:<5} {f['filing_date']}  {f['url']}")

        news = show(
            "4. search_news",
            await client.call_tool(
                "search_news", {
                    "query": f"{COMPANY} earnings", "max_results": 2}
            ),
        )
        for item in (news or {}).get("items", []):
            print(f"  {item['published_date'] or ''}  {item['title']}")
            print(f"    {item['url']}")

        if r:  # how did the stock move during that same quarter?
            p = show(
                "5. get_price_performance (same quarter)",
                await client.call_tool(
                    "get_price_performance",
                    {
                        "ticker": COMPANY,
                        "start_date": r["period_start"],
                        "end_date": r["period_end"],
                    },
                ),
            )
            if p:
                print(
                    f"  ${p['start_close']} -> ${p['end_close']}  ({p['pct_change']:+.1f}%), "
                    f"high ${p['period_high']}, low ${p['period_low']}"
                )


asyncio.run(main())
