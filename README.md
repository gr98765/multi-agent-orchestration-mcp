# Multi agent Orchestration for market research report

**AI agents that research public companies and write reports where every number is verified against official filings.**

## The problem

Researching a company means reading long filings, pulling numbers, checking news, and
writing it up: hours per company. AI chatbots are faster, but they can quote outdated
numbers or make them up, with no source to check.

## Who it's for

| User | Use case |
|---|---|
| **Investment analysts** | Quarterly earnings briefs on many companies, fast |
| **Sales and business teams** | Understand a client's or partner's business before a meeting |
| **Founders** | Track competitors' revenue, growth, and news |
| **Job seekers** | Research a company before an interview |

**Example:** an analyst types *"Nvidia, latest quarter."* AnalystCrew pulls the official
numbers from the SEC, adds recent news and the stock's move, and produces a cited report
in minutes, where every number links to the filing it came from.

## How it works

```
You → Dashboard → API → Agent team → MCP server → SEC · Tavily · Yahoo Finance
                        (planned)    (5 tools)
```

1. **You ask** for a company and period, e.g. "Nvidia, latest quarter."
2. **The Coordinator** (planned) plans the work and hands tasks to the other agents in order.
3. **The Researcher** calls the MCP server's tools to collect facts: official financials,
   filings, news, and the stock's move. Each fact is saved with its source.
4. **The Analyst** calculates growth and margins. Math is done in code, never by the AI.
5. **The Writer** drafts the report. Every sentence must cite the facts it uses.
6. **The Reviewer** checks every number in the draft against its source. Mistakes are sent
   back to be fixed. Only a report that passes is shown to you.

## MCP server tools

The MCP server is the single, safe door to the data. Our agents use it, and so can any
MCP-compatible AI such as Claude Desktop or Cursor. All tools are read-only.

| Tool | You give it | It returns | Source |
|---|---|---|---|
| `resolve_company` | Ticker or name | Official name and SEC ID | SEC |
| `get_financials` | Company, period | Revenue, profit, EPS, cash flow, cash, each labeled and linked to its filing | SEC |
| `list_filings` | Company | Latest 10-K, 10-Q, 8-K reports with links | SEC |
| `search_news` | Search words | Recent articles, flagged as untrusted text | Tavily |
| `get_price_performance` | Ticker, two dates | Start and end price, % change, high, low | Yahoo Finance |

## Approach

- **Official data only.** Financials come live from SEC filings, never from the AI's memory.
- **Every number is labeled:** `reported` (from a filing), `derived` (calculated, with the
  method shown), or `Not Found`. Nothing is ever estimated.
- **AI writes, code checks.** AI handles reading and writing; plain code does all math and
  checks every number in the report against its source.

## Example

```
$ python backend/scripts/check_data.py NVDA

NVIDIA CORP (NVDA), Q2 FY2027 (2026-04-27 to 2026-07-26)

  revenue                     $96.22B   reported
  net_income                  $59.69B   reported
  eps_diluted                   $2.46   reported
  operating_cash_flow         $24.08B   derived
```

These match Nvidia's own announcement for the quarter ($96.2B revenue, $2.46 EPS).
Operating cash flow is *derived*: companies report it year-to-date, so the quarter is
calculated as six months minus three.

## Interesting problems solved

- **No Q4 report exists.** Companies only file Q1–Q3 quarterly; Q4 is calculated from
  the annual report (full year minus nine months), citing both filings.
- **Every company has its own calendar.** Nvidia's fiscal 2027 runs Jan 2026 – Jan 2027.
  Periods are worked out from each company's filings, including the year in progress.
- **Prompt injection.** News text is cut short and flagged as untrusted, so instructions
  hidden in an article can't steer the agents.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e "backend[dev]"
copy .env.example .env        # add your name/email for the SEC and a free Tavily key
python backend/scripts/check_data.py NVDA
python backend/scripts/check_tools.py NVDA     # all 5 MCP tools
cd backend && pytest -q
```

## Status

- [x] Data layer: SEC financials (with Q4 derivation), filings, news, prices
- [x] MCP server with 5 read-only tools
- [ ] Web dashboard (React + TypeScript)
- [ ] AI agent team with a fact-checking reviewer (LangGraph)
- [ ] Accuracy evaluation across many companies

**Tech:** Python · MCP · FastAPI · LangGraph · React · TypeScript · SEC EDGAR · Tavily · yfinance

*Research tool only, not investment advice.*