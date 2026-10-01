"""See Batch 1 working with real SEC data.

python backend/scripts/check_data.py NVDA                  latest quarter for Nvidia
python backend/scripts/check_data.py AAPL --year 2025 --quarter 4
python backend/scripts/check_data.py --search apple        which companies match "apple"?
"""

import argparse

from analystcrew.data.financials import get_financials
from analystcrew.data.sec import SecError, find_company, list_filings, search_companies


def money(value: float | None, unit: str | None) -> str:
    if value is None:
        return "Not Found"
    if unit == "USD/shares":
        return f"${value:.2f}"
    return f"${value / 1e9:,.2f}B" if abs(value) >= 1e9 else f"${value / 1e6:,.1f}M"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("company", nargs="?", default="NVDA")
    p.add_argument("--year", type=int)
    p.add_argument("--quarter", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--search", help="list companies matching these words")
    args = p.parse_args()

    if args.search:
        for c in search_companies(args.search, limit=15):
            print(f"  {c.ticker:<8} {c.name}")
        return

    latest = args.year is None and args.quarter is None
    report = get_financials(args.company, args.year,
                            args.quarter, latest_quarter=latest)
    period = f"Q{report.fiscal_quarter} " if report.fiscal_quarter else ""
    print(
        f"\n{report.company} ({report.ticker}), {period}FY{report.fiscal_year} "
        f"({report.period_start} to {report.period_end})\n"
    )
    for m in report.metrics:
        print(f"  {m.metric:<22} {money(m.value, m.unit):>12}   {m.status}")
    print()
    for note in report.notes:
        print(f"  note: {note}")

    company = find_company(args.company)
    print("\nLatest filings:")
    for f in list_filings(company.cik, limit=3):
        print(f"  {f.form:<5} filed {f.filing_date}  {f.url}")


if __name__ == "__main__":
    try:
        main()
    except (SecError, ValueError) as e:
        print(f"\nProblem: {e}")
