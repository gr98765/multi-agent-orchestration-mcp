"""Write a full research report for one company, using all the agents.

Run from backend/:  python scripts/run_report.py NVDA

It creates two files in backend/reports/:
  <TICKER>.md    the report to read, with an evidence table and the run's trace
  <TICKER>.json  everything (sentences, evidence, review result, trace) for programs
"""

import sys

from analystcrew import audit
from analystcrew.agents.graph import run_report
from analystcrew.data.sec import find_company
from analystcrew.models import Report


def _cell(text: str | None) -> str:
    """Make text safe for one cell of a Markdown table."""
    return (text or "").replace("|", "/").replace("\r", " ").replace("\n", " ").strip()


def render_markdown(report: Report, ticker: str, name: str) -> str:
    drafts = max((t.round for t in report.trace), default=0)
    status = "Passed review" if report.passed else "FAILED review"
    lines = [
        f"# {name} ({ticker}): research report",
        "",
        f"**{status}** after {drafts} draft(s). "
        "Every sentence cites evidence IDs; the sources are listed below.",
        "",
        "## Report",
        "",
    ]
    for s in report.sentences:
        ids = f" [{', '.join(s.evidence_ids)}]" if s.evidence_ids else ""
        lines.append(f"- {s.text}{ids}")

    if report.issues:
        lines += ["", "## Review issues (not fixed)", ""]
        lines += [f"- {_cell(issue)}" for issue in report.issues]

    lines += ["", "## Evidence", ""]
    lines += ["| ID | Kind | Fact | Value | Source |", "|---|---|---|---|---|"]
    for ev in report.evidence:
        if ev.untrusted:  # news: show the headline and date only, never the article text
            kind = "news (untrusted)"
            value = ev.published_date or "undated"
        else:
            kind, value = ev.kind, ev.value
        source = f"[link]({ev.url})" if ev.url else ""
        lines.append(f"| {ev.id} | {kind} | {_cell(ev.summary)} | {_cell(value)} | {source} |")

    lines += ["", "## Trace", "", "| Step | Round | Seconds | Issues |", "|---|---|---|---|"]
    for t in report.trace:
        issues = "<br>".join(_cell(i) for i in t.issues)
        lines.append(f"| {t.step} | {t.round} | {t.seconds:.2f} | {issues} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    company = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    found = find_company(company)
    if found is None:
        sys.exit(f"Not Found: no SEC-registered company matches '{company}'.")

    report = run_report(found.ticker)

    audit.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = audit.REPORTS_DIR / f"{found.ticker}.md"
    json_path = audit.REPORTS_DIR / f"{found.ticker}.json"
    md_path.write_text(render_markdown(report, found.ticker, found.name), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"{'Passed' if report.passed else 'FAILED'} review. Wrote:")
    print(f"  {md_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
