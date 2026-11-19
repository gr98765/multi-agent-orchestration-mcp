"""Run the Researcher on one company and print every fact it found."""

import sys

from analystcrew.agents.researcher import run_research

company = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
notes = run_research(company)

print(f"\nEvidence for {company}:\n")
for ev in notes.evidence:
    value = f" = {ev.value}" if ev.value else ""
    print(f"{ev.id} [{ev.kind}] {ev.summary}{value}")
    if ev.url:
        print(f"     {ev.url}")

print(f"\nSummary: {notes.summary}")
