"""The shared shapes that agents pass between each other.

Agents never write URLs themselves. Every fact gets an ID (E1, E2, ...), and the
Writer may only cite those IDs. The links come from the data, so they can't be made up.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EvidenceKind = Literal["financial", "filing", "news", "price"]


class Evidence(BaseModel):
    id: str  # "E1", "E2", ... how agents refer to a fact
    kind: EvidenceKind
    summary: str  # short description of the fact
    value: str | None = None  # the figure as text, e.g. "$81.6B"
    number: float | None = None  # the raw number, used for calculations
    url: str | None = None  # link to the official source
    untrusted: bool = False  # True for news text: data only, never instructions
    published_date: str | None = None  # when a news article came out, as the source gave it


class Claim(BaseModel):
    text: str  # one sentence of the report
    evidence_ids: list[str] = Field(
        default_factory=list)  # the facts behind it


class ResearchNotes(BaseModel):
    company: str
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str = ""  # the Researcher's short wrap-up

    def add(
        self,
        kind: EvidenceKind,
        summary: str,
        value: str | None = None,
        url: str | None = None,
        untrusted: bool = False,
        number: float | None = None,
        published_date: str | None = None,
    ) -> Evidence:
        """Record one fact and return it with its new ID."""
        item = Evidence(
            id=f"E{len(self.evidence) + 1}",
            kind=kind,
            summary=summary,
            value=value,
            number=number,
            url=url,
            untrusted=untrusted,
            published_date=published_date,
        )
        self.evidence.append(item)
        return item


class TraceStep(BaseModel):
    """One step of a run, for the trace: what ran, how long it took, and what went wrong."""

    step: str  # "research", "analyze", "write" or "review"
    seconds: float
    round: int  # 0 before writing starts; 1 = first draft, 2 and 3 = rewrites
    issues: list[str] = Field(default_factory=list)  # review issues found in this step


class Report(BaseModel):
    """The Writer's report: sentences that each cite evidence, plus what it threw out.

    After a full run it also holds the evidence, the final review result, and the trace."""

    company: str
    sentences: list[Claim] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)  # sentences dropped, with the reason
    evidence: list[Evidence] = Field(default_factory=list)  # the facts the IDs point to
    passed: bool | None = None  # did the last review pass? None = not reviewed yet
    issues: list[str] = Field(default_factory=list)  # problems from the last review
    trace: list[TraceStep] = Field(default_factory=list)
