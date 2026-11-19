"""Tests for the graph and the Coordinator's retry loop, using fake agents (no model, no web).

The fake Reviewer fails a set number of times, so we can check that issues go back to
the Writer, that rewrites stop after 2, and that the trace records every step."""

from analystcrew.agents.coordinator import MAX_RETRIES, Agents
from analystcrew.agents.graph import run_report
from analystcrew.agents.reviewer import Issue, Review
from analystcrew.models import Claim, Report, ResearchNotes


class FakeAgents:
    """Fake agents that remember how they were called. The review fails `fails` times."""

    def __init__(self, fails: int):
        self.fails = fails
        self.feedback_seen: list[list[str]] = []  # what the Writer was told, per draft
        self.reviews = 0

    def research(self, company: str) -> ResearchNotes:
        notes = ResearchNotes(company=company)
        notes.add("financial", "revenue (reported)", value="$35.08B", number=35_082_000_000)
        return notes

    def analyze(self, notes: ResearchNotes) -> list[Claim]:
        return [Claim(text="Revenue was $35.08B.", evidence_ids=["E1"])]

    def write(self, notes, claims, feedback) -> Report:
        self.feedback_seen.append(list(feedback))
        draft = len(self.feedback_seen)
        return Report(
            company=notes.company,
            sentences=[Claim(text=f"Draft {draft}: revenue was $35.08B.", evidence_ids=["E1"])],
            rejected=["no evidence IDs: Nvidia rules."] if draft == 1 else [],
        )

    def review(self, sentences, notes) -> Review:
        self.reviews += 1
        if self.reviews <= self.fails:
            issue = Issue(sentence=0, check="unsupported_number",
                          message=f"bad number in review {self.reviews}")
            return Review(passed=False, issues=[issue])
        return Review(passed=True)

    def agents(self) -> Agents:
        return Agents(research=self.research, analyze=self.analyze,
                      write=self.write, review=self.review)


def steps(report: Report) -> list[tuple[str, int]]:
    return [(t.step, t.round) for t in report.trace]


def test_passing_first_draft_runs_each_step_once():
    fake = FakeAgents(fails=0)

    report = run_report("NVDA", fake.agents())

    assert report.passed is True
    assert report.issues == []
    assert report.sentences[0].text.startswith("Draft 1")
    assert fake.feedback_seen == [[]]
    assert steps(report) == [("research", 0), ("analyze", 0), ("write", 1), ("review", 1)]
    assert [ev.id for ev in report.evidence] == ["E1"]  # evidence is stored on the report
    assert all(t.seconds >= 0 for t in report.trace)


def test_failed_review_sends_issues_back_to_the_writer():
    fake = FakeAgents(fails=1)

    report = run_report("NVDA", fake.agents())

    assert report.passed is True
    assert report.sentences[0].text.startswith("Draft 2")  # the rewrite is what we keep
    assert fake.feedback_seen == [
        [],
        ["bad number in review 1",
         "This sentence was dropped (no evidence IDs: Nvidia rules.)"],
    ]
    assert steps(report) == [
        ("research", 0), ("analyze", 0),
        ("write", 1), ("review", 1),
        ("write", 2), ("review", 2),
    ]
    review_issues = [t.issues for t in report.trace if t.step == "review"]
    assert review_issues == [["bad number in review 1"], []]


def test_stops_after_two_rewrites_even_if_still_failing():
    fake = FakeAgents(fails=99)

    report = run_report("NVDA", fake.agents())

    assert MAX_RETRIES == 2
    assert len(fake.feedback_seen) == 3  # first draft + 2 rewrites, then stop
    assert fake.reviews == 3
    assert report.passed is False
    assert report.issues == ["bad number in review 3"]  # the last review's issues
    assert report.sentences[0].text.startswith("Draft 3")
    assert steps(report)[-2:] == [("write", 3), ("review", 3)]
    assert max(t.round for t in report.trace) == 1 + MAX_RETRIES


def test_passing_on_the_last_allowed_round():
    fake = FakeAgents(fails=2)

    report = run_report("NVDA", fake.agents())

    assert report.passed is True
    assert len(fake.feedback_seen) == 3
    assert report.issues == []
