"""The graph: connects the agents' steps with LangGraph and runs a whole report.

    research -> analyze -> write -> review --passed, or out of rewrites--> finish
                             ^          |
                             +--failed--+   (at most 2 rewrites, see coordinator.py)

Use it:  report = run_report("NVDA")
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..audit import new_run_id
from ..models import Report
from .coordinator import Agents, State, after_review, make_steps


def build_graph(agents: Agents | None = None):
    """Build the graph. Pass fake agents in tests; leave it empty for the real ones."""
    graph = StateGraph(State)
    for name, step in make_steps(agents or Agents()).items():
        graph.add_node(name, step)
    graph.add_edge(START, "research")
    graph.add_edge("research", "analyze")
    graph.add_edge("analyze", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", after_review, {"write": "write", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def run_report(company: str, agents: Agents | None = None) -> Report:
    """Run every step for one company and return the finished report, with its trace."""
    run_id = new_run_id()
    # run_name and metadata label the whole run in LangSmith (ignored when tracing is off).
    config = {"run_name": "analystcrew", "metadata": {"ticker": company, "run_id": run_id}}
    start = {"company": company, "run_id": run_id, "round": 0, "trace": []}
    final = build_graph(agents).invoke(start, config=config)
    return final["report"]
