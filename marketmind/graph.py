"""Assemble the MarketMind workflow graph with SQLite thread memory.

Topology (deterministic linear pipeline; each flow node is gated on the routes
chosen by the supervisor, so unselected flows pass through untouched):

    START
      -> supervisor
      -> stock_fetcher -> risk_calculator      (Stock Analysis Flow)
      -> news_agent                            (News Analysis Flow)
      -> purchase_agent                        (Purchase Flow, human-in-the-loop)
      -> processing -> state_updater -> response_synthesizer
      -> approval_gate --(approve)--> END
                       --(revise)---> response_synthesizer
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import agents, config
from .state import AgentState


def build_graph(checkpointer=None):
    """Construct and compile the workflow graph."""
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("supervisor", agents.supervisor)
    g.add_node("stock_fetcher", agents.stock_fetcher)
    g.add_node("risk_calculator", agents.risk_calculator)
    g.add_node("news_agent", agents.news_agent)
    g.add_node("purchase_agent", agents.purchase_agent)
    g.add_node("processing", agents.processing)
    g.add_node("state_updater", agents.state_updater)
    g.add_node("response_synthesizer", agents.response_synthesizer)
    g.add_node("approval_gate", agents.approval_gate)

    # Edges (deterministic linear pipeline with route-gated flow nodes).
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "stock_fetcher")
    g.add_edge("stock_fetcher", "risk_calculator")
    g.add_edge("risk_calculator", "news_agent")
    g.add_edge("news_agent", "purchase_agent")
    g.add_edge("purchase_agent", "processing")
    g.add_edge("processing", "state_updater")
    g.add_edge("state_updater", "response_synthesizer")
    g.add_edge("response_synthesizer", "approval_gate")

    # Human Approval Gate: approve -> END, revise -> regenerate.
    g.add_conditional_edges(
        "approval_gate",
        agents.approval_decision,
        {"approve": END, "revise": "response_synthesizer"},
    )

    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_app():
    """Return the compiled graph backed by a persistent SQLite checkpointer."""
    conn = sqlite3.connect(config.MARKETMIND_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    return build_graph(checkpointer=checkpointer)


def list_threads() -> list[str]:
    """Return all known conversation thread ids from the checkpointer."""
    app = get_app()
    checkpointer = app.checkpointer
    threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            tid = checkpoint.config.get("configurable", {}).get("thread_id")
            if tid:
                threads.add(tid)
    except Exception:
        pass
    return sorted(threads)
