"""Assemble the MarketMind workflow graph with SQLite thread memory.

Topology — the Supervisor classifies the request, then **conditional edges**
dynamically route to only the workflows that are needed (a deterministic
skip-chain ``stocks -> news -> buy -> processing``). Unneeded flows are skipped
entirely instead of being run-then-gated::

    START
      -> supervisor
           --cond--> stock_fetcher -> risk_calculator   (Stock Analysis Flow)
           --cond--> news_agent                         (News Analysis Flow)
           --cond--> trader -> trade_tools (ToolNode)    (Purchase Flow, HITL)
                              -> trade_collect
           --cond--> response_synthesizer               ("chat": no tools)
      (active flows converge) -> processing -> state_updater
           -> response_synthesizer
           -> approval_gate --(approve)--> END
                            --(revise)---> response_synthesizer

The purchase flow follows the canonical *agent -> ToolNode* loop: ``trader``
emits a tool call, the prebuilt :class:`~langgraph.prebuilt.ToolNode` executes
``purchase_stock`` (which ``interrupt``s for human approval), and the result is
lifted back into state by ``trade_collect``.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from . import agents, config, tools
from .state import AgentState


def build_graph(checkpointer=None):
    """Construct and compile the workflow graph."""
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("supervisor", agents.supervisor)
    g.add_node("stock_fetcher", agents.stock_fetcher)
    g.add_node("risk_calculator", agents.risk_calculator)
    g.add_node("news_agent", agents.news_agent)
    g.add_node("trader", agents.trader)
    # Prebuilt ToolNode executes the emitted tool call on a private channel so
    # the raw tool-call/result messages never leak into the chat transcript.
    g.add_node("trade_tools", ToolNode(tools.TRADE_TOOLS, messages_key="trade_messages"))
    g.add_node("trade_collect", agents.trade_collect)
    g.add_node("processing", agents.processing)
    g.add_node("state_updater", agents.state_updater)
    g.add_node("response_synthesizer", agents.response_synthesizer)
    g.add_node("approval_gate", agents.approval_gate)

    g.add_edge(START, "supervisor")

    # Conditional routing: jump to only the workflows the supervisor selected.
    g.add_conditional_edges(
        "supervisor",
        agents.route_from_supervisor,
        {
            "stock_fetcher": "stock_fetcher",
            "news_agent": "news_agent",
            "trader": "trader",
            "response_synthesizer": "response_synthesizer",
        },
    )

    # Stock Analysis Flow, then conditionally continue down the skip-chain.
    g.add_edge("stock_fetcher", "risk_calculator")
    g.add_conditional_edges(
        "risk_calculator",
        agents.route_after_stocks,
        {"news_agent": "news_agent", "trader": "trader", "processing": "processing"},
    )

    # News Analysis Flow, then conditionally continue.
    g.add_conditional_edges(
        "news_agent",
        agents.route_after_news,
        {"trader": "trader", "processing": "processing"},
    )

    # Purchase Flow: agent -> ToolNode -> collect (only if a tool call exists).
    g.add_conditional_edges(
        "trader",
        agents.route_after_trader,
        {"trade_tools": "trade_tools", "processing": "processing"},
    )
    g.add_edge("trade_tools", "trade_collect")
    g.add_edge("trade_collect", "processing")

    # Aggregate -> persist -> synthesize.
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
