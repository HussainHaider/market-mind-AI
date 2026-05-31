from __future__ import annotations

import sqlite3
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from . import agents, config, tools
from .state import AgentState, SubAgentState

# === Multi-agent architecture scaffold ===
# Each sub-agent will be a compiled sub-graph with its own LLM and tools.
# The orchestrator LLM will call these as tools.

# --- LLM instances for each agent (can share model name, but separate objects) ---
stock_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
news_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
trade_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
chat_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
orch_llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)

# --- Tool bindings for each agent ---
stock_tools = tools.STOCK_TOOLS
news_tools = tools.NEWS_TOOLS
trade_tools = tools.TRADE_TOOLS
chat_tools = []

# --- bind tools to each agent's LLM ---
stock_llm_with_tools = stock_llm.bind_tools(stock_tools)
news_llm_with_tools = news_llm.bind_tools(news_tools)
trade_llm_with_tools = trade_llm.bind_tools(trade_tools)
chat_llm_with_tools = chat_llm.bind_tools(chat_tools)


# --- build the stock analyst sub-graph ---
def stock_agent_node(state):
    return {"messages": [stock_llm_with_tools.invoke(state["messages"])]}


stock_graph = StateGraph(SubAgentState)
stock_graph.add_node("agent",  stock_agent_node)
stock_graph.add_node("tools",  ToolNode(stock_tools))

stock_graph.add_edge(START, "agent")

stock_graph.add_conditional_edges("agent", tools_condition)
stock_graph.add_edge("tools", "agent")
stock_graph.add_edge("agent", END)
stock_agent = stock_graph.compile()


# Wrap each sub-graph as a callable tool
stock_tool = stock_agent.as_tool(
    name="stock_analyst",
    description="Fetches price, fundamentals, and risk metrics for a ticker. Input: ticker symbol."
)

orch_llm_with_tools = orch_llm.bind_tools([stock_tool])  # Add other sub-agent tools as they are implemented

def orchestrator_node(state: AgentState):
    system = SystemMessage(content="""You are a financial research orchestrator.
    Given the user's request, decide which specialist agents to call.
    You may call multiple agents in sequence. Return their combined results.""")
    response = orch_llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

# Main graph is now just: orchestrator → ToolNode(sub-agents) → synthesizer
main_graph = StateGraph(AgentState)
main_graph.add_node("orchestrator",  orchestrator_node)
main_graph.add_node("agent_tools",   ToolNode([stock_tool]))
main_graph.add_node("synthesizer",   response_synthesizer_node)
main_graph.add_node("approval_gate", approval_gate_node)

main_graph.add_edge(START, "orchestrator")
main_graph.add_edge("orchestrator", "agent_tools")
main_graph.add_edge("agent_tools", "synthesizer")
main_graph.add_edge("synthesizer", "approval_gate")
main_graph.add_conditional_edges(
    "approval_gate",
    agents.approval_decision,
    {"approve": END, "revise": "synthesizer"},
)

# --- Sub-graph placeholders (to be implemented) ---
# def stock_agent_node(state): ...
# def news_agent_node(state): ...
# def trade_agent_node(state): ...
# def chat_agent_node(state): ...
#
# stock_graph = ...
# news_graph = ...
# trade_graph = ...
# chat_graph = ...
#
# stock_tool = ...
# news_tool = ...
# trade_tool = ...
# chat_tool = ...

# --- Orchestrator node placeholder (to be implemented) ---
# def orchestrator_node(state): ...

# --- Main graph wiring (to be implemented) ---
# main_graph = StateGraph(AgentState)
# main_graph.add_node("orchestrator", orchestrator_node)
# main_graph.add_node("agent_tools", ToolNode([stock_tool, news_tool, trade_tool, chat_tool]))
# main_graph.add_node("synthesizer", agents.response_synthesizer)
# main_graph.add_node("approval_gate", agents.approval_gate)
# main_graph.set_entry_point("orchestrator")
# main_graph.add_conditional_edges("orchestrator", tools_condition)
# main_graph.add_edge("agent_tools", "orchestrator")
# main_graph.add_conditional_edges("orchestrator", lambda s: "synthesizer" if ... else "agent_tools")
# ...


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
