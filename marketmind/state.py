"""Shared graph state passed between every node in the workflow.

Includes typed result schemas for each sub-agent in the multi-agent architecture.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState


class StockAnalysisResult(TypedDict, total=False):
    """Output schema for the Stock Analyst sub-agent."""
    symbol: str
    price: Optional[float]
    change_percent: Optional[str]
    volume: Optional[int]
    market_cap: Optional[int]
    pe_ratio: Optional[float]
    name: Optional[str]
    currency: str
    volatility: Optional[float]
    annual_return: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    risk_level: Optional[str]
    fx_rate: Optional[dict]
    error: Optional[str]


class NewsAnalysisResult(TypedDict, total=False):
    """Output schema for the News Analyst sub-agent."""
    query: str
    headlines: List[str]
    sentiment: str
    score: float
    positive: int
    negative: int
    error: Optional[str]


class TradeResult(TypedDict, total=False):
    """Output schema for the Trade sub-agent."""
    status: str
    symbol: Optional[str]
    quantity: Optional[int]
    message: str


class ChatResult(TypedDict, total=False):
    """Output schema for the Chat sub-agent."""
    response: str


class AgentState(TypedDict, total=False):
    """The shared state exchanged between nodes throughout the workflow.

    Only ``messages`` is required on input; every other field is populated
    incrementally as the graph executes.
    """

    # Conversation history (auto-merged by LangGraph).
    messages: Annotated[List[BaseMessage], add_messages]

    # The latest user query, lifted out of ``messages`` by the orchestrator.
    query: str

    # Entities extracted from the query.
    ticker: Optional[str]
    quantity: Optional[int]

    # Sub-agent results (typed outputs from each specialist agent).
    stock_result: Optional[StockAnalysisResult]
    news_result: Optional[NewsAnalysisResult]
    trade_result: Optional[TradeResult]
    chat_result: Optional[ChatResult]

    # Aggregated data from all sub-agents for the synthesizer.
    aggregated: dict

    # Track which sub-agents were invoked.
    agents_called: List[str]

    # Final user-facing answer.
    final_response: str

    # Human Approval Gate bookkeeping.
    needs_approval: bool
    revision_notes: str

    # Pending trade awaiting human approval (HITL at main graph level).
    pending_trade: Optional[dict]  # {"symbol": str, "quantity": int, "prompt": str}
    trade_approved: Optional[bool]  # None = pending, True = approved, False = declined

    # Private channel for trade tool execution (isolates tool messages from chat).
    trade_messages: Annotated[List[BaseMessage], add_messages]


class SubAgentState(MessagesState):
    """Minimal state for each sub-agent graph. Only 'messages' is needed."""
    pass
