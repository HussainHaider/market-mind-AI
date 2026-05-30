"""Shared graph state passed between every node in the workflow."""

from __future__ import annotations

from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """The shared state exchanged between nodes throughout the workflow.

    Only ``messages`` is required on input; every other field is populated
    incrementally as the graph executes.
    """

    # Conversation history (auto-merged by LangGraph).
    messages: Annotated[List[BaseMessage], add_messages]

    # The latest user query, lifted out of ``messages`` by the supervisor.
    query: str

    # Routing decision produced by the Supervisor Agent, e.g. ["stocks", "news"].
    routes: List[str]

    # Entities extracted from the query.
    ticker: Optional[str]
    quantity: Optional[int]

    # Stock Analysis Flow outputs.
    stock_data: dict
    risk_score: dict

    # News Analysis Flow output.
    news_results: dict

    # Purchase Stock Flow output.
    purchase: dict

    # Processing Layer aggregate + tracking.
    aggregated: dict
    tool_history: List[str]

    # Final user-facing answer.
    final_response: str

    # Human Approval Gate bookkeeping.
    needs_approval: bool
    revision_notes: str
