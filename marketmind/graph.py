"""Multi-agent graph architecture for MarketMind AI.

This module implements a true multi-agent system where:
- An Orchestrator agent routes requests to specialist sub-agents
- Each sub-agent is a compiled graph with its own LLM and tools
- Sub-agents are wrapped as tools and invoked by the orchestrator
- A Response Synthesizer aggregates all sub-agent outputs
- An Approval Gate provides human-in-the-loop for the final response
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from . import config, tools
from .state import AgentState, SubAgentState


# ===========================================================================
# LLM instances for each agent (separate instances, can use same model)
# ===========================================================================
def _get_llm():
    """Create a configured ChatOpenAI instance, or None when offline."""
    if not config.has_openai():
        return None
    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)


# ===========================================================================
# Sub-agent system prompts
# ===========================================================================
STOCK_AGENT_PROMPT = """You are a Stock Analyst agent specialized in financial market data.
Your job is to analyze stock prices, fundamentals, and risk metrics for the requested ticker.

You have access to these tools:
- get_stock_price: Get the latest trading price for a ticker
- get_stock_fundamentals: Get market cap, P/E ratio, volume, and company name
- compute_risk_metrics: Calculate volatility, Sharpe ratio, max drawdown, and risk level

When given a ticker symbol, use your tools to gather comprehensive data about the stock.
Call multiple tools as needed to provide a complete analysis.
After gathering data, summarize your findings clearly."""

NEWS_AGENT_PROMPT = """You are a News Analyst agent specialized in financial news and sentiment.
Your job is to search for recent news headlines and analyze market sentiment for a topic.

You have access to these tools:
- search_headlines: Search for recent financial news headlines
- analyze_news_sentiment: Search news and return headlines with sentiment analysis
- get_fx_rate: Get foreign exchange rates between currencies

When given a topic or ticker, search for relevant news and analyze the sentiment.
Provide a clear summary of the market mood and key headlines."""

TRADE_AGENT_PROMPT = """You are a Trade Execution agent responsible for processing stock orders.
Your job is to execute purchase orders when requested by the user.

You have access to this tool:
- purchase_stock: Place a simulated order to buy shares (requires human approval)

IMPORTANT: The purchase_stock tool will pause for human approval before executing.
When asked to buy shares, use the purchase_stock tool with the correct symbol and quantity.
Never execute trades without explicit user request."""

CHAT_AGENT_PROMPT = """You are a helpful Chat agent for MarketMind AI, a financial research assistant.
Your job is to handle greetings, general questions, and explain the system's capabilities.

You do NOT have access to any tools. For financial queries, you should explain what
MarketMind can do:
- Stock data: live price, market cap, P/E ratio, volume and FX rates
- Risk analysis: volatility, Sharpe ratio, max drawdown and risk level
- News & sentiment: the latest headlines and market mood
- Simulated trading: buy shares with human approval

IMPORTANT: For simple greetings like "Hi", "Hello", "Hey", respond in a friendly, conversational way.
For example: "Hi! How can I help you today? I can look up stock prices, analyze risk metrics, 
find recent news and sentiment, or help you explore market data."

Be warm, helpful, and guide users on how to use the system effectively. Keep responses concise
but informative."""


# ===========================================================================
# Build Sub-Agent Graphs
# ===========================================================================
def _build_stock_agent():
    """Build the Stock Analyst sub-agent graph."""
    llm = _get_llm()
    if llm is None:
        return None

    stock_tools = tools.STOCK_TOOLS
    llm_with_tools = llm.bind_tools(stock_tools)

    def agent_node(state: SubAgentState):
        system = SystemMessage(content=STOCK_AGENT_PROMPT)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(stock_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _build_news_agent():
    """Build the News Analyst sub-agent graph."""
    llm = _get_llm()
    if llm is None:
        return None

    news_tools = tools.NEWS_TOOLS
    llm_with_tools = llm.bind_tools(news_tools)

    def agent_node(state: SubAgentState):
        system = SystemMessage(content=NEWS_AGENT_PROMPT)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(news_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _build_trade_agent():
    """Build the Trade sub-agent graph with HITL interrupt."""
    llm = _get_llm()
    if llm is None:
        return None

    trade_tools = tools.TRADE_TOOLS
    llm_with_tools = llm.bind_tools(trade_tools)

    def agent_node(state: SubAgentState):
        system = SystemMessage(content=TRADE_AGENT_PROMPT)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(trade_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def _build_chat_agent():
    """Build the Chat sub-agent graph (no tools)."""
    llm = _get_llm()
    if llm is None:
        return None

    def agent_node(state: SubAgentState):
        system = SystemMessage(content=CHAT_AGENT_PROMPT)
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(SubAgentState)
    graph.add_node("agent", agent_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()


# ===========================================================================
# Lazy initialization of sub-agents (avoids import-time hangs)
# ===========================================================================
_subagent_cache = {}


def _invoke_subagent(agent, message: str) -> str:
    """Helper to invoke a sub-agent graph and extract the final response."""
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    # Extract the last AI message content from the result
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return "No response from agent."


def _get_subagent_tools():
    """Get or create sub-agent tool wrappers (lazy initialization).
    
    Uses custom @tool decorated functions instead of .as_tool() to ensure
    proper compatibility with ToolNode and tool_call_id handling.
    """
    if "tools" in _subagent_cache:
        return _subagent_cache["tools"]

    subagent_tools = []

    stock_agent = _build_stock_agent()
    if stock_agent is not None:
        @tool
        def stock_analyst(query: str) -> str:
            """Analyze a stock ticker. Fetches price, fundamentals (market cap, P/E, volume), and risk metrics (volatility, Sharpe ratio, max drawdown). Input: a message describing what stock to analyze, e.g. 'Analyze AAPL stock'."""
            return _invoke_subagent(stock_agent, query)
        subagent_tools.append(stock_analyst)

    news_agent = _build_news_agent()
    if news_agent is not None:
        @tool
        def news_analyst(query: str) -> str:
            """Search for financial news and analyze sentiment for a company or topic. Returns recent headlines and an overall sentiment score. Input: a message describing what to search, e.g. 'Get news sentiment for Tesla'."""
            return _invoke_subagent(news_agent, query)
        subagent_tools.append(news_analyst)

    trade_agent = _build_trade_agent()
    if trade_agent is not None:
        @tool
        def trade_executor(query: str) -> str:
            """Execute a stock purchase order. IMPORTANT: This will pause for human approval. Input: a message with the trade details, e.g. 'Buy 10 shares of MSFT'."""
            result = trade_agent.invoke({"messages": [HumanMessage(content=query)]})
            messages = result.get("messages", [])
            
            # Check for pending approval in tool messages (raw tool output)
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    content = str(msg.content)
                    if "pending_approval" in content.lower():
                        # Return the raw pending approval info for main graph to detect
                        return content
            
            # Otherwise return the AI response as normal
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content if isinstance(msg.content, str) else str(msg.content)
            return "No response from agent."
        subagent_tools.append(trade_executor)

    chat_agent = _build_chat_agent()
    if chat_agent is not None:
        @tool
        def chat_assistant(query: str) -> str:
            """Handle general conversation, greetings, and capability questions. Use this for non-financial queries or when the user asks what the system can do. Input: the user's message or question."""
            return _invoke_subagent(chat_agent, query)
        subagent_tools.append(chat_assistant)

    _subagent_cache["tools"] = subagent_tools
    return subagent_tools


# ===========================================================================
# Orchestrator Agent
# ===========================================================================
ORCHESTRATOR_PROMPT = """You are the Orchestrator agent for MarketMind AI, a financial research assistant.
Your job is to understand the user's request and route it to the appropriate specialist agents.

You have access to these specialist agents as tools:
- stock_analyst: For stock prices, fundamentals, and risk metrics
- news_analyst: For financial news headlines and sentiment analysis
- trade_executor: For executing stock purchase orders (requires human approval)
- chat_assistant: For greetings, general questions, and capability explanations

Guidelines:
1. For stock/price/risk queries, call the stock_analyst
2. For news/sentiment queries, call the news_analyst
3. For purchase/buy requests, call the trade_executor
4. For greetings or "what can you do" questions, call the chat_assistant
5. You MAY call multiple agents for complex queries (e.g., "price and news for AAPL")
6. Extract the ticker symbol and include it in your tool call

After receiving results from agents, return them as-is. The response synthesizer will format the final answer."""


def _orchestrator_node(state: AgentState):
    """Orchestrator decides which sub-agents to invoke."""
    llm = _get_llm()

    # Extract the latest user message
    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Fast path: if no LLM, use keyword routing
    if llm is None:
        return _fallback_orchestrator(state, query)

    # Bind sub-agent tools to the orchestrator LLM
    subagent_tools = _get_subagent_tools()
    if not subagent_tools:
        return _fallback_orchestrator(state, query)

    llm_with_tools = llm.bind_tools(subagent_tools)

    # Build message list: system prompt + full conversation history
    messages = state.get("messages", [])
    system = SystemMessage(content=ORCHESTRATOR_PROMPT)
    
    # Include system prompt and all messages (user, AI with tool_calls, tool results)
    full_messages = [system] + list(messages)
    
    response = llm_with_tools.invoke(full_messages)

    # Track which agents were called based on tool calls
    agents_called = list(state.get("agents_called", []))
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            agent_name = tc.get("name", "")
            if agent_name and agent_name not in agents_called:
                agents_called.append(agent_name)

    return {
        "messages": [response],
        "query": query,
        "agents_called": agents_called,
    }


def _fallback_orchestrator(state: AgentState, query: str):
    """Keyword-based routing when no LLM is available."""
    from . import agents as agent_helpers

    lowered = query.lower()
    routes = []

    if any(w in lowered for w in ["buy", "purchase", "acquire"]):
        routes.append("trade")
    if any(w in lowered for w in ["price", "stock", "risk", "fundamental", "volatility", "p/e"]):
        routes.append("stock")
    if any(w in lowered for w in ["news", "sentiment", "headline"]):
        routes.append("news")
    if not routes:
        routes.append("chat")

    ticker = agent_helpers.extract_ticker(query)
    quantity = agent_helpers.extract_quantity(query)

    # Build a synthetic response indicating which agents to call
    agent_calls = ", ".join(routes)
    content = f"[Fallback routing: {agent_calls}]"
    if ticker:
        content += f" [Ticker: {ticker}]"
    if quantity:
        content += f" [Quantity: {quantity}]"

    return {
        "messages": [AIMessage(content=content)],
        "query": query,
        "ticker": ticker,
        "quantity": quantity,
        "agents_called": routes,
    }


# ===========================================================================
# Trade Approval Handler (HITL at main graph level)
# ===========================================================================
def _extract_pending_trade(state: AgentState) -> dict | None:
    """Check if any tool output contains a pending trade approval request."""
    for msg in reversed(state.get("messages", [])):
        content = None
        if isinstance(msg, ToolMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        
        if not content:
            continue
            
        # Check for pending_approval status in the content
        if "pending_approval" in content.lower():
            try:
                # Try multiple regex patterns for different formats
                # Pattern 1: 'symbol': 'MSFT' format
                symbol_match = re.search(r"['\"]symbol['\"]\s*:\s*['\"]([A-Z]+)['\"]", content)
                # Pattern 2: "symbol": "MSFT" format  
                if not symbol_match:
                    symbol_match = re.search(r"symbol['\"]?\s*:\s*['\"]?([A-Z]+)['\"]?", content, re.IGNORECASE)
                
                # Pattern 1: 'quantity': 10 format
                quantity_match = re.search(r"['\"]quantity['\"]\s*:\s*(\d+)", content)
                # Pattern 2: quantity: 10 format
                if not quantity_match:
                    quantity_match = re.search(r"quantity['\"]?\s*:\s*(\d+)", content, re.IGNORECASE)
                
                # Pattern for prompt
                prompt_match = re.search(r"['\"]prompt['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
                
                if symbol_match and quantity_match:
                    symbol = symbol_match.group(1).upper()
                    quantity = int(quantity_match.group(1))
                    prompt = prompt_match.group(1) if prompt_match else f"Approve buying {quantity} shares of {symbol}? (yes/no)"
                    return {
                        "symbol": symbol,
                        "quantity": quantity,
                        "prompt": prompt,
                    }
            except Exception:
                pass
    return None


def _trade_approval_node(state: AgentState):
    """Handle HITL interrupt for pending trades at the main graph level."""
    # Check if we already have a pending trade that was approved/declined
    if state.get("trade_approved") is not None:
        # Trade decision already made, execute the trade
        pending = state.get("pending_trade")
        if pending:
            result = tools.execute_trade(
                pending["symbol"],
                pending["quantity"],
                state["trade_approved"]
            )
            return {
                "trade_result": result,
                "pending_trade": None,
                "messages": [AIMessage(content=result["message"])],
            }
        return {}
    
    # Check for a new pending trade
    pending = _extract_pending_trade(state)
    if pending:
        # We have a pending trade, interrupt for human approval
        decision = interrupt({
            "action": "purchase",
            "symbol": pending["symbol"],
            "quantity": pending["quantity"],
            "prompt": pending["prompt"],
        })
        
        # Process the decision
        approved = isinstance(decision, str) and decision.strip().lower() in {"yes", "y", "approve"}
        result = tools.execute_trade(pending["symbol"], pending["quantity"], approved)
        
        return {
            "trade_result": result,
            "pending_trade": None,
            "trade_approved": None,  # Reset for next trade
            "messages": [AIMessage(content=result["message"])],
        }
    
    # No pending trade
    return {}


def _needs_trade_approval(state: AgentState) -> str:
    """Route to trade approval if there's a pending trade, else back to orchestrator."""
    pending = _extract_pending_trade(state)
    if pending:
        return "trade_approval"
    return "continue"  # Go back to orchestrator for normal flow


# ===========================================================================
# Response Synthesizer
# ===========================================================================
SYNTHESIZER_PROMPT = """You are the Response Synthesizer for MarketMind AI.
Your job is to take the outputs from specialist agents and create a clear, well-structured
response for the user.

Using the data provided, write a concise financial summary that includes:
- Stock data (if available): price, market cap, P/E ratio, volume
- Risk metrics (if available): volatility, Sharpe ratio, max drawdown, risk level
- News sentiment (if available): key headlines and overall sentiment
- Trade execution status (if applicable): 
  * If status is "executed" - confirm the purchase was completed successfully
  * If status is "cancelled" - confirm the purchase was declined/cancelled
  * If status is "pending_approval" - this should not appear here (handled separately)

Guidelines:
- Use markdown formatting for readability
- Never invent numbers that aren't in the data
- For trade results, be clear and direct about whether the trade was executed or cancelled
- Give a brief, clearly-hedged recommendation if appropriate
- Be concise but comprehensive

User question: {query}

Agent outputs:
{data}
"""

CHAT_RESPONSE = (
    "Hi! How can I help you today?\n\n"
    "I'm **MarketMind AI**, your financial research assistant. I can:\n\n"
    "- Look up **stock prices** and fundamentals (market cap, P/E ratio, volume)\n"
    "- Analyze **risk metrics** (volatility, Sharpe ratio, max drawdown)\n"
    "- Find **news headlines** and sentiment for any company\n"
    "- Execute **simulated trades** (with your approval)\n\n"
    "Try asking something like:\n"
    "- *What's the price and risk for AAPL?*\n"
    "- *Latest news sentiment on Tesla*\n"
    "- *Buy 10 shares of MSFT*"
)


def _response_synthesizer_node(state: AgentState):
    """Aggregate sub-agent results into a final response."""
    query = state.get("query", "")
    agents_called = state.get("agents_called", [])

    # Check if this was just a chat/greeting query (no financial data needed)
    chat_only = (
        agents_called in [["chat"], ["chat_assistant"]]
        or (len(agents_called) == 1 and agents_called[0] in ("chat", "chat_assistant"))
        or not agents_called
    )
    
    if chat_only:
        # For chat-only queries, extract and return the chat agent's response directly
        msgs = state.get("messages", [])
        
        # Look for the chat agent's actual response in tool messages
        for msg in reversed(msgs):
            # Check for tool message content (the chat_assistant's response)
            if hasattr(msg, "content") and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # Skip orchestrator routing messages and empty responses
                if content and not content.startswith("[Fallback") and "tool_calls" not in content.lower():
                    # This is likely the chat agent's response - use it directly
                    if len(content) > 50 and ("MarketMind" in content or "help" in content.lower() or "?" in content):
                        return {
                            "final_response": content,
                            "messages": [AIMessage(content=content)],
                        }
        
        # Fallback to the default chat response
        return {
            "final_response": CHAT_RESPONSE,
            "messages": [AIMessage(content=CHAT_RESPONSE)],
        }

    # Gather all data from messages
    trade_result = state.get("trade_result")
    aggregated = {
        "stock_result": state.get("stock_result"),
        "news_result": state.get("news_result"),
        "trade_result": trade_result,
        "agents_called": agents_called,
    }
    
    # If there's a trade result with executed/cancelled status, make it prominent
    if trade_result and trade_result.get("status") in ("executed", "cancelled"):
        # Include the trade message directly in outputs for the LLM to see
        trade_msg = trade_result.get("message", "")
        if trade_msg:
            aggregated["trade_execution_message"] = trade_msg

    # Also extract tool results from messages
    tool_outputs = []
    for msg in state.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and not content.startswith("[Fallback"):
                tool_outputs.append(content)

    llm = _get_llm()
    if llm is None:
        final = _template_response(state, aggregated, tool_outputs)
    else:
        try:
            data_str = json.dumps(aggregated, default=str, indent=2)
            data_str += "\n\nRaw outputs:\n" + "\n---\n".join(tool_outputs[-5:])

            prompt = SYNTHESIZER_PROMPT.format(query=query, data=data_str)
            if state.get("revision_notes"):
                prompt += f"\n\nRevision requested: {state['revision_notes']}"

            response = llm.invoke(prompt)
            final = response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            final = _template_response(state, aggregated, tool_outputs)

    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "aggregated": aggregated,
    }


def _template_response(state: AgentState, aggregated: dict, tool_outputs: list) -> str:
    """Fallback template-based response when LLM is unavailable."""
    parts = []

    if tool_outputs:
        for output in tool_outputs[-3:]:
            if output and len(output) > 20:
                parts.append(output)

    if not parts:
        return CHAT_RESPONSE

    return "\n\n---\n\n".join(parts)


# ===========================================================================
# Approval Gate
# ===========================================================================
def _approval_gate_node(state: AgentState):
    """Final review checkpoint."""
    return {"needs_approval": False}


def _approval_decision(state: AgentState) -> str:
    """Route based on revision request."""
    return "revise" if state.get("revision_notes") else "approve"


# ===========================================================================
# Main Graph Assembly
# ===========================================================================
def build_graph(checkpointer=None):
    """Construct and compile the multi-agent workflow graph."""
    graph = StateGraph(AgentState)

    # Get sub-agent tools (lazy initialization)
    subagent_tools = _get_subagent_tools()

    # Add nodes
    graph.add_node("orchestrator", _orchestrator_node)
    graph.add_node("agent_tools", ToolNode(subagent_tools) if subagent_tools else _noop_node)
    graph.add_node("trade_approval", _trade_approval_node)
    graph.add_node("synthesizer", _response_synthesizer_node)
    graph.add_node("approval_gate", _approval_gate_node)

    # Wire the graph
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        tools_condition,
        {"tools": "agent_tools", END: "synthesizer"},
    )
    # After tools execute, check if there's a pending trade needing approval
    graph.add_conditional_edges(
        "agent_tools",
        _needs_trade_approval,
        {"trade_approval": "trade_approval", "continue": "orchestrator"},
    )
    # After trade approval, go to synthesizer to format the response
    graph.add_edge("trade_approval", "synthesizer")
    graph.add_edge("synthesizer", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        _approval_decision,
        {"approve": END, "revise": "synthesizer"},
    )

    return graph.compile(checkpointer=checkpointer)


def _noop_node(state):
    """No-op node when tools are unavailable."""
    return {}


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
