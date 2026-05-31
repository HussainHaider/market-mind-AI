"""Graph nodes (agents) for the MarketMind workflow.

Each node operates on :class:`~marketmind.state.AgentState`. The Supervisor
populates ``state["routes"]`` and the flow nodes (stock / news / purchase) are
*gated* on those routes, so the pipeline executes deterministically while still
honouring the router concept from the architecture spec.

Everything degrades gracefully without an OpenAI key: routing falls back to
keyword heuristics and the response synthesiser falls back to a template.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
import json
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from . import config, tools
from .state import AgentState

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm():
    """Return a configured ChatOpenAI instance, or ``None`` when offline."""
    if not config.has_openai():
        return None

    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Lightweight entity extraction (used as fallback / pre-fill)
# ---------------------------------------------------------------------------
_COMPANY_TO_TICKER = {
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
    "ibm": "IBM", "oracle": "ORCL", "salesforce": "CRM",
    "boeing": "BA", "disney": "DIS", "coca cola": "KO", "coca-cola": "KO",
    "walmart": "WMT", "nike": "NKE", "starbucks": "SBUX", "uber": "UBER",
}

_STOPWORDS = {
    "BUY", "SELL", "STOCK", "STOCKS", "SHARE", "SHARES", "NEWS", "PRICE",
    "RISK", "THE", "AND", "FOR", "WHAT", "HOW", "OF", "IS", "ARE", "ME",
    "MY", "ON", "IN", "TO", "A", "AN", "USD", "EUR", "PE", "FX",
}


def extract_ticker(text: str) -> Optional[str]:
    lowered = text.lower()
    for name, ticker in _COMPANY_TO_TICKER.items():
        if name in lowered:
            return ticker
    # Cashtags like $AAPL.
    cashtag = re.search(r"\$([A-Za-z]{1,5})", text)
    if cashtag:
        return cashtag.group(1).upper()
    # Bare uppercase tokens that look like tickers.
    for token in re.findall(r"\b[A-Z]{1,5}\b", text):
        if token not in _STOPWORDS:
            return token
    return None


def extract_quantity(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{1,7})\s*(?:shares?|stocks?|units?)?\b", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


_BUY_KEYWORDS = ["buy", "purchase", "acquire", "invest in"]
_STOCK_KEYWORDS = [
    "price", "stock", "share", "market cap", "p/e", "pe ratio", "volume",
    "valuation", "fundamental", "risk", "volatility", "fx", "exchange rate",
]
_NEWS_KEYWORDS = ["news", "sentiment", "headline", "analyst", "earnings", "outlook"]
_FINANCE_KEYWORDS = _BUY_KEYWORDS + _STOCK_KEYWORDS + _NEWS_KEYWORDS


def _is_financial(text: str) -> bool:
    """Whether a query is actually about markets/finance.

    Used as a cheap gate so greetings, small-talk and out-of-scope ("garbage")
    input skip the LLM router and all the network-bound tools entirely.
    """
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(kw in lowered for kw in _FINANCE_KEYWORDS):
        return True
    return extract_ticker(text) is not None


def _keyword_routes(text: str) -> List[str]:
    lowered = text.lower()
    routes: List[str] = []
    if any(w in lowered for w in _BUY_KEYWORDS):
        routes.append("buy")
    if any(w in lowered for w in _STOCK_KEYWORDS):
        routes.append("stocks")
    if any(w in lowered for w in _NEWS_KEYWORDS):
        routes.append("news")
    if not routes:
        # A ticker on its own implies a stock lookup; anything else is chat.
        routes.append("stocks" if extract_ticker(text) else "chat")
    return routes


# ---------------------------------------------------------------------------
# 1. Supervisor Agent
# ---------------------------------------------------------------------------
_SUPERVISOR_PROMPT = """You are the supervisor of a financial research agent.
Classify the user's request into one or more workflows and extract entities.

Available workflows:
- "stocks": price, fundamentals (market cap, P/E, volume), FX, and risk metrics.
- "news": financial news headlines and market sentiment.
- "buy": the user wants to purchase shares.
- "chat": greetings, small talk, capability questions, or anything not about
  markets/finance. Use this alone when nothing financial is being asked.

Respond with ONLY a JSON object:
{{"routes": ["stocks"|"news"|"buy"|"chat", ...], "ticker": "SYMBOL or null", "quantity": <int or null>}}

User request: {query}
"""

_VALID_ROUTES = {"stocks", "news", "buy", "chat"}


def supervisor(state: AgentState) -> dict:
    # Lift the latest human message into ``query``.
    query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Fast path: greetings / small talk / out-of-scope input. Skip the LLM
    # router AND every network-bound tool so these turns respond instantly
    # instead of running a slow news/market lookup on irrelevant text.
    if not _is_financial(query):
        return {
            "query": query,
            "routes": ["chat"],
            "ticker": None,
            "quantity": None,
            "tool_history": [],
            "needs_approval": False,
        }

    routes: List[str] = []
    ticker = extract_ticker(query)
    quantity = extract_quantity(query)

    llm = get_llm()
    if llm is not None and query:
        try:
            resp = llm.invoke(_SUPERVISOR_PROMPT.format(query=query))
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            payload = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            routes = [r for r in payload.get("routes", []) if r in _VALID_ROUTES]
            ticker = payload.get("ticker") or ticker
            if isinstance(payload.get("quantity"), int):
                quantity = payload["quantity"]
        except Exception:
            routes = []

    if not routes:
        routes = _keyword_routes(query)

    # "chat" is exclusive: if any real financial workflow was selected, drop it.
    financial = [r for r in routes if r != "chat"]
    routes = financial or ["chat"]

    if ticker in (None, "null", ""):
        ticker = None

    return {
        "query": query,
        "routes": routes,
        "ticker": ticker,
        "quantity": quantity,
        "tool_history": [],
        "needs_approval": "buy" in routes,
    }


# ---------------------------------------------------------------------------
# Stock Analysis Flow
# ---------------------------------------------------------------------------
def stock_fetcher(state: AgentState) -> dict:
    if "stocks" not in state.get("routes", []):
        return {}  # passthrough: this flow was not selected
    ticker = state.get("ticker")
    if not ticker:
        return {"stock_data": {"error": "no ticker identified in the request"}}

    history = list(state.get("tool_history", []))
    price = tools.get_stock_price.invoke({"symbol": ticker})
    fundamentals = tools.get_stock_fundamentals.invoke({"symbol": ticker})
    history += ["get_stock_price", "get_stock_fundamentals"]

    stock_data = {**price, **{k: v for k, v in fundamentals.items() if k != "error"}}

    # Optional FX if the user asked about exchange rates.
    if "fx" in state.get("query", "").lower() or "exchange rate" in state.get("query", "").lower():
        stock_data["fx"] = tools.get_fx_rate.invoke({"from_currency": "EUR", "to_currency": "USD"})
        history.append("get_fx_rate")

    return {"stock_data": stock_data, "tool_history": history}


def risk_calculator(state: AgentState) -> dict:
    if "stocks" not in state.get("routes", []):
        return {}  # passthrough
    ticker = state.get("ticker")
    if not ticker:
        return {"risk_score": {"error": "no ticker for risk analysis"}}
    prices = tools.get_price_history(ticker)
    metrics = tools.compute_risk_metrics(prices)
    history = list(state.get("tool_history", [])) + ["compute_risk_metrics"]
    return {"risk_score": metrics, "tool_history": history}


# ---------------------------------------------------------------------------
# News Analysis Flow
# ---------------------------------------------------------------------------
def news_agent(state: AgentState) -> dict:
    if "news" not in state.get("routes", []):
        return {}  # passthrough
    ticker = state.get("ticker")
    query = state.get("query", "")
    topic = f"{ticker} stock" if ticker else query
    results = tools.analyze_news_sentiment.invoke({"query": topic})
    history = list(state.get("tool_history", [])) + ["analyze_news_sentiment"]
    return {"news_results": results, "tool_history": history}


# ---------------------------------------------------------------------------
# Purchase Stock Flow (human-in-the-loop happens inside the tool)
# ---------------------------------------------------------------------------
def purchase_agent(state: AgentState) -> dict:
    if "buy" not in state.get("routes", []):
        return {}  # passthrough
    ticker = state.get("ticker")
    quantity = state.get("quantity") or 1
    if not ticker:
        return {"purchase": {"status": "error", "message": "no ticker to purchase"}}
    result = tools.purchase_stock.invoke({"symbol": ticker, "quantity": quantity})
    history = list(state.get("tool_history", [])) + ["purchase_stock"]
    return {"purchase": result, "tool_history": history}


# ---------------------------------------------------------------------------
# Processing Layer + State Updater
# ---------------------------------------------------------------------------
def processing(state: AgentState) -> dict:
    aggregated = {
        "ticker": state.get("ticker"),
        "routes": state.get("routes", []),
        "stock_data": state.get("stock_data"),
        "risk_score": state.get("risk_score"),
        "news_results": state.get("news_results"),
        "purchase": state.get("purchase"),
    }
    return {"aggregated": aggregated}


def state_updater(state: AgentState) -> dict:
    # Deduplicate the recorded tool history while preserving order.
    seen = set()
    ordered = []
    for name in state.get("tool_history", []):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return {"tool_history": ordered}


# ---------------------------------------------------------------------------
# Response Synthesizer
# ---------------------------------------------------------------------------
def _template_response(state: AgentState) -> str:
    parts: List[str] = []
    ticker = state.get("ticker")
    sd = state.get("stock_data") or {}
    rs = state.get("risk_score") or {}
    nr = state.get("news_results") or {}
    pr = state.get("purchase") or {}

    if ticker:
        parts.append(f"## {ticker} summary")

    if sd and "error" not in sd:
        line = []
        if sd.get("price"):
            line.append(f"Price: {sd['price']}")
        if sd.get("market_cap"):
            line.append(f"Market cap: {sd['market_cap']:,}")
        if sd.get("pe_ratio"):
            line.append(f"P/E: {round(sd['pe_ratio'], 2)}")
        if sd.get("volume"):
            line.append(f"Volume: {sd['volume']:,}")
        if line:
            parts.append("**Market data** — " + " | ".join(line))

    if rs and "error" not in rs:
        parts.append(
            f"**Risk** — level {rs.get('risk_level')} "
            f"(volatility {rs.get('volatility')}, Sharpe {rs.get('sharpe_ratio')}, "
            f"max drawdown {rs.get('max_drawdown')})."
        )

    if nr and nr.get("headlines"):
        heads = "\n".join(f"- {h}" for h in nr["headlines"][:5])
        parts.append(f"**News sentiment: {nr.get('sentiment', 'neutral')}**\n{heads}")

    if pr:
        parts.append(f"**Order** — {pr.get('message', pr.get('status'))}")

    if not parts:
        return "I couldn't find enough information to answer that. Try naming a ticker (e.g. AAPL)."
    return "\n\n".join(parts)


_SYNTH_PROMPT = """You are a financial research assistant. Using ONLY the data
below, write a concise, well-structured answer for the user. Summarise market
data, risk and news sentiment, and give a brief, clearly-hedged recommendation.
Never invent numbers that are not present.

User question: {query}

Aggregated data (JSON):
{data}
"""

_CHAT_RESPONSE = (
    "Hi! I'm **MarketMind AI**, your financial research assistant. I can help you with:\n\n"
    "- **Stock data** — live price, market cap, P/E ratio, volume and FX rates\n"
    "- **Risk analysis** — volatility, Sharpe ratio, max drawdown and an overall risk level\n"
    "- **News & sentiment** — the latest headlines and market mood for a company\n"
    "- **Simulated trading** — buy shares, with a human approval step before anything executes\n\n"
    "Try asking me something like:\n"
    "- *What's the price and risk for AAPL?*\n"
    "- *Latest news sentiment on Tesla*\n"
    "- *Buy 10 shares of MSFT*\n\n"
    "What would you like to look into?"
)


def response_synthesizer(state: AgentState) -> dict:
    # General chat / greetings / out-of-scope: return the capability overview
    # directly. No LLM or tool calls, so this is effectively instant.
    if state.get("routes") == ["chat"]:
        return {"final_response": _CHAT_RESPONSE, "messages": [AIMessage(content=_CHAT_RESPONSE)]}

    llm = get_llm()
    final = ""
    if llm is not None:
        try:
            data = json.dumps(state.get("aggregated", {}), default=str, indent=2)
            prompt = _SYNTH_PROMPT.format(query=state.get("query", ""), data=data)
            if state.get("revision_notes"):
                prompt += f"\n\nThe user asked to revise the previous answer: {state['revision_notes']}"
            resp = llm.invoke(prompt)
            final = resp.content if isinstance(resp.content, str) else str(resp.content)
        except Exception:
            final = ""
    if not final:
        final = _template_response(state)

    return {"final_response": final, "messages": [AIMessage(content=final)]}


# ---------------------------------------------------------------------------
# Human Approval Gate (structural conditional edge)
# ---------------------------------------------------------------------------
def approval_gate(state: AgentState) -> dict:
    # The critical-action approval (stock purchase) is handled in-tool via
    # ``interrupt``. This node is the final review checkpoint.
    return {"needs_approval": False}


def approval_decision(state: AgentState) -> str:
    return "revise" if state.get("revision_notes") else "approve"
