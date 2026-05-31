"""Tools used by the agents: market data, risk metrics, news search and trading.

Every tool is designed to degrade gracefully: if an optional provider (Alpha
Vantage, NewsAPI, network access) is unavailable, the tool falls back to a
secondary source or returns a structured ``error`` field instead of raising.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool
from langgraph.types import interrupt

from . import config

# ``traceable`` makes plain helper functions (risk maths, price history, news
# scoring) show up as their own steps in LangSmith. It degrades to a no-op
# decorator when LangSmith isn't installed.
try:  # pragma: no cover - optional dependency
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(func=None, **_kwargs):  # type: ignore[misc]
        if func is None:
            return lambda f: f
        return func

# ---------------------------------------------------------------------------
# Optional dependencies (imported lazily / defensively)
# ---------------------------------------------------------------------------
try:  # yfinance is the default free market-data source
    import yfinance as yf
except Exception:  # pragma: no cover - optional
    yf = None

try:  # DuckDuckGo search backend
    from ddgs import DDGS
except Exception:  # pragma: no cover - optional
    DDGS = None


# ===========================================================================
# Stock tools
# ===========================================================================
def _alpha_vantage_quote(symbol: str) -> Dict[str, Any]:
    url = (
        "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
        f"&symbol={symbol}&apikey={config.ALPHA_VANTAGE_API_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("Global Quote", {})
    if not data:
        raise ValueError("Alpha Vantage returned no quote")
    return {
        "symbol": symbol.upper(),
        "price": float(data.get("05. price", 0) or 0),
        "change_percent": data.get("10. change percent", "0%"),
        "volume": int(float(data.get("06. volume", 0) or 0)),
        "source": "alpha_vantage",
    }


def _yfinance_quote(symbol: str) -> Dict[str, Any]:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    price = float(getattr(info, "last_price", None) or info.get("lastPrice", 0))
    return {
        "symbol": symbol.upper(),
        "price": price,
        "volume": int(getattr(info, "last_volume", 0) or 0),
        "source": "yfinance",
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest trading price for a ticker symbol (e.g. 'AAPL', 'TSLA').

    Uses Alpha Vantage when an API key is configured, otherwise falls back to
    Yahoo Finance.
    """
    symbol = symbol.upper().strip()
    try:
        if config.ALPHA_VANTAGE_API_KEY:
            return _alpha_vantage_quote(symbol)
        return _yfinance_quote(symbol)
    except Exception as exc:  # try the fallback before giving up
        try:
            return _yfinance_quote(symbol)
        except Exception:
            return {"symbol": symbol, "error": f"price lookup failed: {exc}"}


@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """Fetch fundamentals for a ticker: market cap, P/E ratio and trading volume."""
    symbol = symbol.upper().strip()
    if yf is None:
        return {"symbol": symbol, "error": "yfinance not available"}
    try:
        info = yf.Ticker(symbol).info
        return {
            "symbol": symbol,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "name": info.get("shortName") or info.get("longName"),
            "currency": info.get("currency", "USD"),
        }
    except Exception as exc:
        return {"symbol": symbol, "error": f"fundamentals lookup failed: {exc}"}


@tool
def get_fx_rate(from_currency: str = "EUR", to_currency: str = "USD") -> dict:
    """Get the foreign-exchange rate between two currencies (default EUR->USD)."""
    pair = f"{from_currency.upper()}{to_currency.upper()}=X"
    if yf is None:
        return {"pair": pair, "error": "yfinance not available"}
    try:
        info = yf.Ticker(pair).fast_info
        rate = float(getattr(info, "last_price", None) or info.get("lastPrice", 0))
        return {"pair": pair, "rate": rate}
    except Exception as exc:
        return {"pair": pair, "error": f"fx lookup failed: {exc}"}


@traceable(run_type="tool", name="get_price_history")
def get_price_history(symbol: str, period: str = "6mo") -> List[float]:
    """Return a list of daily closing prices (helper, not an LLM tool)."""
    if yf is None:
        return []
    try:
        hist = yf.Ticker(symbol).history(period=period)
        return [float(p) for p in hist["Close"].dropna().tolist()]
    except Exception:
        return []


# ===========================================================================
# Risk metrics
# ===========================================================================
@traceable(run_type="tool", name="compute_risk_metrics")
def compute_risk_metrics(prices: List[float]) -> Dict[str, Any]:
    """Compute volatility, Sharpe ratio, max drawdown and a risk level.

    Operates on a list of daily closing prices. Returns an ``error`` field if
    there is not enough data.
    """
    if not prices or len(prices) < 2:
        return {"error": "insufficient price history for risk analysis"}

    # Daily simple returns.
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1]
    ]
    if not returns:
        return {"error": "could not compute returns"}

    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    daily_std = math.sqrt(variance)

    # Annualise (~252 trading days).
    annual_vol = daily_std * math.sqrt(252)
    annual_return = mean * 252

    # Sharpe ratio assuming a ~2% annual risk-free rate.
    risk_free = 0.02
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol else 0.0

    # Maximum drawdown.
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        peak = max(peak, p)
        if peak:
            max_dd = max(max_dd, (peak - p) / peak)

    if annual_vol < 0.20:
        level = "Low"
    elif annual_vol < 0.40:
        level = "Medium"
    else:
        level = "High"

    return {
        "volatility": round(annual_vol, 4),
        "annual_return": round(annual_return, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "risk_level": level,
        "samples": n,
    }


# ===========================================================================
# News tools
# ===========================================================================
_POSITIVE = {
    "beat", "beats", "surge", "surged", "soar", "soared", "record", "growth",
    "profit", "profits", "upgrade", "upgraded", "bullish", "gain", "gains",
    "rally", "strong", "outperform", "rises", "rose", "jumps", "boost",
}
_NEGATIVE = {
    "miss", "misses", "plunge", "plunged", "drop", "dropped", "loss", "losses",
    "downgrade", "downgraded", "bearish", "fall", "falls", "fell", "weak",
    "decline", "cut", "slump", "lawsuit", "probe", "warning", "fraud",
}


@traceable(run_type="tool", name="score_sentiment")
def score_sentiment(texts: List[str]) -> Dict[str, Any]:
    """Lightweight lexicon-based sentiment over a list of headlines/snippets."""
    pos = neg = 0
    for text in texts:
        words = text.lower().replace(",", " ").replace(".", " ").split()
        pos += sum(w in _POSITIVE for w in words)
        neg += sum(w in _NEGATIVE for w in words)
    total = pos + neg
    if total == 0:
        label = "neutral"
        score = 0.0
    else:
        score = (pos - neg) / total
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
    return {"sentiment": label, "score": round(score, 3), "positive": pos, "negative": neg}


@tool
def search_headlines(query: str, max_results: int = 6) -> dict:
    """Search recent financial news headlines for a company or topic via DuckDuckGo."""
    if DDGS is None:
        return {"query": query, "error": "DuckDuckGo backend (ddgs) not available", "headlines": []}
    try:
        headlines: List[str] = []
        # Cap the network wait so a slow/unreachable backend can't stall the
        # whole pipeline (a major source of first-response latency).
        with DDGS(timeout=8) as ddgs:
            for item in ddgs.news(query, region="us-en", max_results=max_results):
                title = item.get("title")
                if title:
                    headlines.append(title)
        if not headlines:
            with DDGS(timeout=8) as ddgs:
                for item in ddgs.text(query, region="us-en", max_results=max_results):
                    title = item.get("title")
                    if title:
                        headlines.append(title)
        return {"query": query, "headlines": headlines}
    except Exception as exc:
        return {"query": query, "error": f"news search failed: {exc}", "headlines": []}


@tool
def analyze_news_sentiment(query: str) -> dict:
    """Search news for a topic and return headlines plus an overall sentiment label."""
    result = search_headlines.invoke({"query": query})
    headlines = result.get("headlines", [])
    sentiment = score_sentiment(headlines)
    return {**result, **sentiment}


# ===========================================================================
# Purchase tool (human-in-the-loop)
# ===========================================================================
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Place a (simulated) order to buy ``quantity`` shares of ``symbol``.

    HUMAN-IN-THE-LOOP: the graph pauses via ``interrupt`` and waits for an
    explicit human decision ("yes"/"no") before the order is executed.
    """
    decision = interrupt(
        {
            "action": "purchase",
            "symbol": symbol.upper(),
            "quantity": quantity,
            "prompt": f"Approve buying {quantity} shares of {symbol.upper()}? (yes/no)",
        }
    )
    approved = isinstance(decision, str) and decision.strip().lower() in {"yes", "y", "approve"}
    if approved:
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "quantity": quantity,
            "message": f"Purchase order placed for {quantity} shares of {symbol.upper()}.",
        }
    return {
        "status": "cancelled",
        "symbol": symbol.upper(),
        "quantity": quantity,
        "message": f"Purchase of {quantity} shares of {symbol.upper()} was declined.",
    }


# Convenience groupings used by the graph.
STOCK_TOOLS = [get_stock_price, get_stock_fundamentals, compute_risk_metrics]
NEWS_TOOLS = [search_headlines, analyze_news_sentiment, get_fx_rate]
TRADE_TOOLS = [purchase_stock]
ALL_TOOLS = STOCK_TOOLS + NEWS_TOOLS + TRADE_TOOLS
