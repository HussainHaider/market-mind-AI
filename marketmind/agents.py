"""Helper functions for the MarketMind multi-agent workflow.

This module provides entity extraction utilities used by the orchestrator
for fallback routing when no LLM is available. The actual agent nodes
are defined in graph.py as part of the multi-agent architecture.
"""

from __future__ import annotations

import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# Entity extraction helpers (used for fallback routing)
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
    """Extract a stock ticker symbol from user input.

    Handles:
    - Company names (e.g., "Apple" -> "AAPL")
    - Cashtags (e.g., "$AAPL")
    - Bare uppercase tokens (e.g., "AAPL")
    """
    lowered = text.lower()
    for name, ticker in _COMPANY_TO_TICKER.items():
        if name in lowered:
            return ticker
    # Cashtags like $AAPL
    cashtag = re.search(r"\$([A-Za-z]{1,5})", text)
    if cashtag:
        return cashtag.group(1).upper()
    # Bare uppercase tokens that look like tickers
    for token in re.findall(r"\b[A-Z]{1,5}\b", text):
        if token not in _STOPWORDS:
            return token
    return None


def extract_quantity(text: str) -> Optional[int]:
    """Extract a share quantity from user input.

    Matches patterns like "10 shares", "5 stocks", or just "10".
    """
    match = re.search(r"\b(\d{1,7})\s*(?:shares?|stocks?|units?)?\b", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Keyword detection for fallback routing
# ---------------------------------------------------------------------------
_BUY_KEYWORDS = ["buy", "purchase", "acquire", "invest in"]
_STOCK_KEYWORDS = [
    "price", "stock", "share", "market cap", "p/e", "pe ratio", "volume",
    "valuation", "fundamental", "risk", "volatility", "fx", "exchange rate",
]
_NEWS_KEYWORDS = ["news", "sentiment", "headline", "analyst", "earnings", "outlook"]
_FINANCE_KEYWORDS = _BUY_KEYWORDS + _STOCK_KEYWORDS + _NEWS_KEYWORDS


def is_financial_query(text: str) -> bool:
    """Check if a query is about markets/finance.

    Used as a cheap gate so greetings and out-of-scope input skip
    unnecessary processing.
    """
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(kw in lowered for kw in _FINANCE_KEYWORDS):
        return True
    return extract_ticker(text) is not None


def get_keyword_routes(text: str) -> List[str]:
    """Determine routing based on keywords (fallback when no LLM)."""
    lowered = text.lower()
    routes: List[str] = []
    if any(w in lowered for w in _BUY_KEYWORDS):
        routes.append("trade")
    if any(w in lowered for w in _STOCK_KEYWORDS):
        routes.append("stock")
    if any(w in lowered for w in _NEWS_KEYWORDS):
        routes.append("news")
    if not routes:
        routes.append("chat" if not extract_ticker(text) else "stock")
    return routes
