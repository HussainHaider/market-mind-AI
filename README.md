# MarketMind AI

**MarketMind AI** is a multi-agent investment research assistant that helps you
evaluate securities, monitor market sentiment, analyze fundamentals, and assess
financial risk — with a human-in-the-loop approval step before any sensitive
action such as placing a trade.

It is built with **LangGraph** + **OpenAI**, a **Gradio** chat frontend,
**DuckDuckGo** for news search, and **SQLite-backed thread memory** for
persistent, multi-turn conversations.

---

## Features

- **Supervisor routing** — classifies each request into one or more workflows
  (`stocks`, `news`, `buy`) and extracts the ticker / quantity.
- **Stock analysis** — price, market cap, P/E ratio, volume and FX rates
  (Alpha Vantage with a Yahoo Finance fallback).
- **Risk metrics** — annualized volatility, Sharpe ratio, maximum drawdown and
  an overall risk level computed from historical prices.
- **News & sentiment** — recent headlines via DuckDuckGo with a lexicon-based
  sentiment score.
- **Human-in-the-loop purchases** — the graph pauses (`interrupt`) and waits for
  explicit approval before simulating a trade.
- **Persistent memory** — every conversation is a thread checkpointed to SQLite,
  so context survives restarts.
- **Graceful offline mode** — with no `OPENAI_API_KEY`, routing falls back to
  keyword heuristics and responses to a structured template, so the app still
  runs end-to-end.

---

## Architecture

```text
START
  -> Supervisor Agent            # intent detection + entity extraction + routing
  -> Stock Fetcher -> Risk Calc  # Stock Analysis Flow   (gated on route "stocks")
  -> News Agent                  # News Analysis Flow    (gated on route "news")
  -> Purchase Agent              # Purchase Flow w/ HITL  (gated on route "buy")
  -> Processing Layer            # aggregate all flow outputs
  -> State Updater               # persist + dedupe tool history
  -> Response Synthesizer        # final user-facing answer
  -> Human Approval Gate
        --(approve)--> END
        --(revise)---> Response Synthesizer
```

The pipeline is a **deterministic linear flow**: the Supervisor populates
`state["routes"]`, and each flow node is *gated* on those routes (passing through
untouched when its workflow was not selected). This avoids the fan-in races that
can occur with uneven parallel branches while still honouring the router concept.

### Project layout

| Path | Responsibility |
|------|----------------|
| `marketmind/config.py` | Environment / `.env` configuration |
| `marketmind/state.py` | `AgentState` shared graph state |
| `marketmind/tools.py` | Stock, FX, risk, news and purchase tools |
| `marketmind/agents.py` | Graph nodes (supervisor, flows, synthesizer, gate) |
| `marketmind/graph.py` | Graph wiring + SQLite checkpointer |
| `app.py` | Gradio frontend |

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env   # then edit .env and add your keys
```

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | recommended | LLM routing & synthesis (offline fallback if absent) |
| `OPENAI_MODEL` | optional | Defaults to `gpt-4o-mini` |
| `ALPHA_VANTAGE_API_KEY` | optional | Price quotes (falls back to Yahoo Finance) |
| `NEWS_API_KEY` | optional | Reserved for richer news (DuckDuckGo needs no key) |
| `MARKETMIND_DB` | optional | SQLite memory file (default `marketmind.db`) |

---

## Run

```bash
python app.py
```

Then open the local URL printed by Gradio (usually `http://127.0.0.1:7860`).

### Example prompts

- `What's the price and risk for AAPL?`
- `Latest news sentiment on Tesla`
- `Buy 10 shares of MSFT` → triggers the **Approve / Decline** panel

---

## How memory works

Each conversation is identified by a `thread_id`. LangGraph's `SqliteSaver`
checkpoints the full state after every step, so you can switch between threads in
the sidebar and resume exactly where you left off — including a purchase that is
still waiting for approval.

---

## Notes & disclaimer

- Trades are **simulated**; no real orders are placed.
- Market data depends on third-party providers and may be delayed.
- This project is for research/educational purposes and is **not financial
  advice**.
