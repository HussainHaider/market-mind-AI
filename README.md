# MarketMind AI

**MarketMind AI** is a **true multi-agent** investment-research assistant built with
**LangChain + LangGraph**. It uses an **Orchestrator-Workers** architecture where
specialized sub-agents handle stock analysis, news sentiment, trading, and general
chat — each with their own LLM and tools.

The system features **human-in-the-loop approval** for sensitive actions (trades),
**persistent memory** via SQLite checkpointing, and a **Gradio chat interface**.

---

## Architecture Overview

MarketMind implements a **hierarchical multi-agent system**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Multi-Agent System                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌──────────────────┐                       │
│                      │   Orchestrator   │                       │
│                      │  LLM + dispatch  │                       │
│                      └────────┬─────────┘                       │
│              ┌────────────────┼────────────────┐                │
│              ▼                ▼                ▼                │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│    │ Stock Analyst   │ │ News Analyst    │ │ Trade Agent     │ │
│    │ own LLM + tools │ │ own LLM + tools │ │ LLM + HITL      │ │
│    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘ │
│             │                   │                   │           │
│             ▼                   ▼                   ▼           │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│    │   ToolNode      │ │   ToolNode      │ │   ToolNode      │ │
│    │ price/risk/fund │ │ headlines/sent  │ │ purchase_stock  │ │
│    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘ │
│             │                   │                   │           │
│             └───────────────────┼───────────────────┘           │
│                                 ▼                               │
│                    ┌──────────────────────┐                     │
│                    │ Response Synthesizer │                     │
│                    │   LLM aggregates     │                     │
│                    └──────────┬───────────┘                     │
│                               ▼                                 │
│                    ┌──────────────────────┐                     │
│                    │   Approval Gate +    │                     │
│                    │   SqliteSaver        │                     │
│                    └──────────┬───────────┘                     │
│                               ▼                                 │
│                         END / Gradio UI                         │
└─────────────────────────────────────────────────────────────────┘
```

See [`marketmind_target_architecture.svg`](marketmind_target_architecture.svg) for the detailed visual diagram.

---

## 1. Project Title

**MarketMind AI — A Multi-Agent Financial Research Assistant**

## 2. Selected Use Case

A conversational **financial research analyst** powered by specialized agents. Users can:

- Look up a stock's **price, fundamentals (market cap, P/E, volume) and FX rate**
- Get a quantitative **risk assessment** (volatility, Sharpe ratio, max drawdown)
- Read the latest **news headlines and sentiment** for a company
- **Place a (simulated) buy order** with human-in-the-loop approval

This is a strong multi-agent use case because a single query like "*What's the price, risk
and news on AAPL, and should I buy 10 shares?*" requires **multiple specialized agents
working together** with different tools and capabilities.

## 3. Multi-Agent Architecture

### Sub-Agents

| Agent | LLM | Tools | Output |
|-------|-----|-------|--------|
| **Stock Analyst** | ChatOpenAI | `get_stock_price`, `get_stock_fundamentals`, `compute_risk_metrics` | `StockAnalysisResult` |
| **News Analyst** | ChatOpenAI | `search_headlines`, `analyze_news_sentiment`, `get_fx_rate` | `NewsAnalysisResult` |
| **Trade Executor** | ChatOpenAI | `purchase_stock` (with `interrupt`) | `TradeResult` |
| **Chat Assistant** | ChatOpenAI | (none) | General conversation |

Each sub-agent is a **compiled LangGraph** with its own:
- System prompt defining its specialization
- LLM instance bound to its specific tools
- ReAct-style tool loop (`agent → ToolNode → tools_condition`)

### Orchestrator

The Orchestrator agent:
1. Receives the user's query
2. Decides which sub-agents to invoke (can call multiple)
3. Calls sub-agents as **tools** via `graph.as_tool()`
4. Collects results for the Response Synthesizer

### Response Synthesizer

Aggregates all sub-agent outputs into a coherent, well-formatted response using an LLM.

### Approval Gate

Final checkpoint that can route back to the synthesizer for revisions if requested.

## 4. Tools Used

| Tool | Type | What it does |
|------|------|--------------|
| `get_stock_price` | External API | Latest price (Alpha Vantage → Yahoo Finance fallback) |
| `get_stock_fundamentals` | External API | Market cap, P/E, volume, name (Yahoo Finance) |
| `compute_risk_metrics` | Custom + API | Volatility, Sharpe, drawdown from price history |
| `get_fx_rate` | External API | FX rate between currencies (Yahoo Finance) |
| `search_headlines` | External API | Recent news headlines (DuckDuckGo) |
| `analyze_news_sentiment` | External API + custom | Headlines + lexicon sentiment scoring |
| `purchase_stock` | Custom (HITL) | Simulated order; `interrupt`s for human approval |

## 5. APIs Integrated

- **Alpha Vantage** — real-time stock quotes (when `ALPHA_VANTAGE_API_KEY` is set)
- **Yahoo Finance** (via `yfinance`) — quotes, fundamentals, FX and historical prices
- **DuckDuckGo** (via `ddgs`) — keyless news/web search for headlines
- **OpenAI** — LLM for all agents (Orchestrator, sub-agents, Synthesizer)

All API calls **degrade gracefully**: missing keys or failed requests fall back to
secondary providers or structured error responses.

## 6. LangGraph Workflow

### State

A typed `AgentState` (`marketmind/state.py`) flows through the graph:

- `messages` — conversation history (reduced with `add_messages`)
- `query` — the latest user query
- `ticker`, `quantity` — extracted entities
- `stock_result`, `news_result`, `trade_result` — typed sub-agent outputs
- `aggregated` — combined data for the synthesizer
- `agents_called` — which sub-agents were invoked
- `final_response` — the user-facing answer
- `trade_messages` — private channel for trade tool execution

### Graph Flow

```
START
  → Orchestrator (decides which sub-agents to call)
       ──tools_condition──▶ agent_tools (ToolNode with sub-agent tools)
       ──no tools──────────▶ synthesizer
  (loop back to orchestrator if more tool calls)
  → Response Synthesizer (aggregates all outputs)
  → Approval Gate
       ──approve──▶ END
       ──revise───▶ Response Synthesizer
```

### Key Patterns

1. **Sub-agents as tools**: Each compiled sub-graph is wrapped with `.as_tool()` and
   called by the orchestrator like any other tool.

2. **ReAct loops**: Sub-agents use `tools_condition` to loop between their LLM and
   ToolNode until the task is complete.

3. **HITL interrupt**: The `purchase_stock` tool uses LangGraph's `interrupt()` to
   pause for human approval before executing trades.

## 7. Memory Implementation

Memory uses LangGraph **checkpointing** via `SqliteSaver`:

- Each conversation is a **thread** keyed by `thread_id`
- Full state is checkpointed after every super-step
- Conversations survive process restarts
- Pending approvals can be resumed after app restart

## 8. How to Run

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional)
cp .env.example .env            # then add your API keys

# 4. Run
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | recommended | LLM for all agents (offline fallback if absent) |
| `OPENAI_MODEL` | optional | Defaults to `gpt-4o-mini` |
| `ALPHA_VANTAGE_API_KEY` | optional | Price quotes (falls back to Yahoo Finance) |
| `NEWS_API_KEY` | optional | Reserved for richer news |
| `MARKETMIND_DB` | optional | SQLite memory file (default `marketmind.db`) |
| `LANGSMITH_*` | optional | LangSmith tracing |

## 9. Example Prompts

- `What's the price and risk for AAPL?` → Stock Analyst sub-agent
- `Latest news sentiment on Tesla` → News Analyst sub-agent
- `Price and news for MSFT` → Stock + News sub-agents (multi-agent)
- `Buy 10 shares of MSFT` → Trade Executor (triggers approval panel)
- `Hi, what can you do?` → Chat Assistant sub-agent
- `and what about its news?` → multi-turn follow-up using memory

## 10. Project Layout

| Path | Responsibility |
|------|----------------|
| `marketmind/config.py` | Environment / `.env` configuration |
| `marketmind/state.py` | `AgentState` + typed result schemas |
| `marketmind/tools.py` | Stock, FX, risk, news and purchase tools |
| `marketmind/agents.py` | Entity extraction helpers |
| `marketmind/graph.py` | Multi-agent graph: sub-agents, orchestrator, synthesizer |
| `app.py` | Gradio frontend (chat, threads, approval panel) |

## 11. Challenges & Solutions

- **Sub-agent composition**: Used `graph.as_tool()` to wrap compiled sub-graphs as
  callable tools for the orchestrator.

- **Graceful offline mode**: Every LLM and API path has a deterministic fallback
  (keyword routing, template synthesis, secondary providers).

- **HITL through ToolNode**: The trade sub-agent's `purchase_stock` tool uses
  `interrupt()` which pauses the entire graph for human approval.

- **State isolation**: Each sub-agent uses `SubAgentState` (just messages) while
  the main graph uses the full `AgentState` with typed result schemas.

## 12. Future Improvements

- **Parallel sub-agent execution**: Call multiple sub-agents concurrently
- **RAG**: Upload earnings reports / 10-K PDFs as context
- **Portfolio tracking**: Persist holdings and watch-lists per user
- **Richer news**: NewsAPI/Tavily integration with ML-based sentiment
- **Streaming**: Stream sub-agent outputs as they complete

---

## Notes & Disclaimer

- Trades are **simulated**; no real orders are placed
- Market data depends on third-party providers and may be delayed
- For research/educational purposes only — **not financial advice**
