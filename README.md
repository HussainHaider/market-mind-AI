# MarketMind AI

**MarketMind AI** is a multi-agent investment-research assistant built with
**LangChain + LangGraph**. It evaluates securities, monitors market sentiment,
analyses fundamentals and assesses financial risk — with a **human-in-the-loop
approval** step before any sensitive action such as placing a trade.

It is powered by a **LangGraph** state machine with **conditional routing** and a
prebuilt **`ToolNode`**, an **OpenAI** LLM (with a full offline fallback), a
**Gradio** chat frontend, **DuckDuckGo / Alpha Vantage / Yahoo Finance** as
external API tools, and a **SQLite-backed checkpointer** for persistent,
multi-turn conversation memory.

> Built for the *LangChain + LangGraph — Agentic AI Application* assignment. The
> section headings below map 1:1 to the required README deliverables.

---

## 1. Project title

**MarketMind AI — A Multi-Agent Financial Research Assistant.**

## 2. Selected use case

A conversational **financial research analyst**. A user can, in plain English:

- Look up a stock's **price, fundamentals (market cap, P/E, volume) and FX rate**.
- Get a quantitative **risk assessment** (volatility, Sharpe ratio, max drawdown).
- Read the latest **news headlines and a sentiment read** for a company.
- **Place a (simulated) buy order**, which always pauses for explicit human
  approval before "executing".

It is a good agentic use case because a single request ("*What's the price, risk
and news on AAPL, and should I buy 10 shares?*") legitimately requires **multiple
tools, multiple external APIs, dynamic routing and a human approval gate** — the
exact capabilities the assignment asks for.

## 3. Tools used

| Tool | Type | What it does |
|------|------|--------------|
| `get_stock_price` | External API | Latest price (Alpha Vantage → Yahoo Finance fallback) |
| `get_stock_fundamentals` | External API | Market cap, P/E, volume, name (Yahoo Finance) |
| `get_fx_rate` | External API | FX rate between two currencies (Yahoo Finance) |
| `search_headlines` | External API | Recent news headlines (DuckDuckGo) |
| `analyze_news_sentiment` | External API + custom | Headlines **+** lexicon sentiment scoring |
| `compute_risk_metrics` | **Custom Python** | Volatility, Sharpe, drawdown, risk level |
| `score_sentiment` | **Custom Python** | Lexicon-based sentiment over text |
| `purchase_stock` | **Custom Python** (HITL) | Simulated order; `interrupt`s for approval |

This satisfies the requirement of **≥3 tools, ≥2 external-API tools and ≥1 custom
Python tool**.

## 4. APIs integrated

- **Alpha Vantage** — real-time stock quotes (`GLOBAL_QUOTE`); used when
  `ALPHA_VANTAGE_API_KEY` is set.
- **Yahoo Finance** (via `yfinance`) — quotes, fundamentals, FX and historical
  prices; also the automatic fallback for Alpha Vantage.
- **DuckDuckGo** (via `ddgs`) — keyless news / web search for headlines.
- **OpenAI** — LLM used by the Supervisor (routing/entity extraction) and the
  Response Synthesizer (final answer).

Every API call **degrades gracefully**: a missing key, a failed request or an
offline environment falls back to a secondary provider or a structured `error`
field rather than crashing.

## 5. LangGraph workflow explanation

### State

A single typed `AgentState` (`marketmind/state.py`) flows through every node. Key
channels:

- `messages` — the user-facing conversation (reduced with `add_messages`).
- `trade_messages` — a **private** channel that drives the trade `ToolNode`, so
  raw tool-call/tool-result messages never pollute the chat transcript.
- `query`, `routes`, `ticker`, `quantity` — the Supervisor's intent + entities.
- `stock_data`, `risk_score`, `news_results`, `purchase` — per-flow outputs.
- `aggregated`, `tool_history`, `final_response`, `needs_approval`.

### Nodes

| Node | Role |
|------|------|
| `supervisor` | Classifies intent into one or more routes + extracts ticker/quantity |
| `stock_fetcher` → `risk_calculator` | Stock Analysis Flow (price/fundamentals → risk) |
| `news_agent` | News Analysis Flow (headlines + sentiment) |
| `trader` | Emits a `purchase_stock` **tool call** |
| `trade_tools` | **Prebuilt `ToolNode`** — executes the tool (HITL `interrupt`) |
| `trade_collect` | Lifts the `ToolMessage` result back into typed state |
| `processing` → `state_updater` | Aggregate flow outputs + dedupe tool history |
| `response_synthesizer` | LLM (or template fallback) writes the final answer |
| `approval_gate` | Final review checkpoint (`approve` → END, `revise` → regenerate) |

### Edges, conditional routing & tool flow

The graph is **not** a fixed linear pipeline — the Supervisor's decision drives
**conditional edges** that *skip* straight to whichever workflows are actually
needed (a deterministic `stocks → news → buy → processing` skip-chain):

```text
START
  → supervisor
       ──cond──▶ stock_fetcher → risk_calculator        (route: "stocks")
       ──cond──▶ news_agent                             (route: "news")
       ──cond──▶ trader → trade_tools(ToolNode) → trade_collect   (route: "buy")
       ──cond──▶ response_synthesizer                   (route: "chat" — no tools)
  (active flows converge) → processing → state_updater
       → response_synthesizer
       → approval_gate ──approve──▶ END
                        ──revise───▶ response_synthesizer
```

This demonstrates the full required cycle: **user input → tool selection (the
Supervisor) → conditional routing → tool execution (`ToolNode` / flow nodes) →
results returned to state → final response generation.**

- A greeting like *"hi"* is classified as `chat` and skips **every** tool/API.
- *"news on Tesla"* skips the stock, risk and trade nodes entirely.
- *"buy 10 MSFT"* routes through the canonical **agent → `ToolNode`** loop, which
  pauses for human approval.

You can see exactly which path was taken for each turn in the **"Agent activity"**
panel in the UI (intent, ticker, tools/APIs called, approval status).

## 6. Memory implementation

Memory is implemented with LangGraph **checkpointing** via `SqliteSaver`
(`marketmind/graph.py`). Each conversation is a **thread** keyed by `thread_id`.

### Graph state vs. memory

- **Graph state** (`AgentState`) is the *working memory for a single run* — the
  data passed between nodes as one request is processed. It is transient to that
  invocation.
- **Memory (checkpointing)** is the *durable persistence of that state across
  runs*. After **every** super-step the checkpointer writes the full state to
  SQLite under the thread's id, so the next turn resumes with the complete prior
  state instead of starting blank.

### How checkpoints persist conversations

Because the entire state (including `messages`) is snapshotted per step:

- Multi-turn context survives — turn 2 sees everything from turn 1.
- A conversation **survives a process restart** (it's on disk, not in RAM).
- A run **paused on an `interrupt`** (a pending purchase approval) is itself
  checkpointed, so you can close the app, reopen it, pick the thread from the
  sidebar and still approve/decline the waiting order.

### How memory improves UX

Users get natural follow-ups (*"and the risk?"*, *"now buy 5"*) without repeating
context, can juggle multiple independent research threads in the sidebar, and
never lose an in-progress approval.

## 7. How to run the application

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — the app runs offline without keys)
cp .env.example .env            # then edit .env and add your keys

# 4. Run
python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | recommended | LLM routing & synthesis (offline fallback if absent) |
| `OPENAI_MODEL` | optional | Defaults to `gpt-4o-mini` |
| `ALPHA_VANTAGE_API_KEY` | optional | Price quotes (falls back to Yahoo Finance) |
| `NEWS_API_KEY` | optional | Reserved for richer news (DuckDuckGo needs no key) |
| `MARKETMIND_DB` | optional | SQLite memory file (default `marketmind.db`) |
| `LANGSMITH_*` | optional | LangSmith tracing (see `.env.example`) |

## 8. Example prompts

- `What's the price and risk for AAPL?` → stock + risk flow.
- `Latest news sentiment on Tesla` → news flow only (stock/trade skipped).
- `Price and news for MSFT` → stock + risk + news flows.
- `Buy 10 shares of MSFT` → triggers the **Approve / Decline** panel (HITL).
- `and what about its news?` → multi-turn follow-up using thread memory.
- `hi, what can you do?` → `chat` route, no tools called.

## 9. Challenges faced

- **Fan-in races with parallel branches.** An early design fanned out to all
  flow nodes in parallel, but uneven branch lengths (stocks is 2 nodes, news is
  1) caused the converging `processing` node to fire early/twice. Solved with a
  **deterministic conditional skip-chain** — genuine conditional routing that
  still skips unneeded nodes, but with a single, predictable convergence point.
- **Human-in-the-loop through a `ToolNode`.** Combining LangGraph's `interrupt`
  with the prebuilt `ToolNode` required driving the tool call over a **dedicated
  `trade_messages` channel** so the internal tool-call/result messages never
  leaked into the user-facing chat or the token stream.
- **Streaming the right tokens.** Gradio streaming used `stream_mode="messages"`
  and had to be filtered to the `response_synthesizer` node so the Supervisor's
  internal routing tokens never appeared in the chat.
- **Graceful offline mode.** Every LLM and API path needed a deterministic
  fallback (keyword routing, template synthesis, secondary data provider) so the
  app remains fully runnable and demoable without any API keys.
- **Runtime theme switching in Gradio**, which bakes themes at build time —
  worked around by injecting each theme's CSS variables via a `<style>` block.

## 10. Future improvements

- A full **ReAct tool-calling loop** (`bind_tools` + `tools_condition`) for the
  stock/news flows so the LLM, not the supervisor, chooses tools turn-by-turn.
- **RAG** over uploaded earnings reports / 10-K PDFs as an additional tool.
- Portfolio-level analytics and watch-lists persisted per user.
- **Deployment to Hugging Face Spaces** with a hosted SQLite/Postgres backend.
- Richer news via a dedicated provider (NewsAPI/Tavily) and ML-based sentiment.
- Automated tests + CI around the routing and HITL logic.

---

## Architecture

See [`financial_research_agent_langgraph.svg`](financial_research_agent_langgraph.svg)
for the full workflow diagram (User → Gradio → LangGraph state/nodes/ToolNode →
APIs → SQLite memory → final response).

### Project layout

| Path | Responsibility |
|------|----------------|
| `marketmind/config.py` | Environment / `.env` configuration |
| `marketmind/state.py` | `AgentState` shared graph state (+ private trade channel) |
| `marketmind/tools.py` | Stock, FX, risk, news and purchase tools |
| `marketmind/agents.py` | Graph nodes + conditional-routing functions |
| `marketmind/graph.py` | Graph wiring, `ToolNode`, SQLite checkpointer |
| `app.py` | Gradio frontend (chat, threads, approval panel, activity panel) |

---

## Bonus features implemented

- **Multi-agent + supervisor workflow** — a Supervisor routes to specialised
  stock / news / trade sub-flows.
- **Human-in-the-loop approval** — `interrupt`-based gate before any trade.
- **Database integration** — SQLite-backed checkpointer for thread memory.
- **LangSmith tracing** — `@traceable` helpers + env config for full run
  observability.
- **Streaming responses** — token-level streaming into the Gradio chat.

---

## Notes & disclaimer

- Trades are **simulated**; no real orders are placed.
- Market data depends on third-party providers and may be delayed.
- For research/educational purposes only — **not financial advice**.
