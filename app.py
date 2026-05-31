"""Gradio frontend for MarketMind AI.

Provides a chat UI with persistent threads (SQLite memory) and an inline
human-in-the-loop approval panel for sensitive actions (stock purchases).
"""

from __future__ import annotations

import uuid
import warnings
from typing import List, Optional, Tuple

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from marketmind import config
from marketmind.graph import get_app, list_threads

# Gradio 5.50 emits forward-looking 6.0 deprecation notices that we cannot avoid
# while keeping the current, intended behaviour:
#   * ``theme`` must still be passed to the Blocks constructor (``launch(theme=...)``
#     is not supported yet in 5.50).
#   * ``allow_tags`` warns whenever it is not ``True`` — even when explicitly set to
#     ``False`` — but we deliberately keep HTML tags disabled.
for _msg in (
    r".*'theme' parameter in the Blocks constructor.*",
    r".*default value of 'allow_tags' in gr\.Chatbot.*",
):
    warnings.filterwarnings("ignore", message=_msg, category=DeprecationWarning)

APP = get_app()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; }

/* Sidebar conversation list should span the full column width */
#thread-list, #thread-list .wrap { width: 100% !important; }
#thread-list .wrap label {
    display: flex !important;
    width: 100% !important;
    box-sizing: border-box;
}

/* Chat column is a vertical flex so the composer hugs the bottom */
#chat-col { display: flex; flex-direction: column; }

/* Header: title on the left, theme picker pinned to the top-right */
#header-row { flex-wrap: nowrap !important; align-items: flex-start; }
#app-title { flex: 1 1 auto; min-width: 0 !important; }
#theme-dd {
    flex: 0 0 auto !important;
    margin-left: auto !important;
}
"""


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
# Gradio bakes the theme at ``Blocks`` build time, so true runtime switching is
# done by overriding the theme's CSS variables with a later ``<style>`` block.
# We build each theme once, precompute its variable CSS, and inject every theme's
# font stylesheets up front so a switched-to theme still renders with its font.
DEFAULT_THEME_NAME = "Default"
THEMES = {
    "Terminal": gr.Theme.from_hub("hmb/terminal"),
    "Monochrome": gr.themes.Monochrome(),
    "Default": gr.themes.Default(),
}
THEME_CSS = {name: theme._get_theme_css() for name, theme in THEMES.items()}


def _font_links_html() -> str:
    """Collect the external font stylesheets across all themes (deduped)."""
    seen: set = set()
    links: List[str] = []
    for theme in THEMES.values():
        for sheet in getattr(theme, "_stylesheets", []) or []:
            if not sheet or sheet in seen:
                continue
            seen.add(sheet)
            href = f"https:{sheet}" if sheet.startswith("//") else sheet
            links.append(f'<link rel="stylesheet" href="{href}">')
    return "\n".join(links)


def _theme_style(name: str) -> str:
    return f"<style>{THEME_CSS.get(name, '')}</style>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "run_name": "chat_turn"}


def _pending_interrupt(thread_id: str) -> Optional[dict]:
    """Return the interrupt payload if the graph is paused, else None."""
    try:
        state = APP.get_state(_config(thread_id))
    except Exception:
        return None
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _thread_choices(threads: List[str]) -> List[Tuple[str, str]]:
    """Build (label, value) pairs so the thread list shows short, readable names.

    Newest conversation first (creation-order numbering is preserved).
    """
    labels: List[Tuple[str, str]] = []
    for i, tid in enumerate(threads, start=1):
        labels.append((f"Chat {i} · {str(tid)[:8]}…", tid))
    return list(reversed(labels))


def _messages_to_history(messages) -> List[dict]:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
    return history


def _latest_answer(thread_id: str) -> str:
    try:
        state = APP.get_state(_config(thread_id))
    except Exception:
        return ""
    for msg in reversed(state.values.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _activity_md(thread_id: str) -> str:
    """Summarise the agent's reasoning + tool/API usage for the latest turn.

    Surfaces the Supervisor's routing decision, the extracted ticker and the
    tools/APIs actually executed — so the UI visibly demonstrates *why* the
    agent did what it did and *which* tools ran (assignment Steps 5 & 6).
    """
    try:
        vals = APP.get_state(_config(thread_id)).values
    except Exception:
        return "_No activity yet — ask a question to see the agent's routing and tool usage._"

    routes = vals.get("routes") or []
    if not routes or routes == ["chat"]:
        return "**Detected intent:** `chat` — handled directly, _no tools or external APIs were called for this turn._"

    lines = ["**Detected intent:** " + ", ".join(f"`{r}`" for r in routes)]
    if vals.get("ticker"):
        lines.append(f"**Ticker:** `{vals['ticker']}`")
    if vals.get("quantity"):
        lines.append(f"**Quantity:** `{vals['quantity']}`")
    tools_used = vals.get("tool_history") or []
    if tools_used:
        lines.append("**Tools / APIs called:** " + ", ".join(f"`{t}`" for t in tools_used))
    if _pending_interrupt(thread_id):
        lines.append("**Status:** ⏸ paused — awaiting human approval")
    return "\n\n".join(lines)


def _approval_update(payload: Optional[dict]):
    """Build component updates for the approval panel."""
    if payload:
        prompt = payload.get("prompt", "Approval required") if isinstance(payload, dict) else str(payload)
        return (
            gr.update(visible=True),
            gr.update(value=f"### Human approval required\n\n{prompt}"),
            gr.update(interactive=False, placeholder="Resolve the approval above to continue…"),
        )
    return (
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(interactive=True, placeholder="Ask about a stock, news sentiment, or risk…"),
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def _stream_synthesis(payload, history: List[dict], thread_id: str):
    """Run the graph and stream the Response Synthesizer's tokens.

    Yields ``(history, pending_interrupt)`` updates. Only chunks emitted by the
    ``response_synthesizer`` node are surfaced, so the supervisor's internal
    routing tokens never leak into the chat. Mirrors the ``stream_mode="messages"``
    approach used in the omnichat Streamlit frontend.
    """
    # Show an immediate placeholder so the user gets feedback while the LLM /
    # tools run, before the first streamed token arrives.
    history = history + [{"role": "assistant", "content": "_Thinking…_"}]
    yield history, None

    acc = ""
    streamed = False
    try:
        for chunk, meta in APP.stream(payload, config=_config(thread_id), stream_mode="messages"):
            if meta.get("langgraph_node") != "response_synthesizer":
                continue
            if not isinstance(chunk, AIMessage):
                continue
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue
            acc += text
            streamed = True
            history[-1]["content"] = acc
            yield history, None
    except Exception as exc:
        history[-1]["content"] = f"Sorry, something went wrong: {exc}"
        yield history, None
        return

    # Finalize: prefer the persisted answer; drop the empty bubble if the graph
    # paused (e.g. waiting for purchase approval) or produced nothing.
    pending = _pending_interrupt(thread_id)
    answer = _latest_answer(thread_id)
    if answer:
        history[-1]["content"] = answer
    elif not streamed:
        history.pop()
    yield history, pending


def on_submit(user_msg: str, history: List[dict], thread_id: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield history, "", *_approval_update(None), gr.update()
        return

    history = (history or []) + [{"role": "user", "content": user_msg}]
    yield history, "", *_approval_update(None), gr.update(value="_Working…_")

    payload = {"messages": [HumanMessage(content=user_msg)]}
    pending = None
    for history, pending in _stream_synthesis(payload, history, thread_id):
        yield history, "", *_approval_update(pending), _activity_md(thread_id)


def on_decision(decision: str, history: List[dict], thread_id: str):
    history = history or []
    pending = None
    for history, pending in _stream_synthesis(Command(resume=decision), history, thread_id):
        yield history, *_approval_update(pending), _activity_md(thread_id)


def on_approve(history: List[dict], thread_id: str):
    yield from on_decision("yes", history, thread_id)


def on_decline(history: List[dict], thread_id: str):
    yield from on_decision("no", history, thread_id)


def on_new_chat(threads: List[str]):
    tid = str(uuid.uuid4())
    threads = (threads or []) + [tid]
    return (
        tid,
        threads,
        [],
        gr.update(choices=_thread_choices(threads), value=tid),
        *_approval_update(None),
        _activity_md(tid),
    )


def on_load_thread(thread_id: str):
    if not thread_id:
        return [], *_approval_update(None), _activity_md("")
    try:
        state = APP.get_state(_config(thread_id))
        history = _messages_to_history(state.values.get("messages", []))
    except Exception:
        history = []
    return history, *_approval_update(_pending_interrupt(thread_id)), _activity_md(thread_id)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    initial_thread = str(uuid.uuid4())
    existing = list_threads()
    thread_choices = existing + ([initial_thread] if initial_thread not in existing else [])

    banner = (
        "" if config.has_openai()
        else "**Note:** No `OPENAI_API_KEY` detected — running in offline mode "
        "(keyword routing + template responses). Set the key in `.env` for full LLM reasoning."
    )

    with gr.Blocks(
        title="MarketMind AI",
        theme=THEMES[DEFAULT_THEME_NAME],
        fill_height=True,
    ) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>", padding=False)
        gr.HTML(_font_links_html(), padding=False)
        theme_style = gr.HTML(_theme_style(DEFAULT_THEME_NAME), padding=False)
        thread_state = gr.State(initial_thread)
        threads_state = gr.State(thread_choices)
        tips_visible = gr.State(False)

        with gr.Row(equal_height=False, elem_id="header-row"):
            gr.Markdown(
                "# MarketMind AI\nA multi-agent financial research assistant.",
                elem_id="app-title",
            )
            theme_dd = gr.Dropdown(
                label="Theme",
                choices=list(THEMES.keys()),
                value=DEFAULT_THEME_NAME,
                interactive=True,
                min_width=160,
                elem_id="theme-dd",
            )
        if banner:
            gr.Markdown(banner)

        tips_text = (
            "**Try asking:**\n"
            "- *What's the price and risk for AAPL?*\n"
            "- *Latest news sentiment on Tesla*\n"
            "- *Buy 10 shares of MSFT*"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                with gr.Row():
                    new_btn = gr.Button("New chat", variant="primary", scale=5)
                    info_btn = gr.Button("ⓘ", scale=1, min_width=44)
                tips_md = gr.Markdown(tips_text, visible=False)
                thread_list = gr.Radio(
                    label="Conversation threads",
                    choices=_thread_choices(thread_choices),
                    value=initial_thread,
                    interactive=True,
                    elem_id="thread-list",
                )

            with gr.Column(scale=4, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    type="messages",
                    label="MarketMind",
                    allow_tags=False,
                    scale=1,
                    height="70vh",
                )

                with gr.Group(visible=False) as approval_group:
                    approval_md = gr.Markdown("")
                    with gr.Row():
                        approve_btn = gr.Button("Approve", variant="primary")
                        decline_btn = gr.Button("Decline", variant="stop")

                with gr.Accordion("Agent activity (routing & tools)", open=False):
                    activity_md = gr.Markdown(
                        "_No activity yet — ask a question to see the agent's "
                        "routing and tool usage._"
                    )

                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Ask about a stock, news sentiment, or risk…",
                        scale=8,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        approval_outputs = [approval_group, approval_md, txt]

        send_btn.click(
            on_submit,
            [txt, chatbot, thread_state],
            [chatbot, txt, *approval_outputs, activity_md],
        )
        txt.submit(
            on_submit,
            [txt, chatbot, thread_state],
            [chatbot, txt, *approval_outputs, activity_md],
        )

        approve_btn.click(
            on_approve,
            [chatbot, thread_state],
            [chatbot, *approval_outputs, activity_md],
        )
        decline_btn.click(
            on_decline,
            [chatbot, thread_state],
            [chatbot, *approval_outputs, activity_md],
        )

        new_btn.click(
            on_new_chat,
            [threads_state],
            [thread_state, threads_state, chatbot, thread_list, *approval_outputs, activity_md],
        )
        thread_list.change(
            lambda tid: (tid, *on_load_thread(tid)),
            [thread_list],
            [thread_state, chatbot, *approval_outputs, activity_md],
        )

        info_btn.click(
            lambda visible: (gr.update(visible=not visible), not visible),
            [tips_visible],
            [tips_md, tips_visible],
        )

        theme_dd.change(_theme_style, [theme_dd], [theme_style])

    return demo


if __name__ == "__main__":
    build_ui().launch()
