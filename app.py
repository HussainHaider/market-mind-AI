"""Gradio frontend for MarketMind AI.

Provides a chat UI with persistent threads (SQLite memory) and an inline
human-in-the-loop approval panel for sensitive actions (stock purchases).
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from marketmind import config
from marketmind.graph import get_app, list_threads

APP = get_app()


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
    history = history + [{"role": "assistant", "content": ""}]
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
        yield history, "", *_approval_update(None)
        return

    history = (history or []) + [{"role": "user", "content": user_msg}]
    yield history, "", *_approval_update(None)

    payload = {"messages": [HumanMessage(content=user_msg)]}
    pending = None
    for history, pending in _stream_synthesis(payload, history, thread_id):
        yield history, "", *_approval_update(pending)


def on_decision(decision: str, history: List[dict], thread_id: str):
    history = history or []
    pending = None
    for history, pending in _stream_synthesis(Command(resume=decision), history, thread_id):
        yield history, *_approval_update(pending)


def on_new_chat(threads: List[str]):
    tid = str(uuid.uuid4())
    threads = (threads or []) + [tid]
    return (
        tid,
        [],
        gr.update(choices=threads, value=tid),
        *_approval_update(None),
    )


def on_load_thread(thread_id: str):
    if not thread_id:
        return [], *_approval_update(None)
    try:
        state = APP.get_state(_config(thread_id))
        history = _messages_to_history(state.values.get("messages", []))
    except Exception:
        history = []
    return history, *_approval_update(_pending_interrupt(thread_id))


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

    with gr.Blocks(title="MarketMind AI", theme=gr.themes.Soft()) as demo:
        thread_state = gr.State(initial_thread)
        threads_state = gr.State(thread_choices)

        gr.Markdown("# MarketMind AI\nA multi-agent financial research assistant.")
        if banner:
            gr.Markdown(banner)

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                new_btn = gr.Button("New chat", variant="primary")
                thread_dd = gr.Dropdown(
                    label="Conversation threads",
                    choices=thread_choices,
                    value=initial_thread,
                    interactive=True,
                )
                gr.Markdown(
                    "**Try:**\n"
                    "- *What's the price and risk for AAPL?*\n"
                    "- *Latest news sentiment on Tesla*\n"
                    "- *Buy 10 shares of MSFT*"
                )

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(type="messages", height=460, label="MarketMind")

                with gr.Group(visible=False) as approval_group:
                    approval_md = gr.Markdown("")
                    with gr.Row():
                        approve_btn = gr.Button("Approve", variant="primary")
                        decline_btn = gr.Button("Decline", variant="stop")

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
            [chatbot, txt, *approval_outputs],
        )
        txt.submit(
            on_submit,
            [txt, chatbot, thread_state],
            [chatbot, txt, *approval_outputs],
        )

        approve_btn.click(
            lambda h, t: on_decision("yes", h, t),
            [chatbot, thread_state],
            [chatbot, *approval_outputs],
        )
        decline_btn.click(
            lambda h, t: on_decision("no", h, t),
            [chatbot, thread_state],
            [chatbot, *approval_outputs],
        )

        new_btn.click(
            on_new_chat,
            [threads_state],
            [thread_state, chatbot, thread_dd, *approval_outputs],
        )
        thread_dd.change(
            lambda tid: (tid, *on_load_thread(tid)),
            [thread_dd],
            [thread_state, chatbot, *approval_outputs],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
