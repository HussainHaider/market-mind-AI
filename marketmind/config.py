"""Central configuration loaded from the environment / .env file."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value


# ---- LLM ----
OPENAI_API_KEY: str = _get("OPENAI_API_KEY")
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o-mini")

# ---- Data providers ----
ALPHA_VANTAGE_API_KEY: str = _get("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY: str = _get("NEWS_API_KEY")

# ---- Memory ----
MARKETMIND_DB: str = _get("MARKETMIND_DB", "marketmind.db")


def has_openai() -> bool:
    """Whether a usable OpenAI key is configured.

    When no key is present the agent degrades gracefully to deterministic,
    rule-based routing / synthesis so the app remains runnable offline.
    """
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY.lower().startswith("sk-")
