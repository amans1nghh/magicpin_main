"""Configuration for Vera Merchant Bot.

Centralizes team metadata, LLM provider settings, and timeout budgets.
All LLM settings are loaded from the .env file (or environment variables).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env file automatically (no extra dependency needed)
# ---------------------------------------------------------------------------

def _load_dotenv(path: str = ".env") -> None:
    """Read a .env file and set any missing environment variables."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if not already in the environment (env vars take precedence)
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


# ---------------------------------------------------------------------------
# Team metadata (returned by GET /v1/metadata)
# ---------------------------------------------------------------------------

TEAM_NAME = "Vera"
TEAM_MEMBERS = ["Shubham Krishna"]
CONTACT_EMAIL = "krishnashubham09@gmail.com"
VERSION = "0.1.0"
APPROACH = (
    "4-context LLM composition framework with trigger-kind prompt dispatch, "
    "anti-pattern validation, auto-reply escalation, intent detection, "
    "and compulsion-lever injection optimized for the 5-dimension scoring rubric."
)
SUBMITTED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings. Reads from env vars with sensible defaults."""

    provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("LLM_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o")
    )
    timeout: float = 25.0  # seconds — per-call budget for LLM requests


def get_llm_config() -> LLMConfig:
    """Return an LLMConfig populated from environment variables."""
    return LLMConfig()


# ---------------------------------------------------------------------------
# Timeout budgets (seconds) — per the judge harness contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeoutBudgets:
    """Maximum allowed response times for each endpoint."""

    healthz: float = 2.0
    metadata: float = 2.0
    context: float = 5.0
    tick: float = 10.0
    reply: float = 10.0


TIMEOUT_BUDGETS = TimeoutBudgets()


# ---------------------------------------------------------------------------
# Valid context scopes
# ---------------------------------------------------------------------------

VALID_SCOPES = frozenset({"category", "merchant", "customer", "trigger"})
