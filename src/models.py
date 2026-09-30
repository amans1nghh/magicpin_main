"""Pydantic request/response models and internal dataclasses for Vera Merchant Bot.

Defines all API contract shapes (request models, response models) and internal
state representations (StoredContext, ConversationState, Turn, ComposedMessage,
MessageClassification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ContextPushRequest(BaseModel):
    """POST /v1/context request body."""

    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    """POST /v1/tick request body."""

    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    """POST /v1/reply request body."""

    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /v1/healthz response body."""

    status: str = "ok"
    uptime_seconds: int
    contexts_loaded: dict[str, int]


class MetadataResponse(BaseModel):
    """GET /v1/metadata response body."""

    team_name: str
    team_members: list[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


class ContextAcceptedResponse(BaseModel):
    """POST /v1/context response when context is accepted."""

    accepted: bool = True
    ack_id: str
    stored_at: str


class ContextRejectedResponse(BaseModel):
    """POST /v1/context response when context is rejected."""

    accepted: bool = False
    reason: str
    current_version: int | None = None


class TickAction(BaseModel):
    """A single action within a tick response."""

    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    send_as: str
    trigger_id: str
    template_name: str
    template_params: list[str]
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    """POST /v1/tick response body."""

    actions: list[TickAction]


class SendAction(BaseModel):
    """Reply action: send a message."""

    action: str = "send"
    body: str
    cta: str | None = None
    rationale: str


class WaitAction(BaseModel):
    """Reply action: wait before next contact."""

    action: str = "wait"
    wait_seconds: int
    rationale: str


class EndAction(BaseModel):
    """Reply action: end the conversation."""

    action: str = "end"
    rationale: str


# ---------------------------------------------------------------------------
# Internal State Models
# ---------------------------------------------------------------------------


class MessageClassification(str, Enum):
    """Classification of an incoming message in a conversation."""

    AUTO_REPLY = "auto_reply"
    INTENT_COMMITTED = "intent_committed"
    HOSTILE_OPTOUT = "hostile_optout"
    OFF_TOPIC = "off_topic"
    NORMAL = "normal"


@dataclass
class StoredContext:
    """A versioned context stored in the Context Store."""

    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: str
    stored_at: str


@dataclass
class PutResult:
    """Result of a context store put operation."""

    accepted: bool
    reason: str | None = None  # "stale_version" | "invalid_scope" | None
    current_version: int | None = None
    ack_id: str | None = None
    stored_at: str | None = None


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str  # "bot" | "merchant" | "customer"
    message: str
    timestamp: str
    turn_number: int


@dataclass
class ConversationState:
    """Tracks the full state of a multi-turn conversation."""

    conversation_id: str
    merchant_id: str
    customer_id: str | None
    trigger_id: str | None
    phase: str  # "initiating" | "qualifying" | "action_committed" | "waiting" | "ended"
    turns: list[Turn] = field(default_factory=list)
    auto_reply_streak: int = 0
    sent_bodies: set[str] = field(default_factory=set)
    created_at: str = ""


@dataclass
class ComposedMessage:
    """Output of the Composer engine."""

    body: str
    cta: str
    send_as: str  # "vera" | "merchant_on_behalf"
    suppression_key: str
    rationale: str
    template_name: str
    template_params: list[str]
