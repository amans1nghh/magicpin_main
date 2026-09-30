"""Conversation Manager for Vera Merchant Bot.

Tracks multi-turn conversation state, classifies incoming messages
(auto-reply, intent commitment, hostile/opt-out, off-topic, normal),
applies auto-reply escalation logic, and routes to appropriate handlers.
"""

from __future__ import annotations

import re
from typing import Any

from src.context_store import ContextStore
from src.models import ConversationState, MessageClassification, Turn


# ---------------------------------------------------------------------------
# Auto-reply detection patterns (case-insensitive substrings)
# ---------------------------------------------------------------------------

_AUTO_REPLY_PATTERNS: list[str] = [
    "thank you for contacting",
    "our team will respond shortly",
    "we have received your message",
    "automated assistant",
    "will get back to you",
    "auto-reply",
    "automated response",
]

# ---------------------------------------------------------------------------
# Intent commitment phrases (case-insensitive, full-match after strip)
# ---------------------------------------------------------------------------

_INTENT_COMMITMENT_PHRASES: list[str] = [
    "let's do it",
    "yes go ahead",
    "ok what's next",
    "sounds good",
    "let's proceed",
    "yes please",
    "go ahead",
    "i'm in",
    "let's start",
    "ok do it",
    "yes",
    "haan",
    "chalega",
    "chalo",
    "theek hai",
    "kar do",
    "ho jayega",
    "done",
    "ok done",
    "ban jao",
    "sahi hai",
    "bilkul",
    "zaroor",
    "haan ji",
    "ok ji",
    "karo",
    "bhej do",
    "ship it",
    "do it",
    "sure",
    "ok let's go",
    "perfect",
    "agreed",
    "confirm",
    "yes do it",
]

# ---------------------------------------------------------------------------
# Hostile / opt-out phrases (case-insensitive substrings)
# ---------------------------------------------------------------------------

_HOSTILE_OPTOUT_PHRASES: list[str] = [
    "stop messaging",
    "not interested",
    "unsubscribe",
    "leave me alone",
    "stop",
    "don't message",
    "useless",
    "spam",
    "bothering me",
    "stop sending",
]

# ---------------------------------------------------------------------------
# Off-topic keywords (outside Vera's scope)
# ---------------------------------------------------------------------------

_OFF_TOPIC_KEYWORDS: list[str] = [
    "gst",
    "tax filing",
    "income tax",
    "legal advice",
    "court case",
    "lawsuit",
    "passport",
    "visa application",
    "aadhaar",
    "pan card",
    "voter id",
]

# ---------------------------------------------------------------------------
# Valid conversation phases
# ---------------------------------------------------------------------------

VALID_PHASES = frozenset({
    "initiating",
    "qualifying",
    "action_committed",
    "waiting",
    "ended",
})

# ---------------------------------------------------------------------------
# Auto-reply escalation constants
# ---------------------------------------------------------------------------

_AUTO_REPLY_WAIT_SECONDS = 14400  # 4 hours


class ConversationManager:
    """Manages multi-turn conversation state and message classification."""

    def __init__(self, context_store: ContextStore) -> None:
        self._store = context_store
        self._conversations: dict[str, ConversationState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_reply(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: str | None,
        from_role: str,
        message: str,
        received_at: str,
        turn_number: int,
        composer: Any = None,
    ) -> dict:
        """Process an incoming reply and return an action dict.

        Returns one of:
        - {"action": "send", "body": "...", "cta": "...", "rationale": "..."}
        - {"action": "wait", "wait_seconds": N, "rationale": "..."}
        - {"action": "end", "rationale": "..."}
        """
        # 1. Get or create conversation state
        conversation = self._get_or_create_conversation(
            conversation_id, merchant_id, customer_id
        )

        # 2. Append incoming message as a Turn
        turn = Turn(
            role=from_role,
            message=message,
            timestamp=received_at,
            turn_number=turn_number,
        )
        conversation.turns.append(turn)

        # 3. If conversation is already ended, return end action
        if conversation.phase == "ended":
            return {"action": "end", "rationale": "Conversation already ended."}

        # 4. Classify the message
        classification = self.classify_message(message, conversation)

        # 5. Route based on classification
        result = await self._route_classification(
            classification, conversation, composer
        )

        # 6. Track sent_bodies for anti-repetition
        if result.get("action") == "send" and "body" in result:
            conversation.sent_bodies.add(result["body"])

        return result

    def classify_message(
        self, message: str, conversation: ConversationState
    ) -> MessageClassification:
        """Classify an incoming message using pattern matching and heuristics.

        Priority order:
        1. Hostile/opt-out (highest priority — respect merchant boundaries)
        2. Auto-reply detection
        3. Intent commitment
        4. Off-topic
        5. Normal (default)
        """
        msg_lower = message.strip().lower()

        # 1. Hostile / opt-out detection
        if self._is_hostile_optout(msg_lower):
            return MessageClassification.HOSTILE_OPTOUT

        # 2. Auto-reply detection (pattern-based)
        if self._is_auto_reply(msg_lower, conversation):
            return MessageClassification.AUTO_REPLY

        # 3. Intent commitment detection
        if self._is_intent_committed(msg_lower):
            return MessageClassification.INTENT_COMMITTED

        # 4. Off-topic detection
        if self._is_off_topic(msg_lower):
            return MessageClassification.OFF_TOPIC

        # 5. Default: normal
        return MessageClassification.NORMAL

    def get_history(self, conversation_id: str) -> list[Turn]:
        """Return full conversation history for a conversation_id."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return []
        return list(conversation.turns)

    def register_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: str | None,
        trigger_id: str | None,
        initial_body: str,
    ) -> None:
        """Register a conversation created during tick processing.

        Records the initial bot message as the first turn and tracks
        the body for anti-repetition.
        """
        conversation = self._get_or_create_conversation(
            conversation_id, merchant_id, customer_id
        )
        conversation.trigger_id = trigger_id

        # Record the initial bot message
        turn = Turn(
            role="bot",
            message=initial_body,
            timestamp="",
            turn_number=0,
        )
        conversation.turns.append(turn)
        conversation.sent_bodies.add(initial_body)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_hostile_optout(msg_lower: str) -> bool:
        """Check if message contains hostile or opt-out language."""
        for phrase in _HOSTILE_OPTOUT_PHRASES:
            if phrase in msg_lower:
                return True
        return False

    @staticmethod
    def _is_auto_reply(msg_lower: str, conversation: ConversationState) -> bool:
        """Check if message matches auto-reply patterns or is an exact repeat."""
        # Pattern-based detection
        for pattern in _AUTO_REPLY_PATTERNS:
            if pattern in msg_lower:
                return True

        # Exact-match consecutive detection: same text 2+ times from merchant
        merchant_turns = [
            t for t in conversation.turns
            if t.role in ("merchant", "customer")
        ]
        if len(merchant_turns) >= 2:
            current_msg = merchant_turns[-1].message.strip()
            previous_msg = merchant_turns[-2].message.strip()
            if current_msg == previous_msg:
                return True

        return False

    @staticmethod
    def _is_intent_committed(msg_lower: str) -> bool:
        """Check if message contains intent commitment language."""
        for phrase in _INTENT_COMMITMENT_PHRASES:
            if msg_lower == phrase:
                return True
        return False

    @staticmethod
    def _is_off_topic(msg_lower: str) -> bool:
        """Check if message is about topics outside Vera's scope."""
        for keyword in _OFF_TOPIC_KEYWORDS:
            if keyword in msg_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # Routing and handlers
    # ------------------------------------------------------------------

    async def _route_classification(
        self,
        classification: MessageClassification,
        conversation: ConversationState,
        composer: Any,
    ) -> dict:
        """Route to the appropriate handler based on classification."""
        if classification == MessageClassification.AUTO_REPLY:
            return self._handle_auto_reply(conversation)

        if classification == MessageClassification.INTENT_COMMITTED:
            return await self._handle_intent_committed(conversation, composer)

        if classification == MessageClassification.HOSTILE_OPTOUT:
            return self._handle_hostile_optout(conversation)

        if classification == MessageClassification.OFF_TOPIC:
            return self._handle_off_topic(conversation)

        # NORMAL
        return await self._handle_normal(conversation, composer)

    def _handle_auto_reply(self, conversation: ConversationState) -> dict:
        """Apply auto-reply escalation logic (no LLM needed).

        Escalation within the same conversation:
        1st auto-reply with prior bot msg: send brief acknowledgment
        1st auto-reply without prior bot msg: wait 4 hours (cold contact)
        2nd consecutive: wait 4 hours
        3rd+: end conversation
        """
        conversation.auto_reply_streak += 1
        streak = conversation.auto_reply_streak

        if streak == 1:
            # Check if there's any prior bot message in this conversation
            has_prior_bot_msg = any(
                t.role == "bot" for t in conversation.turns
            )

            if not has_prior_bot_msg:
                # Cold auto-reply — no point acknowledging, wait immediately
                conversation.phase = "waiting"
                return {
                    "action": "wait",
                    "wait_seconds": _AUTO_REPLY_WAIT_SECONDS,
                    "rationale": (
                        "Auto-reply detected on first contact — owner not available. "
                        "Waiting 4 hours."
                    ),
                }

            # Has prior bot message — try to reach the owner
            conversation.phase = "qualifying"
            return {
                "action": "send",
                "body": (
                    "Samajh gayi — yeh auto-reply lag raha hai. "
                    "Kya owner/manager directly dekh sakte hain? "
                    "Sirf 2 min ka kaam hai."
                ),
                "cta": "open_ended",
                "rationale": (
                    "First auto-reply detected. Sending brief acknowledgment "
                    "to try reaching the actual owner."
                ),
            }

        if streak == 2:
            conversation.phase = "waiting"
            return {
                "action": "wait",
                "wait_seconds": _AUTO_REPLY_WAIT_SECONDS,
                "rationale": (
                    "Second consecutive auto-reply — owner still unavailable. "
                    "Waiting 4 hours before retry."
                ),
            }

        # 3rd+ consecutive auto-reply — end
        conversation.phase = "ended"
        return {
            "action": "end",
            "rationale": (
                "Auto-reply 3x in a row — no real reply. "
                "Gracefully exiting. Best wishes!"
            ),
        }

    async def _handle_intent_committed(
        self, conversation: ConversationState, composer: Any
    ) -> dict:
        """Handle intent commitment: switch to action mode."""
        conversation.phase = "action_committed"

        # Reset auto-reply streaks
        conversation.auto_reply_streak = 0

        if composer is not None:
            try:
                # Look up merchant, category, trigger, and customer contexts
                merchant_ctx = self._store.get("merchant", conversation.merchant_id)
                category_ctx = None
                trigger_ctx = None
                customer_ctx = None

                if merchant_ctx is not None:
                    cat_slug = merchant_ctx.payload.get("category_slug")
                    if cat_slug:
                        category_ctx = self._store.get("category", cat_slug)

                if conversation.trigger_id:
                    trigger_ctx = self._store.get("trigger", conversation.trigger_id)

                if conversation.customer_id:
                    customer_ctx = self._store.get("customer", conversation.customer_id)

                result = await composer.compose_reply(
                    conversation_history=conversation.turns,
                    classification=MessageClassification.INTENT_COMMITTED,
                    sent_bodies=conversation.sent_bodies,
                    merchant=merchant_ctx.payload if merchant_ctx else None,
                    category=category_ctx.payload if category_ctx else None,
                    trigger=trigger_ctx.payload if trigger_ctx else None,
                    customer=customer_ctx.payload if customer_ctx else None,
                    is_customer_facing=conversation.customer_id is not None,
                )
                return result
            except Exception:
                pass

        # Stub response when no composer available
        return {
            "action": "send",
            "body": "Great — drafting that for you now. Confirm once you've reviewed it.",
            "cta": "binary_confirm_cancel",
            "rationale": "Intent committed — switching to action mode.",
        }

    def _handle_hostile_optout(self, conversation: ConversationState) -> dict:
        """Handle hostile/opt-out: end conversation and suppress."""
        conversation.phase = "ended"
        conversation.auto_reply_streak = 0
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out. Closing conversation.",
        }

    def _handle_off_topic(self, conversation: ConversationState) -> dict:
        """Handle off-topic: politely decline and redirect (no LLM needed)."""
        conversation.auto_reply_streak = 0
        return {
            "action": "send",
            "body": (
                "That's outside what I can help with directly. "
                "Coming back to our earlier topic — "
                "would you like to continue where we left off?"
            ),
            "cta": "open_ended",
            "rationale": (
                "Off-topic request politely declined; "
                "redirecting to original trigger."
            ),
        }

    async def _handle_normal(
        self, conversation: ConversationState, composer: Any
    ) -> dict:
        """Handle normal message: compose reply via LLM if available."""
        conversation.auto_reply_streak = 0

        if composer is not None:
            try:
                # Look up merchant, category, trigger, and customer contexts
                merchant_ctx = self._store.get("merchant", conversation.merchant_id)
                category_ctx = None
                trigger_ctx = None
                customer_ctx = None

                if merchant_ctx is not None:
                    cat_slug = merchant_ctx.payload.get("category_slug")
                    if cat_slug:
                        category_ctx = self._store.get("category", cat_slug)

                if conversation.trigger_id:
                    trigger_ctx = self._store.get("trigger", conversation.trigger_id)

                if conversation.customer_id:
                    customer_ctx = self._store.get("customer", conversation.customer_id)

                result = await composer.compose_reply(
                    conversation_history=conversation.turns,
                    classification=MessageClassification.NORMAL,
                    sent_bodies=conversation.sent_bodies,
                    merchant=merchant_ctx.payload if merchant_ctx else None,
                    category=category_ctx.payload if category_ctx else None,
                    trigger=trigger_ctx.payload if trigger_ctx else None,
                    customer=customer_ctx.payload if customer_ctx else None,
                    is_customer_facing=conversation.customer_id is not None,
                )
                return result
            except Exception:
                pass

        # Stub response when no composer available
        return {
            "action": "send",
            "body": "Thanks for your message! Let me look into that for you.",
            "cta": "open_ended",
            "rationale": "Normal reply — continuing conversation.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: str | None,
    ) -> ConversationState:
        """Retrieve existing conversation or create a new one."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                trigger_id=None,
                phase="initiating",
            )
        return self._conversations[conversation_id]
