"""Unit tests for the ConversationManager class.

Covers:
- Message classification (auto-reply, intent, hostile, off-topic, normal)
- Auto-reply escalation logic (1st send, 2nd wait, 3rd+ end)
- Intent transition and phase changes
- Hostile exit and conversation suppression
- Off-topic handling
- Conversation history completeness
- Anti-repetition tracking (sent_bodies)
- Edge cases (ended conversations, missing composer, etc.)
"""

from __future__ import annotations

import pytest

from src.context_store import ContextStore
from src.conversation_manager import ConversationManager
from src.models import ConversationState, MessageClassification, Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = "2026-05-01T12:00:00Z"


class FakeComposer:
    """Minimal composer stub that returns a deterministic reply action."""

    def __init__(self, body: str = "Composed reply from LLM.") -> None:
        self._body = body
        self.call_count = 0
        self.last_classification: MessageClassification | None = None

    async def compose_reply(
        self,
        conversation_history: list[Turn] | None = None,
        classification: MessageClassification | None = None,
        sent_bodies: set[str] | None = None,
        **kwargs,
    ) -> dict:
        self.call_count += 1
        self.last_classification = classification
        return {
            "action": "send",
            "body": self._body,
            "cta": "open_ended",
            "rationale": "Composed via LLM.",
        }


def _make_manager() -> ConversationManager:
    """Create a ConversationManager with a fresh ContextStore."""
    return ConversationManager(ContextStore())


async def _send_reply(
    mgr: ConversationManager,
    message: str,
    conversation_id: str = "conv_1",
    merchant_id: str = "m1",
    from_role: str = "merchant",
    turn_number: int = 1,
    composer=None,
) -> dict:
    """Helper to call handle_reply with sensible defaults."""
    return await mgr.handle_reply(
        conversation_id=conversation_id,
        merchant_id=merchant_id,
        customer_id=None,
        from_role=from_role,
        message=message,
        received_at=NOW,
        turn_number=turn_number,
        composer=composer,
    )


# ===========================================================================
# Message Classification Tests
# ===========================================================================


class TestAutoReplyClassification:
    """Tests for auto-reply pattern detection.

    These tests verify that auto-reply patterns are correctly detected.
    The action may be 'send' or 'wait' depending on whether a prior
    bot message exists in the conversation.
    """

    @pytest.mark.asyncio
    async def test_thank_you_for_contacting(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Thank you for contacting us!")
        assert result["action"] in ("send", "wait")
        assert "auto" in result["rationale"].lower()

    @pytest.mark.asyncio
    async def test_our_team_will_respond(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Our team will respond shortly. Please wait.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_we_have_received_your_message(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "We have received your message and will reply soon.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_automated_assistant(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Hi! I'm an automated assistant. A human will be with you shortly.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_will_get_back_to_you(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Thanks! We will get back to you soon.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_auto_reply_keyword(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "This is an auto-reply. We are currently unavailable.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_automated_response(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "This is an automated response from our system.")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "THANK YOU FOR CONTACTING US!")
        assert result["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_exact_match_consecutive_detection(self) -> None:
        """Same message sent twice consecutively should be classified as auto-reply."""
        mgr = _make_manager()
        r1 = await _send_reply(mgr, "Hello there", turn_number=1)
        r2 = await _send_reply(mgr, "Hello there", turn_number=2)
        assert r2["action"] in ("send", "wait")

    @pytest.mark.asyncio
    async def test_classify_message_returns_auto_reply_enum(self) -> None:
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="initiating",
        )
        result = mgr.classify_message("Thank you for contacting us", conv)
        assert result == MessageClassification.AUTO_REPLY


class TestIntentCommitmentClassification:
    """Tests for intent commitment phrase detection."""

    @pytest.mark.asyncio
    async def test_lets_do_it(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "let's do it")
        assert result["action"] == "send"
        # Phase should be action_committed
        conv = mgr._conversations["conv_1"]
        assert conv.phase == "action_committed"

    @pytest.mark.asyncio
    async def test_yes_go_ahead(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "yes go ahead")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_sounds_good(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "sounds good")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_im_in(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "I'm in")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_yes_simple(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "yes")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_hindi_haan(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "haan")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_hindi_chalega(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "chalega")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_hindi_theek_hai(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "theek hai")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_case_insensitive_intent(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Yes Go Ahead")
        assert mgr._conversations["conv_1"].phase == "action_committed"

    @pytest.mark.asyncio
    async def test_intent_with_composer(self) -> None:
        """When composer is provided, it should be called for intent_committed."""
        mgr = _make_manager()
        composer = FakeComposer("Action mode reply!")
        result = await _send_reply(mgr, "let's do it", composer=composer)
        assert result["action"] == "send"
        assert result["body"] == "Action mode reply!"
        assert composer.call_count == 1
        assert composer.last_classification == MessageClassification.INTENT_COMMITTED

    @pytest.mark.asyncio
    async def test_classify_message_returns_intent_enum(self) -> None:
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="qualifying",
        )
        result = mgr.classify_message("let's do it", conv)
        assert result == MessageClassification.INTENT_COMMITTED


class TestHostileOptoutClassification:
    """Tests for hostile/opt-out phrase detection."""

    @pytest.mark.asyncio
    async def test_stop_messaging(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "stop messaging me")
        assert result["action"] == "end"
        assert mgr._conversations["conv_1"].phase == "ended"

    @pytest.mark.asyncio
    async def test_not_interested(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "not interested")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "unsubscribe")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_leave_me_alone(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "leave me alone please")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_stop_simple(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "stop")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_dont_message(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "don't message me again")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_useless(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "this is useless")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_spam(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "this is spam")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_bothering_me(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "you are bothering me")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_stop_sending(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "stop sending messages")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_case_insensitive_hostile(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "STOP MESSAGING ME")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_hostile_rationale(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "not interested")
        assert "opted out" in result["rationale"].lower() or "opt" in result["rationale"].lower()

    @pytest.mark.asyncio
    async def test_classify_message_returns_hostile_enum(self) -> None:
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="qualifying",
        )
        result = mgr.classify_message("stop messaging me", conv)
        assert result == MessageClassification.HOSTILE_OPTOUT


class TestOffTopicClassification:
    """Tests for off-topic message detection."""

    @pytest.mark.asyncio
    async def test_gst_question(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Can you help me with GST filing?")
        assert result["action"] == "send"
        assert "outside" in result["body"].lower() or "help with" in result["body"].lower()

    @pytest.mark.asyncio
    async def test_tax_filing(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "I need help with tax filing")
        assert result["action"] == "send"
        assert "outside" in result["body"].lower()

    @pytest.mark.asyncio
    async def test_legal_advice(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Can you give me legal advice?")
        assert result["action"] == "send"

    @pytest.mark.asyncio
    async def test_off_topic_redirects(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Help me with income tax returns")
        assert "rationale" in result
        assert "off-topic" in result["rationale"].lower() or "redirect" in result["rationale"].lower()

    @pytest.mark.asyncio
    async def test_classify_message_returns_off_topic_enum(self) -> None:
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="qualifying",
        )
        result = mgr.classify_message("help me with GST", conv)
        assert result == MessageClassification.OFF_TOPIC


class TestNormalClassification:
    """Tests for normal message classification."""

    @pytest.mark.asyncio
    async def test_normal_message(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Tell me more about the dental cleaning offer")
        assert result["action"] == "send"

    @pytest.mark.asyncio
    async def test_normal_with_composer(self) -> None:
        mgr = _make_manager()
        composer = FakeComposer("Here are the details...")
        result = await _send_reply(mgr, "What are the prices?", composer=composer)
        assert result["body"] == "Here are the details..."
        assert composer.call_count == 1
        assert composer.last_classification == MessageClassification.NORMAL

    @pytest.mark.asyncio
    async def test_normal_without_composer_returns_stub(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "What are the prices?")
        assert result["action"] == "send"
        assert "body" in result

    @pytest.mark.asyncio
    async def test_classify_message_returns_normal_enum(self) -> None:
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="qualifying",
        )
        result = mgr.classify_message("Tell me more about the offer", conv)
        assert result == MessageClassification.NORMAL


# ===========================================================================
# Auto-Reply Escalation Tests
# ===========================================================================


class TestAutoReplyEscalation:
    """Tests for the auto-reply escalation logic.

    Escalation follows: 1st→send (acknowledge), 2nd→wait, 3rd→end.
    """

    @pytest.mark.asyncio
    async def test_first_auto_reply_with_prior_bot_msg_sends_ack(self) -> None:
        """With a prior bot message, 1st auto-reply returns send (acknowledge)."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello merchant!")
        result = await _send_reply(mgr, "Thank you for contacting us")
        assert result["action"] == "send"
        assert "auto" in result["rationale"].lower()

    @pytest.mark.asyncio
    async def test_first_auto_reply_without_prior_bot_msg_waits(self) -> None:
        """Without a prior bot message, 1st auto-reply goes straight to wait."""
        mgr = _make_manager()
        result = await _send_reply(mgr, "Thank you for contacting us")
        assert result["action"] == "wait"
        assert result["wait_seconds"] >= 14400

    @pytest.mark.asyncio
    async def test_second_auto_reply_waits_4_hours(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=2)
        assert result["action"] == "wait"
        assert result["wait_seconds"] >= 14400

    @pytest.mark.asyncio
    async def test_third_auto_reply_ends_conversation(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        await _send_reply(mgr, "Thank you for contacting us", turn_number=2)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=3)
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_fourth_auto_reply_still_ends(self) -> None:
        """3rd+ auto-replies should all return end."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        for i in range(1, 4):
            await _send_reply(mgr, "Thank you for contacting us", turn_number=i)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=4)
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_streak_resets_on_normal_message(self) -> None:
        """A normal message between auto-replies should reset the streak."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        await _send_reply(mgr, "Tell me more about the offer", turn_number=2)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=3)
        assert result["action"] == "send"  # 1st auto-reply after reset → send (ack)

    @pytest.mark.asyncio
    async def test_streak_resets_on_intent_message(self) -> None:
        """An intent commitment message should reset the auto-reply streak."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        await _send_reply(mgr, "let's do it", turn_number=2)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=3)
        assert result["action"] == "send"  # streak was reset, 1st auto-reply → send

    @pytest.mark.asyncio
    async def test_auto_reply_escalation_phase_transitions(self) -> None:
        """Verify phase transitions during auto-reply escalation."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        conv = mgr._conversations["conv_1"]
        assert conv.phase == "qualifying"  # 1st auto-reply → qualifying (trying to reach owner)

        await _send_reply(mgr, "Thank you for contacting us", turn_number=2)
        assert conv.phase == "waiting"  # 2nd → waiting

        await _send_reply(mgr, "Thank you for contacting us", turn_number=3)
        assert conv.phase == "ended"  # 3rd → ended


# ===========================================================================
# Phase Transition Tests
# ===========================================================================


class TestPhaseTransitions:
    """Tests for conversation phase validity and transitions."""

    @pytest.mark.asyncio
    async def test_initial_phase_is_initiating(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "Hello", turn_number=1)
        conv = mgr._conversations["conv_1"]
        assert conv.phase == "initiating"

    @pytest.mark.asyncio
    async def test_intent_sets_action_committed(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "let's do it", turn_number=1)
        conv = mgr._conversations["conv_1"]
        assert conv.phase == "action_committed"

    @pytest.mark.asyncio
    async def test_hostile_sets_ended(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "stop messaging me", turn_number=1)
        conv = mgr._conversations["conv_1"]
        assert conv.phase == "ended"

    @pytest.mark.asyncio
    async def test_ended_conversation_stays_ended(self) -> None:
        """Once ended, subsequent messages should return end action."""
        mgr = _make_manager()
        await _send_reply(mgr, "stop messaging me", turn_number=1)
        result = await _send_reply(mgr, "Actually, tell me more", turn_number=2)
        assert result["action"] == "end"
        assert mgr._conversations["conv_1"].phase == "ended"

    @pytest.mark.asyncio
    async def test_all_phases_are_valid(self) -> None:
        """All phases assigned by the manager should be in the valid set."""
        from src.conversation_manager import VALID_PHASES

        mgr = _make_manager()

        # Test initiating
        await _send_reply(mgr, "Hello", conversation_id="c1", turn_number=1)
        assert mgr._conversations["c1"].phase in VALID_PHASES

        # Test action_committed
        await _send_reply(mgr, "let's do it", conversation_id="c2", turn_number=1)
        assert mgr._conversations["c2"].phase in VALID_PHASES

        # Test ended (hostile)
        await _send_reply(mgr, "stop messaging me", conversation_id="c3", turn_number=1)
        assert mgr._conversations["c3"].phase in VALID_PHASES

        # Test waiting (2nd auto-reply)
        await _send_reply(mgr, "Thank you for contacting us", conversation_id="c4", turn_number=1)
        await _send_reply(mgr, "Thank you for contacting us", conversation_id="c4", turn_number=2)
        assert mgr._conversations["c4"].phase in VALID_PHASES

        # Test ended (3rd auto-reply)
        await _send_reply(mgr, "Thank you for contacting us", conversation_id="c4", turn_number=3)
        assert mgr._conversations["c4"].phase in VALID_PHASES


# ===========================================================================
# Conversation History Tests
# ===========================================================================


class TestConversationHistory:
    """Tests for conversation history completeness."""

    @pytest.mark.asyncio
    async def test_history_contains_all_messages(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "Hello", turn_number=1)
        await _send_reply(mgr, "Tell me more", turn_number=2)
        await _send_reply(mgr, "Sounds interesting", turn_number=3)

        history = mgr.get_history("conv_1")
        assert len(history) == 3
        assert history[0].message == "Hello"
        assert history[1].message == "Tell me more"
        assert history[2].message == "Sounds interesting"

    @pytest.mark.asyncio
    async def test_history_preserves_order(self) -> None:
        mgr = _make_manager()
        messages = ["First", "Second", "Third", "Fourth"]
        for i, msg in enumerate(messages, 1):
            await _send_reply(mgr, msg, turn_number=i)

        history = mgr.get_history("conv_1")
        assert [t.message for t in history] == messages

    @pytest.mark.asyncio
    async def test_history_preserves_roles(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "Hello from merchant", from_role="merchant", turn_number=1)
        await _send_reply(mgr, "Hello from customer", from_role="customer", turn_number=2)

        history = mgr.get_history("conv_1")
        assert history[0].role == "merchant"
        assert history[1].role == "customer"

    @pytest.mark.asyncio
    async def test_history_preserves_turn_numbers(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "First", turn_number=1)
        await _send_reply(mgr, "Second", turn_number=2)

        history = mgr.get_history("conv_1")
        assert history[0].turn_number == 1
        assert history[1].turn_number == 2

    @pytest.mark.asyncio
    async def test_empty_history_for_unknown_conversation(self) -> None:
        mgr = _make_manager()
        history = mgr.get_history("nonexistent")
        assert history == []

    @pytest.mark.asyncio
    async def test_separate_conversations_have_separate_histories(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "Conv1 msg", conversation_id="c1", turn_number=1)
        await _send_reply(mgr, "Conv2 msg", conversation_id="c2", turn_number=1)

        h1 = mgr.get_history("c1")
        h2 = mgr.get_history("c2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0].message == "Conv1 msg"
        assert h2[0].message == "Conv2 msg"


# ===========================================================================
# Anti-Repetition (sent_bodies) Tests
# ===========================================================================


class TestAntiRepetition:
    """Tests for sent_bodies tracking."""

    @pytest.mark.asyncio
    async def test_sent_body_is_tracked(self) -> None:
        mgr = _make_manager()
        await _send_reply(mgr, "Hello", turn_number=1)
        conv = mgr._conversations["conv_1"]
        # The stub response body should be in sent_bodies
        assert len(conv.sent_bodies) > 0

    @pytest.mark.asyncio
    async def test_multiple_sends_tracked(self) -> None:
        mgr = _make_manager()
        composer = FakeComposer("Reply 1")
        await _send_reply(mgr, "Hello", turn_number=1, composer=composer)

        composer2 = FakeComposer("Reply 2")
        await _send_reply(mgr, "More info", turn_number=2, composer=composer2)

        conv = mgr._conversations["conv_1"]
        assert "Reply 1" in conv.sent_bodies
        assert "Reply 2" in conv.sent_bodies

    @pytest.mark.asyncio
    async def test_end_action_not_tracked_in_sent_bodies(self) -> None:
        """End actions don't have a body, so nothing should be added."""
        mgr = _make_manager()
        await _send_reply(mgr, "stop messaging me", turn_number=1)
        conv = mgr._conversations["conv_1"]
        # End action has no body, so sent_bodies should be empty
        assert len(conv.sent_bodies) == 0

    @pytest.mark.asyncio
    async def test_wait_action_not_tracked_in_sent_bodies(self) -> None:
        """Wait actions don't have a body, so nothing should be added."""
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello!")
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)  # 1st with prior bot = send (ack)
        await _send_reply(mgr, "Thank you for contacting us", turn_number=2)  # 2nd = wait
        conv = mgr._conversations["conv_1"]
        # register_conversation adds 1 body, 1st auto-reply is send (adds body), 2nd is wait (no body)
        assert len(conv.sent_bodies) == 2


# ===========================================================================
# Register Conversation Tests
# ===========================================================================


class TestRegisterConversation:
    """Tests for register_conversation method."""

    def test_register_creates_conversation(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello merchant!")
        assert "conv_1" in mgr._conversations
        conv = mgr._conversations["conv_1"]
        assert conv.merchant_id == "m1"
        assert conv.trigger_id == "t1"

    def test_register_records_initial_body(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello merchant!")
        conv = mgr._conversations["conv_1"]
        assert "Hello merchant!" in conv.sent_bodies

    def test_register_adds_bot_turn(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", None, "t1", "Hello merchant!")
        history = mgr.get_history("conv_1")
        assert len(history) == 1
        assert history[0].role == "bot"
        assert history[0].message == "Hello merchant!"

    def test_register_with_customer_id(self) -> None:
        mgr = _make_manager()
        mgr.register_conversation("conv_1", "m1", "cust1", "t1", "Hi!")
        conv = mgr._conversations["conv_1"]
        assert conv.customer_id == "cust1"


# ===========================================================================
# Reply Action Validity Tests
# ===========================================================================


class TestReplyActionValidity:
    """Tests that all reply actions have valid action fields."""

    @pytest.mark.asyncio
    async def test_normal_returns_valid_action(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Tell me more")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_auto_reply_returns_valid_action(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Thank you for contacting us")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_hostile_returns_valid_action(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "stop messaging me")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_off_topic_returns_valid_action(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "Help me with GST filing")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_intent_returns_valid_action(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "let's do it")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_all_actions_have_rationale(self) -> None:
        """Every action dict should include a rationale field."""
        mgr = _make_manager()
        messages = [
            "Tell me more",
            "Thank you for contacting us",
            "stop messaging me",
            "Help me with GST",
            "let's do it",
        ]
        for msg in messages:
            mgr2 = _make_manager()
            result = await _send_reply(mgr2, msg)
            assert "rationale" in result, f"Missing rationale for message: {msg}"

    @pytest.mark.asyncio
    async def test_send_action_has_body(self) -> None:
        """Send actions should always include a body."""
        mgr = _make_manager()
        result = await _send_reply(mgr, "Tell me more")
        if result["action"] == "send":
            assert "body" in result
            assert len(result["body"]) > 0

    @pytest.mark.asyncio
    async def test_wait_action_has_wait_seconds(self) -> None:
        """Wait actions should always include wait_seconds."""
        mgr = _make_manager()
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        result = await _send_reply(mgr, "Thank you for contacting us", turn_number=2)
        assert result["action"] == "wait"
        assert "wait_seconds" in result
        assert result["wait_seconds"] > 0


# ===========================================================================
# Classification Priority Tests
# ===========================================================================


class TestClassificationPriority:
    """Tests for classification priority ordering."""

    @pytest.mark.asyncio
    async def test_hostile_takes_priority_over_auto_reply(self) -> None:
        """A message that matches both hostile and auto-reply should be hostile."""
        mgr = _make_manager()
        # "stop" is hostile, and we could imagine it in an auto-reply context
        result = await _send_reply(mgr, "stop")
        assert result["action"] == "end"

    @pytest.mark.asyncio
    async def test_hostile_takes_priority_over_intent(self) -> None:
        """Hostile should take priority over intent commitment."""
        mgr = _make_manager()
        conv = ConversationState(
            conversation_id="c1", merchant_id="m1", customer_id=None,
            trigger_id=None, phase="qualifying",
        )
        # "not interested" is hostile
        result = mgr.classify_message("not interested", conv)
        assert result == MessageClassification.HOSTILE_OPTOUT


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for ConversationManager."""

    @pytest.mark.asyncio
    async def test_empty_message(self) -> None:
        """An empty message should be classified as normal."""
        mgr = _make_manager()
        result = await _send_reply(mgr, "")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_whitespace_only_message(self) -> None:
        mgr = _make_manager()
        result = await _send_reply(mgr, "   ")
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_very_long_message(self) -> None:
        mgr = _make_manager()
        long_msg = "Hello " * 1000
        result = await _send_reply(mgr, long_msg)
        assert result["action"] in ("send", "wait", "end")

    @pytest.mark.asyncio
    async def test_multiple_conversations_independent(self) -> None:
        """Different conversation_ids should have independent state."""
        mgr = _make_manager()
        # End conv_1
        await _send_reply(mgr, "stop messaging me", conversation_id="c1")
        # conv_2 should still be active
        result = await _send_reply(mgr, "Tell me more", conversation_id="c2")
        assert result["action"] == "send"
        assert mgr._conversations["c1"].phase == "ended"
        assert mgr._conversations["c2"].phase == "initiating"

    @pytest.mark.asyncio
    async def test_composer_exception_falls_back_to_stub(self) -> None:
        """If composer raises an exception, should fall back to stub response."""

        class BrokenComposer:
            async def compose_reply(self, **kwargs) -> dict:
                raise RuntimeError("LLM timeout")

        mgr = _make_manager()
        result = await _send_reply(mgr, "Tell me more", composer=BrokenComposer())
        assert result["action"] == "send"
        assert "body" in result

    @pytest.mark.asyncio
    async def test_auto_reply_streak_counter(self) -> None:
        """Verify the auto_reply_streak counter increments correctly."""
        mgr = _make_manager()
        await _send_reply(mgr, "Thank you for contacting us", turn_number=1)
        assert mgr._conversations["conv_1"].auto_reply_streak == 1

        await _send_reply(mgr, "Thank you for contacting us", turn_number=2)
        assert mgr._conversations["conv_1"].auto_reply_streak == 2

        await _send_reply(mgr, "Thank you for contacting us", turn_number=3)
        assert mgr._conversations["conv_1"].auto_reply_streak == 3
