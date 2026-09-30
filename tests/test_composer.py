"""Tests for the Composer Engine and prompt templates.

Covers:
- Prompt dispatch: verify correct prompt selected for each trigger kind
- Customer-scoped trigger routing: verify send_as = "merchant_on_behalf"
- Context block building: verify all context sections present
- LLM response parsing: valid JSON, malformed JSON fallback
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.composer import Composer
from src.context_store import ContextStore
from src.llm_client import LLMClient
from src.models import ComposedMessage, MessageClassification
from src.prompts import (
    CUSTOMER_SCOPED_KINDS,
    PROMPT_REGISTRY,
    get_trigger_instruction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_category(slug: str = "dentists") -> dict:
    return {
        "slug": slug,
        "voice": {
            "tone": "peer_clinical",
            "vocab_allowed": ["fluoride varnish", "scaling"],
            "vocab_taboo": ["guaranteed", "100% safe"],
        },
        "offer_catalog": [
            {"id": "den_001", "title": "Dental Cleaning @ ₹299", "type": "service_at_price"},
        ],
        "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030},
        "digest": [
            {
                "id": "d_001",
                "title": "3-month fluoride varnish recall outperforms 6-month",
                "source": "JIDA Oct 2026, p.14",
            }
        ],
        "seasonal_beats": [
            {"month_range": "Nov-Feb", "note": "exam-stress bruxism spike"},
        ],
        "trend_signals": [
            {"query": "clear aligners delhi", "delta_yoy": 0.62},
        ],
    }


def _make_merchant() -> dict:
    return {
        "merchant_id": "m_001",
        "category_slug": "dentists",
        "identity": {
            "name": "Dr. Meera's Dental Clinic",
            "owner_first_name": "Meera",
            "city": "Delhi",
            "locality": "Lajpat Nagar",
            "languages": ["en", "hi"],
        },
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {
            "views": 2410,
            "calls": 18,
            "ctr": 0.021,
            "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05},
        },
        "offers": [
            {"id": "o_001", "title": "Dental Cleaning @ ₹299", "status": "active"},
        ],
        "conversation_history": [
            {"from": "vera", "body": "Profile audit done."},
        ],
        "customer_aggregate": {
            "total_unique_ytd": 540,
            "lapsed_180d_plus": 78,
            "high_risk_adult_count": 124,
        },
        "signals": ["stale_posts:22d", "ctr_below_peer_median"],
        "review_themes": [
            {
                "theme": "wait_time",
                "sentiment": "neg",
                "occurrences_30d": 3,
                "common_quote": "had to wait 30 min",
            }
        ],
    }


def _make_trigger(kind: str = "research_digest", scope: str = "merchant") -> dict:
    return {
        "id": "trg_001",
        "kind": kind,
        "scope": scope,
        "source": "external",
        "urgency": 2,
        "suppression_key": f"test:{kind}:2026",
        "payload": {"category": "dentists"},
    }


def _make_customer() -> dict:
    return {
        "customer_id": "c_001",
        "merchant_id": "m_001",
        "identity": {
            "name": "Priya",
            "language_pref": "hi-en mix",
            "age_band": "25-35",
        },
        "state": "lapsed_soft",
        "relationship": {
            "first_visit": "2025-11-04",
            "last_visit": "2026-05-12",
            "visits_total": 4,
            "services_received": ["cleaning", "whitening"],
        },
        "preferences": {"preferred_slots": "weekday_evening"},
        "consent": {"scope": ["recall_reminders", "appointment_reminders"]},
    }


def _make_llm_response(
    body: str = "Test message",
    cta: str = "open_ended",
    send_as: str = "vera",
    suppression_key: str = "test:key",
    rationale: str = "Test rationale",
    template_name: str = "vera_test_v1",
) -> str:
    return json.dumps({
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
        "template_name": template_name,
        "template_params": [],
    })


def _make_composer(llm_response: str | None = None) -> Composer:
    """Create a Composer with a mocked LLM client."""
    llm_client = AsyncMock(spec=LLMClient)
    if llm_response is not None:
        llm_client.complete_with_fallback = AsyncMock(return_value=llm_response)
    else:
        llm_client.complete_with_fallback = AsyncMock(
            return_value=_make_llm_response()
        )
    store = ContextStore()
    return Composer(llm_client=llm_client, context_store=store)


# ---------------------------------------------------------------------------
# Prompt dispatch tests
# ---------------------------------------------------------------------------


class TestPromptDispatch:
    """Verify correct prompt selected for each trigger kind."""

    def test_all_known_trigger_kinds_have_registry_entries(self) -> None:
        """Every trigger kind in the registry should have a non-empty instruction."""
        for kind, instruction in PROMPT_REGISTRY.items():
            if kind == "_default":
                continue
            assert instruction, f"Empty instruction for trigger kind: {kind}"

    @pytest.mark.parametrize(
        "trigger_kind",
        [
            "research_digest",
            "recall_due",
            "perf_dip",
            "seasonal_perf_dip",
            "active_planning_intent",
            "supply_alert",
            "renewal_due",
            "competitor_opened",
            "review_theme_emerged",
            "milestone_reached",
            "ipl_match_today",
            "customer_lapsed_hard",
            "customer_lapsed_soft",
            "chronic_refill_due",
            "trial_followup",
            "festival_upcoming",
            "dormant_with_vera",
            "cde_opportunity",
            "gbp_unverified",
            "winback_eligible",
            "wedding_package_followup",
            "category_seasonal",
            "regulation_change",
            "perf_spike",
            "curious_ask_due",
        ],
    )
    def test_trigger_kind_has_specific_prompt(self, trigger_kind: str) -> None:
        """Each known trigger kind should map to a specific prompt instruction."""
        instruction = get_trigger_instruction(trigger_kind)
        assert instruction
        assert trigger_kind.replace("_", " ") or True  # non-empty
        # Verify it's not the default
        default = PROMPT_REGISTRY["_default"]
        assert instruction != default, (
            f"Trigger kind '{trigger_kind}' fell through to default"
        )

    def test_unknown_trigger_kind_returns_default(self) -> None:
        """Unknown trigger kinds should return the default instruction."""
        instruction = get_trigger_instruction("totally_unknown_kind")
        assert instruction == PROMPT_REGISTRY["_default"]

    def test_select_prompt_includes_trigger_instruction(self) -> None:
        """_select_prompt should embed the trigger-specific instruction."""
        composer = _make_composer()
        category = _make_category()
        prompt = composer._select_prompt("research_digest", category)
        assert "source citation" in prompt.lower()
        assert "trial size" in prompt.lower()

    def test_select_prompt_includes_voice_rules(self) -> None:
        """_select_prompt should include category voice rules."""
        composer = _make_composer()
        category = _make_category()
        prompt = composer._select_prompt("research_digest", category)
        assert "peer_clinical" in prompt
        assert "guaranteed" in prompt  # taboo word listed
        assert "fluoride varnish" in prompt  # allowed vocab


# ---------------------------------------------------------------------------
# Customer-scoped trigger routing tests
# ---------------------------------------------------------------------------


class TestCustomerScopedRouting:
    """Verify send_as = 'merchant_on_behalf' for customer-scoped triggers."""

    def test_customer_scoped_kinds_defined(self) -> None:
        """CUSTOMER_SCOPED_KINDS should contain the expected trigger kinds."""
        expected = {
            "recall_due",
            "customer_lapsed_hard",
            "customer_lapsed_soft",
            "chronic_refill_due",
            "trial_followup",
            "wedding_package_followup",
            "appointment_tomorrow",
        }
        assert CUSTOMER_SCOPED_KINDS == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "trigger_kind",
        list(CUSTOMER_SCOPED_KINDS),
    )
    async def test_customer_scoped_trigger_sets_merchant_on_behalf(
        self, trigger_kind: str
    ) -> None:
        """Customer-scoped triggers with a customer should set send_as to merchant_on_behalf."""
        llm_response = _make_llm_response(
            body="Hi Priya, test message",
            send_as="vera",  # LLM might return wrong value
        )
        composer = _make_composer(llm_response)
        category = _make_category()
        merchant = _make_merchant()
        trigger = _make_trigger(kind=trigger_kind, scope="customer")
        customer = _make_customer()

        result = await composer.compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )

        assert result.send_as == "merchant_on_behalf"

    @pytest.mark.asyncio
    async def test_merchant_scoped_trigger_keeps_vera(self) -> None:
        """Merchant-scoped triggers should keep send_as as 'vera'."""
        llm_response = _make_llm_response(send_as="vera")
        composer = _make_composer(llm_response)
        category = _make_category()
        merchant = _make_merchant()
        trigger = _make_trigger(kind="research_digest", scope="merchant")

        result = await composer.compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
        )

        assert result.send_as == "vera"


# ---------------------------------------------------------------------------
# Context block building tests
# ---------------------------------------------------------------------------


class TestContextBlockBuilding:
    """Verify all context sections are present in the built context block."""

    def test_context_block_has_category_section(self) -> None:
        """Context block should contain the CATEGORY section."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "=== CATEGORY ===" in block
        assert "dentists" in block
        assert "peer_clinical" in block

    def test_context_block_has_merchant_section(self) -> None:
        """Context block should contain the MERCHANT section."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "=== MERCHANT ===" in block
        assert "Dr. Meera" in block
        assert "Lajpat Nagar" in block
        assert "2410" in block  # views

    def test_context_block_has_trigger_section(self) -> None:
        """Context block should contain the TRIGGER section."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "=== TRIGGER" in block
        assert "research_digest" in block

    def test_context_block_has_customer_section_when_present(self) -> None:
        """Context block should contain the CUSTOMER section when customer is provided."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(),
            _make_merchant(),
            _make_trigger(),
            _make_customer(),
        )
        assert "=== CUSTOMER ===" in block
        assert "Priya" in block
        assert "hi-en mix" in block

    def test_context_block_no_customer_section_when_absent(self) -> None:
        """Context block should NOT contain the CUSTOMER section when no customer."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "=== CUSTOMER ===" not in block

    def test_context_block_includes_offers(self) -> None:
        """Context block should include merchant offers."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "Dental Cleaning" in block

    def test_context_block_includes_signals(self) -> None:
        """Context block should include merchant signals."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "stale_posts" in block

    def test_context_block_includes_digest(self) -> None:
        """Context block should include category digest items."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "fluoride varnish" in block.lower() or "JIDA" in block

    def test_context_block_includes_review_themes(self) -> None:
        """Context block should include merchant review themes."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "wait_time" in block

    def test_context_block_includes_conversation_history(self) -> None:
        """Context block should include recent conversation history."""
        composer = _make_composer()
        block = composer._build_context_block(
            _make_category(), _make_merchant(), _make_trigger()
        )
        assert "Profile audit done" in block


# ---------------------------------------------------------------------------
# LLM response parsing tests
# ---------------------------------------------------------------------------


class TestLLMResponseParsing:
    """Verify JSON parsing from LLM output, including malformed responses."""

    def test_parse_valid_json(self) -> None:
        """Valid JSON should be parsed correctly."""
        composer = _make_composer()
        raw = _make_llm_response(body="Hello Dr. Meera")
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response(raw, trigger, merchant, None)

        assert isinstance(result, ComposedMessage)
        assert result.body == "Hello Dr. Meera"
        assert result.cta == "open_ended"

    def test_parse_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code blocks should be parsed."""
        composer = _make_composer()
        raw = '```json\n{"body": "Code block message", "cta": "none", "send_as": "vera", "suppression_key": "k", "rationale": "r", "template_name": "t", "template_params": []}\n```'
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response(raw, trigger, merchant, None)

        assert result.body == "Code block message"

    def test_parse_json_with_surrounding_text(self) -> None:
        """JSON embedded in surrounding text should be extracted."""
        composer = _make_composer()
        raw = 'Here is the message:\n{"body": "Embedded message", "cta": "none", "send_as": "vera", "suppression_key": "k", "rationale": "r", "template_name": "t", "template_params": []}\nDone!'
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response(raw, trigger, merchant, None)

        assert result.body == "Embedded message"

    def test_parse_malformed_json_returns_fallback(self) -> None:
        """Completely malformed output should return a fallback message."""
        composer = _make_composer()
        raw = "This is not JSON at all, just random text."
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response(raw, trigger, merchant, None)

        assert isinstance(result, ComposedMessage)
        assert result.body  # Should have some fallback body
        assert result.send_as == "vera"

    def test_parse_empty_string_returns_fallback(self) -> None:
        """Empty string should return a fallback message."""
        composer = _make_composer()
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response("", trigger, merchant, None)

        assert isinstance(result, ComposedMessage)
        assert result.body  # Non-empty fallback

    def test_parse_preserves_suppression_key(self) -> None:
        """Parsed message should preserve the suppression key from trigger."""
        composer = _make_composer()
        raw = _make_llm_response(suppression_key="custom:key")
        trigger = _make_trigger()
        merchant = _make_merchant()

        result = composer._parse_llm_response(raw, trigger, merchant, None)

        assert result.suppression_key == "custom:key"

    def test_parse_customer_scoped_overrides_send_as(self) -> None:
        """Customer-scoped triggers should override send_as to merchant_on_behalf."""
        composer = _make_composer()
        raw = _make_llm_response(send_as="vera")
        trigger = _make_trigger(kind="recall_due", scope="customer")
        merchant = _make_merchant()
        customer = _make_customer()

        result = composer._parse_llm_response(raw, trigger, merchant, customer)

        assert result.send_as == "merchant_on_behalf"


# ---------------------------------------------------------------------------
# Compose integration tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestComposeIntegration:
    """Integration tests for the full compose pipeline."""

    @pytest.mark.asyncio
    async def test_compose_returns_composed_message(self) -> None:
        """compose() should return a valid ComposedMessage."""
        llm_response = _make_llm_response(body="Dr. Meera, JIDA's Oct issue landed.")
        composer = _make_composer(llm_response)

        result = await composer.compose(
            category=_make_category(),
            merchant=_make_merchant(),
            trigger=_make_trigger(),
        )

        assert isinstance(result, ComposedMessage)
        assert "JIDA" in result.body

    @pytest.mark.asyncio
    async def test_compose_calls_llm_with_context(self) -> None:
        """compose() should call the LLM with system and user prompts."""
        composer = _make_composer()

        await composer.compose(
            category=_make_category(),
            merchant=_make_merchant(),
            trigger=_make_trigger(),
        )

        composer._llm.complete_with_fallback.assert_called_once()
        call_args = composer._llm.complete_with_fallback.call_args
        system_prompt = call_args[0][0]
        user_prompt = call_args[0][1]

        # System prompt should contain voice rules and trigger instructions
        assert "peer_clinical" in system_prompt
        assert "source citation" in system_prompt.lower()

        # User prompt should contain context block
        assert "=== CATEGORY ===" in user_prompt
        assert "=== MERCHANT ===" in user_prompt
        assert "=== TRIGGER" in user_prompt

    @pytest.mark.asyncio
    async def test_compose_with_sent_bodies_includes_them(self) -> None:
        """compose() should include sent_bodies in the user prompt."""
        composer = _make_composer()
        sent_bodies = {"Previous message body"}

        await composer.compose(
            category=_make_category(),
            merchant=_make_merchant(),
            trigger=_make_trigger(),
            sent_bodies=sent_bodies,
        )

        call_args = composer._llm.complete_with_fallback.call_args
        user_prompt = call_args[0][1]
        assert "Previous message body" in user_prompt

    @pytest.mark.asyncio
    async def test_compose_auto_fixes_url_violations(self) -> None:
        """compose() should auto-fix URL violations in the response."""
        llm_response = _make_llm_response(
            body="Check this out: https://example.com for details."
        )
        composer = _make_composer(llm_response)

        result = await composer.compose(
            category=_make_category(),
            merchant=_make_merchant(),
            trigger=_make_trigger(),
        )

        assert "https://" not in result.body


# ---------------------------------------------------------------------------
# Compose reply tests
# ---------------------------------------------------------------------------


class TestComposeReply:
    """Tests for compose_reply method."""

    @pytest.mark.asyncio
    async def test_compose_reply_intent_committed(self) -> None:
        """compose_reply with INTENT_COMMITTED should produce action-mode output."""
        reply_response = json.dumps({
            "action": "send",
            "body": "Great, drafting the plan now.",
            "cta": "open_ended",
            "rationale": "Intent committed — action mode.",
        })
        composer = _make_composer(reply_response)

        result = await composer.compose_reply(
            conversation_history=[],
            classification=MessageClassification.INTENT_COMMITTED,
            sent_bodies=set(),
        )

        assert result["action"] == "send"
        assert result["body"]

    @pytest.mark.asyncio
    async def test_compose_reply_normal(self) -> None:
        """compose_reply with NORMAL should continue conversation."""
        reply_response = json.dumps({
            "action": "send",
            "body": "Sure, let me check that for you.",
            "cta": "open_ended",
            "rationale": "Normal reply.",
        })
        composer = _make_composer(reply_response)

        result = await composer.compose_reply(
            conversation_history=[],
            classification=MessageClassification.NORMAL,
            sent_bodies=set(),
        )

        assert result["action"] == "send"

    @pytest.mark.asyncio
    async def test_compose_reply_malformed_llm_returns_fallback(self) -> None:
        """compose_reply with malformed LLM output should return a fallback."""
        composer = _make_composer("not json at all")

        result = await composer.compose_reply(
            conversation_history=[],
            classification=MessageClassification.NORMAL,
            sent_bodies=set(),
        )

        assert result["action"] == "send"
        assert result["body"]  # Non-empty fallback
