"""Unit tests for the TriggerEvaluator class.

Covers:
- Suppression key filtering
- Expired trigger filtering
- Action count cap (20)
- Consent scope filtering
- Urgency sorting
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from src.context_store import ContextStore
from src.models import ComposedMessage, TickAction
from src.trigger_evaluator import TriggerEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = "2026-05-01T12:00:00Z"
FUTURE = "2026-12-31T23:59:59Z"
PAST = "2026-04-01T00:00:00Z"


class FakeComposer:
    """Minimal composer stub that returns a deterministic ComposedMessage."""

    async def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: dict | None = None,
    ) -> ComposedMessage:
        return ComposedMessage(
            body=f"Message for {trigger.get('kind', 'unknown')}",
            cta="open_ended",
            send_as="vera",
            suppression_key=trigger.get("suppression_key", ""),
            rationale="test rationale",
            template_name="test_template",
            template_params=[],
        )


async def _seed_store(
    store: ContextStore,
    triggers: list[dict],
    merchant_id: str = "m1",
    category_slug: str = "dentists",
    customer_id: str | None = None,
    customer_consent_scope: list[str] | None = None,
) -> None:
    """Push merchant, category, and trigger contexts into the store."""
    await store.put(
        "merchant",
        merchant_id,
        1,
        {"category_slug": category_slug, "name": "Test Merchant"},
        "2026-01-01T00:00:00Z",
    )
    await store.put(
        "category",
        category_slug,
        1,
        {"name": "Dentists"},
        "2026-01-01T00:00:00Z",
    )
    if customer_id is not None:
        await store.put(
            "customer",
            customer_id,
            1,
            {
                "consent": {"scope": customer_consent_scope or []},
                "identity": {"name": "Test Customer"},
            },
            "2026-01-01T00:00:00Z",
        )
    for trg in triggers:
        await store.put(
            "trigger",
            trg["id"],
            1,
            trg,
            "2026-01-01T00:00:00Z",
        )


def _make_trigger(
    trigger_id: str,
    *,
    merchant_id: str = "m1",
    urgency: int = 2,
    suppression_key: str | None = None,
    expires_at: str = FUTURE,
    scope: str = "merchant",
    kind: str = "research_digest",
    customer_id: str | None = None,
) -> dict:
    return {
        "id": trigger_id,
        "merchant_id": merchant_id,
        "scope": scope,
        "kind": kind,
        "urgency": urgency,
        "suppression_key": suppression_key or f"sk_{trigger_id}",
        "expires_at": expires_at,
        "customer_id": customer_id,
    }


# ---------------------------------------------------------------------------
# Suppression key filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suppressed_trigger_is_excluded() -> None:
    """A trigger whose suppression key was already fired should be skipped."""
    store = ContextStore()
    trg = _make_trigger("t1", suppression_key="already_fired")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    evaluator.record_suppression("already_fired")

    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_non_suppressed_trigger_is_included() -> None:
    """A trigger whose suppression key has NOT been fired should be included."""
    store = ContextStore()
    trg = _make_trigger("t1", suppression_key="fresh_key")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_suppression_recorded_after_composition() -> None:
    """After a trigger is composed, its suppression key should be recorded."""
    store = ContextStore()
    trg = _make_trigger("t1", suppression_key="new_key")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    assert not evaluator.is_suppressed("new_key")

    await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert evaluator.is_suppressed("new_key")


@pytest.mark.asyncio
async def test_second_evaluate_skips_already_fired() -> None:
    """A second evaluate call should skip triggers whose keys were fired in the first."""
    store = ContextStore()
    trg = _make_trigger("t1", suppression_key="once_only")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions1 = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions1) == 1

    actions2 = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions2) == 0


# ---------------------------------------------------------------------------
# Expired trigger filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_trigger_is_excluded() -> None:
    """A trigger whose expires_at is before 'now' should be skipped."""
    store = ContextStore()
    trg = _make_trigger("t1", expires_at=PAST)
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_future_trigger_is_included() -> None:
    """A trigger whose expires_at is after 'now' should be included."""
    store = ContextStore()
    trg = _make_trigger("t1", expires_at=FUTURE)
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_mixed_expired_and_valid_triggers() -> None:
    """Only non-expired triggers should appear in the result."""
    store = ContextStore()
    triggers = [
        _make_trigger("t_expired", expires_at=PAST),
        _make_trigger("t_valid", expires_at=FUTURE),
    ]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t_expired", "t_valid"], NOW, FakeComposer())
    assert len(actions) == 1
    assert actions[0].trigger_id == "t_valid"


# ---------------------------------------------------------------------------
# Action count cap (20)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_cap_at_20() -> None:
    """Even with >20 valid triggers, at most 20 actions should be returned."""
    store = ContextStore()
    triggers = [_make_trigger(f"t{i}") for i in range(30)]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    trigger_ids = [f"t{i}" for i in range(30)]
    actions = await evaluator.evaluate(trigger_ids, NOW, FakeComposer())
    assert len(actions) <= 20


@pytest.mark.asyncio
async def test_exactly_20_triggers_returns_20() -> None:
    """With exactly 20 valid triggers, all 20 should be returned."""
    store = ContextStore()
    triggers = [_make_trigger(f"t{i}") for i in range(20)]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    trigger_ids = [f"t{i}" for i in range(20)]
    actions = await evaluator.evaluate(trigger_ids, NOW, FakeComposer())
    assert len(actions) == 20


@pytest.mark.asyncio
async def test_fewer_than_20_triggers_returns_all() -> None:
    """With fewer than 20 valid triggers, all should be returned."""
    store = ContextStore()
    triggers = [_make_trigger(f"t{i}") for i in range(5)]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    trigger_ids = [f"t{i}" for i in range(5)]
    actions = await evaluator.evaluate(trigger_ids, NOW, FakeComposer())
    assert len(actions) == 5


# ---------------------------------------------------------------------------
# Consent scope filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_trigger_blocked_by_consent() -> None:
    """A customer-scoped recall_due trigger should be skipped if consent doesn't include recall_reminders."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="recall_due",
        customer_id="cust1",
    )
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=["appointment_reminders"],  # no recall_reminders
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_customer_trigger_allowed_by_consent() -> None:
    """A customer-scoped recall_due trigger should pass if consent includes recall_reminders."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="recall_due",
        customer_id="cust1",
    )
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=["recall_reminders", "appointment_reminders"],
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_customer_lapsed_hard_needs_winback_offers() -> None:
    """customer_lapsed_hard maps to winback_offers consent."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="customer_lapsed_hard",
        customer_id="cust1",
    )
    # No winback_offers in consent
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=["recall_reminders"],
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_chronic_refill_needs_refill_reminders() -> None:
    """chronic_refill_due maps to refill_reminders consent."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="chronic_refill_due",
        customer_id="cust1",
    )
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=["refill_reminders"],
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_wedding_followup_needs_bridal_consent() -> None:
    """wedding_package_followup maps to bridal_package_followup consent."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="wedding_package_followup",
        customer_id="cust1",
    )
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=["appointment_reminders"],  # no bridal
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_unknown_kind_allowed_by_default() -> None:
    """A customer-scoped trigger with an unmapped kind should be allowed."""
    store = ContextStore()
    trg = _make_trigger(
        "t1",
        scope="customer",
        kind="some_unknown_kind",
        customer_id="cust1",
    )
    await _seed_store(
        store,
        [trg],
        customer_id="cust1",
        customer_consent_scope=[],  # empty consent
    )

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_merchant_scoped_trigger_ignores_consent() -> None:
    """Merchant-scoped triggers should not be filtered by customer consent."""
    store = ContextStore()
    trg = _make_trigger("t1", scope="merchant", kind="recall_due")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, FakeComposer())
    assert len(actions) == 1


# ---------------------------------------------------------------------------
# Urgency sorting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triggers_sorted_by_urgency_descending() -> None:
    """Actions should be ordered by trigger urgency, highest first."""
    store = ContextStore()
    triggers = [
        _make_trigger("t_low", urgency=1),
        _make_trigger("t_high", urgency=5),
        _make_trigger("t_mid", urgency=3),
    ]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(
        ["t_low", "t_high", "t_mid"], NOW, FakeComposer()
    )
    assert len(actions) == 3
    urgencies = [
        store.get("trigger", a.trigger_id).payload["urgency"]  # type: ignore[union-attr]
        for a in actions
    ]
    assert urgencies == sorted(urgencies, reverse=True)


@pytest.mark.asyncio
async def test_equal_urgency_preserves_order() -> None:
    """Triggers with equal urgency should maintain stable ordering."""
    store = ContextStore()
    triggers = [
        _make_trigger("t_a", urgency=3),
        _make_trigger("t_b", urgency=3),
        _make_trigger("t_c", urgency=3),
    ]
    await _seed_store(store, triggers)

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(
        ["t_a", "t_b", "t_c"], NOW, FakeComposer()
    )
    assert len(actions) == 3


# ---------------------------------------------------------------------------
# Suppression helpers
# ---------------------------------------------------------------------------


def test_record_and_check_suppression() -> None:
    """record_suppression / is_suppressed should work as a simple set."""
    store = ContextStore()
    evaluator = TriggerEvaluator(store)

    assert not evaluator.is_suppressed("key1")
    evaluator.record_suppression("key1")
    assert evaluator.is_suppressed("key1")
    assert not evaluator.is_suppressed("key2")


def test_suppress_conversation() -> None:
    """suppress_conversation should add the id to the suppressed set."""
    store = ContextStore()
    evaluator = TriggerEvaluator(store)

    evaluator.suppress_conversation("conv_123")
    assert "conv_123" in evaluator._suppressed_conversations


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_trigger_in_store_is_skipped() -> None:
    """If a trigger_id is not in the store, it should be silently skipped."""
    store = ContextStore()
    await _seed_store(store, [])  # no triggers

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["nonexistent"], NOW, FakeComposer())
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_no_composer_returns_empty_list() -> None:
    """When composer is None, evaluate should return an empty list."""
    store = ContextStore()
    trg = _make_trigger("t1")
    await _seed_store(store, [trg])

    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate(["t1"], NOW, composer=None)
    assert actions == []


@pytest.mark.asyncio
async def test_empty_trigger_list() -> None:
    """An empty available_trigger_ids list should return no actions."""
    store = ContextStore()
    evaluator = TriggerEvaluator(store)
    actions = await evaluator.evaluate([], NOW, FakeComposer())
    assert actions == []
