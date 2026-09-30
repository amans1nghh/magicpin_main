"""Tests for the FastAPI bot endpoints (healthz, metadata, context, tick, reply)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.bot import app, context_store, conversation_manager, trigger_evaluator


@pytest.fixture(autouse=True)
def _reset_store():
    """Clear the context store and conversation/trigger state between tests."""
    context_store._store.clear()
    conversation_manager._conversations.clear()
    trigger_evaluator._fired_suppression_keys.clear()
    trigger_evaluator._suppressed_conversations.clear()
    yield
    context_store._store.clear()
    conversation_manager._conversations.clear()
    trigger_evaluator._fired_suppression_keys.clear()
    trigger_evaluator._suppressed_conversations.clear()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthz_returns_ok(client: AsyncClient):
    resp = await client.get("/v1/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["uptime_seconds"], int)
    assert data["uptime_seconds"] >= 0
    assert "contexts_loaded" in data
    # All four scopes should be present with zero counts initially
    for scope in ("category", "merchant", "customer", "trigger"):
        assert data["contexts_loaded"][scope] == 0


@pytest.mark.asyncio
async def test_healthz_counts_after_context_push(client: AsyncClient):
    # Push two merchant contexts
    for i in range(2):
        await client.post(
            "/v1/context",
            json={
                "scope": "merchant",
                "context_id": f"m{i}",
                "version": 1,
                "payload": {"name": f"Merchant {i}"},
                "delivered_at": "2025-01-01T00:00:00Z",
            },
        )
    resp = await client.get("/v1/healthz")
    data = resp.json()
    assert data["contexts_loaded"]["merchant"] == 2
    assert data["contexts_loaded"]["category"] == 0


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_returns_expected_fields(client: AsyncClient):
    resp = await client.get("/v1/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["team_name"] == "Vera"
    assert isinstance(data["team_members"], list)
    assert len(data["team_members"]) > 0
    assert "model" in data
    assert "approach" in data
    assert "contact_email" in data
    assert "version" in data
    assert "submitted_at" in data


# ---------------------------------------------------------------------------
# POST /v1/context — accepted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_push_accepted(client: AsyncClient):
    resp = await client.post(
        "/v1/context",
        json={
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": {"name": "Dentists"},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert "ack_id" in data
    assert "stored_at" in data


@pytest.mark.asyncio
async def test_context_push_higher_version_accepted(client: AsyncClient):
    # Push v1
    await client.post(
        "/v1/context",
        json={
            "scope": "merchant",
            "context_id": "m1",
            "version": 1,
            "payload": {"name": "Old"},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    # Push v2
    resp = await client.post(
        "/v1/context",
        json={
            "scope": "merchant",
            "context_id": "m1",
            "version": 2,
            "payload": {"name": "New"},
            "delivered_at": "2025-01-01T00:01:00Z",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


# ---------------------------------------------------------------------------
# POST /v1/context — stale version (409)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_push_stale_version_rejected(client: AsyncClient):
    # Push v2 first
    await client.post(
        "/v1/context",
        json={
            "scope": "trigger",
            "context_id": "t1",
            "version": 2,
            "payload": {"kind": "recall_due"},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    # Push v1 — should be rejected as stale
    resp = await client.post(
        "/v1/context",
        json={
            "scope": "trigger",
            "context_id": "t1",
            "version": 1,
            "payload": {"kind": "old"},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    assert resp.status_code == 409
    data = resp.json()
    assert data["accepted"] is False
    assert data["reason"] == "stale_version"
    assert data["current_version"] == 2


@pytest.mark.asyncio
async def test_context_push_same_version_rejected(client: AsyncClient):
    # Push v1
    await client.post(
        "/v1/context",
        json={
            "scope": "customer",
            "context_id": "c1",
            "version": 1,
            "payload": {"name": "Alice"},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    # Push v1 again — should be rejected (equal version is stale)
    resp = await client.post(
        "/v1/context",
        json={
            "scope": "customer",
            "context_id": "c1",
            "version": 1,
            "payload": {"name": "Alice v2"},
            "delivered_at": "2025-01-01T00:01:00Z",
        },
    )
    assert resp.status_code == 409
    data = resp.json()
    assert data["accepted"] is False
    assert data["reason"] == "stale_version"


# ---------------------------------------------------------------------------
# POST /v1/context — invalid scope (400)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_push_invalid_scope_rejected(client: AsyncClient):
    resp = await client.post(
        "/v1/context",
        json={
            "scope": "unknown_scope",
            "context_id": "x1",
            "version": 1,
            "payload": {},
            "delivered_at": "2025-01-01T00:00:00Z",
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["accepted"] is False
    assert data["reason"] == "invalid_scope"


# ---------------------------------------------------------------------------
# POST /v1/tick — wired pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_returns_empty_actions_when_no_triggers_in_store(client: AsyncClient):
    """Tick with trigger IDs not in the store returns empty actions."""
    resp = await client.post(
        "/v1/tick",
        json={
            "now": "2025-01-01T12:00:00Z",
            "available_triggers": ["t1", "t2"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["actions"] == []


@pytest.mark.asyncio
async def test_tick_returns_empty_actions_when_no_triggers_provided(client: AsyncClient):
    """Tick with empty available_triggers returns empty actions."""
    resp = await client.post(
        "/v1/tick",
        json={
            "now": "2025-01-01T12:00:00Z",
            "available_triggers": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["actions"] == []


# ---------------------------------------------------------------------------
# POST /v1/reply — wired pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply_returns_valid_action(client: AsyncClient):
    """Reply endpoint returns a valid action dict (send, wait, or end)."""
    resp = await client.post(
        "/v1/reply",
        json={
            "conversation_id": "conv-1",
            "merchant_id": "m1",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Hello",
            "received_at": "2025-01-01T12:00:00Z",
            "turn_number": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] in ("send", "wait", "end")
    assert "rationale" in data


@pytest.mark.asyncio
async def test_reply_hostile_returns_end(client: AsyncClient):
    """Reply with hostile/opt-out message returns end action."""
    resp = await client.post(
        "/v1/reply",
        json={
            "conversation_id": "conv-hostile",
            "merchant_id": "m1",
            "customer_id": None,
            "from_role": "merchant",
            "message": "stop messaging me",
            "received_at": "2025-01-01T12:00:00Z",
            "turn_number": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "end"


@pytest.mark.asyncio
async def test_reply_auto_reply_escalation(client: AsyncClient):
    """Three consecutive auto-replies escalate to end."""
    for i in range(1, 4):
        resp = await client.post(
            "/v1/reply",
            json={
                "conversation_id": "conv-auto",
                "merchant_id": "m1",
                "customer_id": None,
                "from_role": "merchant",
                "message": "Thank you for contacting us. Our team will respond shortly.",
                "received_at": f"2025-01-01T12:0{i}:00Z",
                "turn_number": i,
            },
        )
        assert resp.status_code == 200

    # The third auto-reply should end the conversation
    data = resp.json()
    assert data["action"] == "end"
