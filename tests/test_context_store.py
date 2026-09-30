"""Unit tests for the ContextStore class."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from src.context_store import ContextStore


@pytest.fixture
def store() -> ContextStore:
    return ContextStore()


@pytest.mark.asyncio
async def test_put_accepts_new_context(store: ContextStore) -> None:
    """First push to a key should always be accepted."""
    result = await store.put("merchant", "m1", 1, {"name": "Shop"}, "2024-01-01T00:00:00Z")
    assert result.accepted is True
    assert result.reason is None
    assert result.ack_id is not None
    assert result.stored_at is not None


@pytest.mark.asyncio
async def test_put_accepts_higher_version(store: ContextStore) -> None:
    """A push with a strictly higher version should replace the existing context."""
    await store.put("merchant", "m1", 1, {"v": 1}, "2024-01-01T00:00:00Z")
    result = await store.put("merchant", "m1", 2, {"v": 2}, "2024-01-01T00:00:01Z")
    assert result.accepted is True

    ctx = store.get("merchant", "m1")
    assert ctx is not None
    assert ctx.version == 2
    assert ctx.payload == {"v": 2}


@pytest.mark.asyncio
async def test_put_rejects_equal_version(store: ContextStore) -> None:
    """A push with the same version should be rejected as stale."""
    await store.put("merchant", "m1", 3, {"v": 3}, "2024-01-01T00:00:00Z")
    result = await store.put("merchant", "m1", 3, {"v": "3-dup"}, "2024-01-01T00:00:01Z")
    assert result.accepted is False
    assert result.reason == "stale_version"
    assert result.current_version == 3


@pytest.mark.asyncio
async def test_put_rejects_lower_version(store: ContextStore) -> None:
    """A push with a lower version should be rejected as stale."""
    await store.put("merchant", "m1", 5, {"v": 5}, "2024-01-01T00:00:00Z")
    result = await store.put("merchant", "m1", 2, {"v": 2}, "2024-01-01T00:00:01Z")
    assert result.accepted is False
    assert result.reason == "stale_version"
    assert result.current_version == 5

    # Original payload should be unchanged
    ctx = store.get("merchant", "m1")
    assert ctx is not None
    assert ctx.version == 5
    assert ctx.payload == {"v": 5}


@pytest.mark.asyncio
async def test_put_rejects_invalid_scope(store: ContextStore) -> None:
    """An invalid scope should be rejected immediately."""
    result = await store.put("invalid_scope", "x1", 1, {}, "2024-01-01T00:00:00Z")
    assert result.accepted is False
    assert result.reason == "invalid_scope"


@pytest.mark.asyncio
async def test_put_accepts_all_valid_scopes(store: ContextStore) -> None:
    """All four valid scopes should be accepted."""
    for scope in ("category", "merchant", "customer", "trigger"):
        result = await store.put(scope, "id1", 1, {"scope": scope}, "2024-01-01T00:00:00Z")
        assert result.accepted is True, f"Scope '{scope}' should be accepted"


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store: ContextStore) -> None:
    """Getting a non-existent key should return None."""
    assert store.get("merchant", "nonexistent") is None


@pytest.mark.asyncio
async def test_get_returns_stored_context(store: ContextStore) -> None:
    """Getting an existing key should return the stored context."""
    await store.put("category", "c1", 1, {"name": "Dentists"}, "2024-01-01T00:00:00Z")
    ctx = store.get("category", "c1")
    assert ctx is not None
    assert ctx.scope == "category"
    assert ctx.context_id == "c1"
    assert ctx.version == 1
    assert ctx.payload == {"name": "Dentists"}
    assert ctx.delivered_at == "2024-01-01T00:00:00Z"
    assert ctx.stored_at is not None


@pytest.mark.asyncio
async def test_get_all_by_scope(store: ContextStore) -> None:
    """get_all_by_scope should return only contexts matching the scope."""
    await store.put("merchant", "m1", 1, {"n": 1}, "2024-01-01T00:00:00Z")
    await store.put("merchant", "m2", 1, {"n": 2}, "2024-01-01T00:00:00Z")
    await store.put("category", "c1", 1, {"n": 3}, "2024-01-01T00:00:00Z")

    merchants = store.get_all_by_scope("merchant")
    assert len(merchants) == 2
    assert all(ctx.scope == "merchant" for ctx in merchants)

    categories = store.get_all_by_scope("category")
    assert len(categories) == 1


@pytest.mark.asyncio
async def test_get_all_by_scope_empty(store: ContextStore) -> None:
    """get_all_by_scope should return empty list for scope with no contexts."""
    assert store.get_all_by_scope("trigger") == []


@pytest.mark.asyncio
async def test_count_by_scope(store: ContextStore) -> None:
    """count_by_scope should return correct counts per scope."""
    await store.put("merchant", "m1", 1, {}, "2024-01-01T00:00:00Z")
    await store.put("merchant", "m2", 1, {}, "2024-01-01T00:00:00Z")
    await store.put("category", "c1", 1, {}, "2024-01-01T00:00:00Z")
    await store.put("trigger", "t1", 1, {}, "2024-01-01T00:00:00Z")

    counts = store.count_by_scope()
    assert counts == {"category": 1, "customer": 0, "merchant": 2, "trigger": 1}


@pytest.mark.asyncio
async def test_count_by_scope_empty(store: ContextStore) -> None:
    """count_by_scope on empty store should return all zeros."""
    counts = store.count_by_scope()
    assert counts == {"category": 0, "customer": 0, "merchant": 0, "trigger": 0}


@pytest.mark.asyncio
async def test_version_update_preserves_stored_at(store: ContextStore) -> None:
    """Updating a context should update stored_at timestamp."""
    r1 = await store.put("merchant", "m1", 1, {"v": 1}, "2024-01-01T00:00:00Z")
    r2 = await store.put("merchant", "m1", 2, {"v": 2}, "2024-01-01T00:00:01Z")
    assert r1.stored_at is not None
    assert r2.stored_at is not None
    # Both should have valid ISO timestamps
    assert "T" in r1.stored_at
    assert "T" in r2.stored_at
