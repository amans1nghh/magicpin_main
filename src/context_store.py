"""In-memory versioned context store for Vera Merchant Bot.

Provides thread-safe (asyncio-safe) storage of contexts keyed by (scope, context_id)
with strict version ordering. Reads are lock-free; writes are serialized via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from datetime import datetime, timezone

from src.config import VALID_SCOPES
from src.models import PutResult, StoredContext


class ContextStore:
    """Thread-safe in-memory context store with version tracking."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], StoredContext] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def put(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: dict,
        delivered_at: str,
    ) -> PutResult:
        """Atomically store a context if version is strictly higher than current.

        Returns a PutResult indicating acceptance or rejection with reason.
        """
        if scope not in VALID_SCOPES:
            return PutResult(accepted=False, reason="invalid_scope")

        async with self._lock:
            key = (scope, context_id)
            existing = self._store.get(key)

            if existing is not None and version <= existing.version:
                return PutResult(
                    accepted=False,
                    reason="stale_version",
                    current_version=existing.version,
                )

            stored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ack_id = str(uuid.uuid4())

            self._store[key] = StoredContext(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload,
                delivered_at=delivered_at,
                stored_at=stored_at,
            )

            return PutResult(
                accepted=True,
                ack_id=ack_id,
                stored_at=stored_at,
            )

    def get(self, scope: str, context_id: str) -> StoredContext | None:
        """Retrieve the latest version of a context. Lock-free read."""
        return self._store.get((scope, context_id))

    def get_all_by_scope(self, scope: str) -> list[StoredContext]:
        """Return all contexts for a given scope."""
        return [ctx for key, ctx in self._store.items() if key[0] == scope]

    def count_by_scope(self) -> dict[str, int]:
        """Return counts of distinct context_ids per scope.

        Always includes all valid scopes, even if count is zero.
        """
        counts: Counter[str] = Counter()
        for scope, _ in self._store:
            counts[scope] += 1
        # Ensure all valid scopes appear in the result
        return {s: counts.get(s, 0) for s in sorted(VALID_SCOPES)}
