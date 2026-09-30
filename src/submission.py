"""Submission Generator for Vera Merchant Bot.

Generates submission.jsonl from the expanded dataset's 30 canonical test pairs
by loading all contexts into the ContextStore and invoking the Composer for each pair.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.composer import Composer
from src.context_store import ContextStore

logger = logging.getLogger(__name__)

# 15-minute hard ceiling for the full generation run
MAX_GENERATION_SECONDS = 15 * 60


class SubmissionGenerator:
    """Generates submission.jsonl from expanded dataset test pairs.

    Loads all expanded dataset files into the ContextStore, then iterates
    over the 30 canonical test pairs, composing a message for each and
    writing the result as a JSON line.
    """

    def __init__(self, context_store: ContextStore, composer: Composer) -> None:
        self._store = context_store
        self._composer = composer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        test_pairs_path: str,
        output_path: str = "submission.jsonl",
    ) -> None:
        """Generate submission.jsonl for all test pairs.

        Steps:
        1. Load test_pairs.json
        2. For each pair: retrieve contexts, invoke Composer, write JSON line
        3. Track progress and respect the 15-minute budget
        """
        start = time.monotonic()

        # 1. Load test pairs
        pairs_file = Path(test_pairs_path)
        with open(pairs_file) as f:
            data = json.load(f)
        pairs: list[dict[str, Any]] = data["pairs"]
        logger.info("Loaded %d test pairs from %s", len(pairs), test_pairs_path)

        # 2. Process each pair
        lines_written = 0
        with open(output_path, "w", encoding="utf-8") as out:
            for idx, pair in enumerate(pairs):
                elapsed = time.monotonic() - start
                if elapsed >= MAX_GENERATION_SECONDS:
                    logger.warning(
                        "Time budget exhausted after %d/%d pairs (%.1fs)",
                        idx, len(pairs), elapsed,
                    )
                    break

                test_id = pair["test_id"]
                trigger_id = pair["trigger_id"]
                merchant_id = pair["merchant_id"]
                customer_id = pair.get("customer_id")

                logger.info(
                    "[%d/%d] Composing %s  trigger=%s  merchant=%s",
                    idx + 1, len(pairs), test_id, trigger_id, merchant_id,
                )

                try:
                    line = await self._compose_pair(
                        test_id, trigger_id, merchant_id, customer_id,
                    )
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    lines_written += 1
                except Exception:
                    logger.exception("Failed to compose pair %s", test_id)

        total = time.monotonic() - start
        logger.info(
            "Submission complete: %d lines written to %s in %.1fs",
            lines_written, output_path, total,
        )

    async def load_expanded_dataset(self, expanded_dir: str) -> None:
        """Load all expanded dataset files into the context store.

        Reads categories, merchants, customers, and triggers from the
        expanded directory and pushes each into the ContextStore with
        version=1.
        """
        base = Path(expanded_dir)

        # Categories
        cat_dir = base / "categories"
        if cat_dir.is_dir():
            for fp in sorted(cat_dir.glob("*.json")):
                payload = _load_json(fp)
                context_id = payload.get("slug", fp.stem)
                await self._store.put("category", context_id, 1, payload, _now_iso())
            logger.info("Loaded categories from %s", cat_dir)

        # Merchants
        merch_dir = base / "merchants"
        if merch_dir.is_dir():
            for fp in sorted(merch_dir.glob("*.json")):
                payload = _load_json(fp)
                context_id = payload.get("merchant_id", fp.stem)
                await self._store.put("merchant", context_id, 1, payload, _now_iso())
            logger.info("Loaded merchants from %s", merch_dir)

        # Customers
        cust_dir = base / "customers"
        if cust_dir.is_dir():
            for fp in sorted(cust_dir.glob("*.json")):
                payload = _load_json(fp)
                context_id = payload.get("customer_id", fp.stem)
                await self._store.put("customer", context_id, 1, payload, _now_iso())
            logger.info("Loaded customers from %s", cust_dir)

        # Triggers
        trig_dir = base / "triggers"
        if trig_dir.is_dir():
            for fp in sorted(trig_dir.glob("*.json")):
                payload = _load_json(fp)
                context_id = payload.get("id", fp.stem)
                await self._store.put("trigger", context_id, 1, payload, _now_iso())
            logger.info("Loaded triggers from %s", trig_dir)

        counts = self._store.count_by_scope()
        logger.info("Context store loaded: %s", counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compose_pair(
        self,
        test_id: str,
        trigger_id: str,
        merchant_id: str,
        customer_id: str | None,
    ) -> dict[str, Any]:
        """Compose a single test pair and return the submission line dict."""
        # Retrieve trigger context
        trigger_ctx = self._store.get("trigger", trigger_id)
        if trigger_ctx is None:
            raise ValueError(f"Trigger {trigger_id} not found in store")
        trigger = trigger_ctx.payload

        # Retrieve merchant context
        merchant_ctx = self._store.get("merchant", merchant_id)
        if merchant_ctx is None:
            raise ValueError(f"Merchant {merchant_id} not found in store")
        merchant = merchant_ctx.payload

        # Retrieve category context via merchant's category_slug
        category_slug = merchant.get("category_slug", "")
        category_ctx = self._store.get("category", category_slug)
        category = category_ctx.payload if category_ctx else {}

        # Optionally retrieve customer context
        customer: dict[str, Any] | None = None
        if customer_id:
            customer_ctx = self._store.get("customer", customer_id)
            if customer_ctx:
                customer = customer_ctx.payload

        # Compose via the Composer engine
        composed = await self._composer.compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )

        return {
            "test_id": test_id,
            "body": composed.body,
            "cta": composed.cta,
            "send_as": composed.send_as,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
