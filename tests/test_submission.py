"""Tests for the SubmissionGenerator class.

Validates that the generator correctly loads expanded datasets,
retrieves contexts for each test pair, invokes the Composer, and
writes well-formed JSON lines to the output file.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.composer import Composer
from src.context_store import ContextStore
from src.llm_client import LLMClient
from src.models import ComposedMessage
from src.submission import SubmissionGenerator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def context_store() -> ContextStore:
    return ContextStore()


@pytest.fixture
def mock_llm() -> LLMClient:
    """LLM client that never makes real API calls."""
    return LLMClient(provider="openai", api_key="test-key", model="test-model")


@pytest.fixture
def composer(mock_llm: LLMClient, context_store: ContextStore) -> Composer:
    return Composer(llm_client=mock_llm, context_store=context_store)


@pytest.fixture
def generator(context_store: ContextStore, composer: Composer) -> SubmissionGenerator:
    return SubmissionGenerator(context_store=context_store, composer=composer)


# ------------------------------------------------------------------
# Helper: build a minimal expanded dataset in a temp directory
# ------------------------------------------------------------------

def _build_mini_dataset(tmp: Path) -> Path:
    """Create a minimal expanded dataset with 1 category, 1 merchant,
    1 customer, 1 trigger, and 2 test pairs."""
    (tmp / "categories").mkdir()
    (tmp / "merchants").mkdir()
    (tmp / "customers").mkdir()
    (tmp / "triggers").mkdir()

    # Category
    cat = {
        "slug": "dentists",
        "voice": {"tone": "professional", "vocab_allowed": [], "vocab_taboo": []},
        "peer_stats": {},
        "digest": [],
        "seasonal_beats": [],
        "trend_signals": [],
    }
    (tmp / "categories" / "dentists.json").write_text(json.dumps(cat))

    # Merchant
    merchant = {
        "merchant_id": "m_001_test",
        "category_slug": "dentists",
        "identity": {
            "name": "Test Dental",
            "owner_first_name": "Dr. Test",
            "city": "Delhi",
            "locality": "Saket",
            "languages": ["en"],
        },
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 100},
        "performance": {"window_days": 30, "views": 1000, "calls": 50, "ctr": 0.05, "delta_7d": {}},
        "offers": [],
        "conversation_history": [],
        "customer_aggregate": {},
        "signals": [],
        "review_themes": [],
    }
    (tmp / "merchants" / "m_001_test.json").write_text(json.dumps(merchant))

    # Customer
    customer = {
        "customer_id": "c_001_test",
        "merchant_id": "m_001_test",
        "identity": {"name": "Priya", "language_pref": "en", "age_band": "25-35"},
        "relationship": {"visits_total": 3, "last_visit": "2026-03-01", "services_received": []},
        "state": "active",
        "preferences": {"channel": "whatsapp"},
        "consent": {"scope": ["promotional_offers"]},
    }
    (tmp / "customers" / "c_001_test.json").write_text(json.dumps(customer))

    # Triggers
    trigger_merchant = {
        "id": "trg_001_test",
        "kind": "perf_dip",
        "scope": "merchant",
        "source": "internal",
        "merchant_id": "m_001_test",
        "customer_id": None,
        "urgency": 3,
        "suppression_key": "perf_dip:m_001_test:1",
        "expires_at": "2026-06-30T00:00:00Z",
        "payload": {"metric_or_topic": "perf_dip"},
    }
    (tmp / "triggers" / "trg_001_test.json").write_text(json.dumps(trigger_merchant))

    trigger_customer = {
        "id": "trg_002_test",
        "kind": "recall_due",
        "scope": "customer",
        "source": "internal",
        "merchant_id": "m_001_test",
        "customer_id": "c_001_test",
        "urgency": 2,
        "suppression_key": "recall_due:m_001_test:2",
        "expires_at": "2026-06-30T00:00:00Z",
        "payload": {"metric_or_topic": "recall_due"},
    }
    (tmp / "triggers" / "trg_002_test.json").write_text(json.dumps(trigger_customer))

    # Test pairs
    pairs = {
        "pairs": [
            {"test_id": "T01", "trigger_id": "trg_001_test", "merchant_id": "m_001_test", "customer_id": None},
            {"test_id": "T02", "trigger_id": "trg_002_test", "merchant_id": "m_001_test", "customer_id": "c_001_test"},
        ]
    }
    (tmp / "test_pairs.json").write_text(json.dumps(pairs))

    return tmp


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_expanded_dataset(generator: SubmissionGenerator, context_store: ContextStore):
    """load_expanded_dataset should push all files into the context store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _build_mini_dataset(tmp)

        await generator.load_expanded_dataset(str(tmp))

        counts = context_store.count_by_scope()
        assert counts["category"] == 1
        assert counts["merchant"] == 1
        assert counts["customer"] == 1
        assert counts["trigger"] == 2


@pytest.mark.asyncio
async def test_generate_writes_correct_lines(generator: SubmissionGenerator, context_store: ContextStore):
    """generate() should write one JSON line per test pair with all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _build_mini_dataset(tmp)

        # Load dataset into store
        await generator.load_expanded_dataset(str(tmp))

        # Mock the Composer.compose method to return deterministic results
        async def mock_compose(category, merchant, trigger, customer=None, **kwargs):
            return ComposedMessage(
                body=f"Test body for {trigger.get('id', 'unknown')}",
                cta="binary_yes_no",
                send_as="merchant_on_behalf" if customer else "vera",
                suppression_key=trigger.get("suppression_key", ""),
                rationale="Test rationale",
                template_name="test_template",
                template_params=[],
            )

        with patch.object(generator._composer, "compose", side_effect=mock_compose):
            output_path = str(tmp / "submission.jsonl")
            await generator.generate(
                test_pairs_path=str(tmp / "test_pairs.json"),
                output_path=output_path,
            )

            # Read and verify output
            with open(output_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]

            assert len(lines) == 2

            # Verify required fields present in each line
            required_fields = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
            for line in lines:
                assert required_fields.issubset(line.keys()), f"Missing fields in {line}"

            # Verify first pair (merchant-scoped)
            assert lines[0]["test_id"] == "T01"
            assert lines[0]["send_as"] == "vera"
            assert "trg_001_test" in lines[0]["body"]

            # Verify second pair (customer-scoped)
            assert lines[1]["test_id"] == "T02"
            assert lines[1]["send_as"] == "merchant_on_behalf"
            assert "trg_002_test" in lines[1]["body"]


@pytest.mark.asyncio
async def test_generate_handles_missing_trigger(generator: SubmissionGenerator, context_store: ContextStore):
    """generate() should skip pairs where the trigger is not in the store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _build_mini_dataset(tmp)

        await generator.load_expanded_dataset(str(tmp))

        # Create test pairs referencing a non-existent trigger
        pairs = {
            "pairs": [
                {"test_id": "T99", "trigger_id": "trg_nonexistent", "merchant_id": "m_001_test", "customer_id": None},
            ]
        }
        pairs_path = str(tmp / "test_pairs_bad.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f)

        output_path = str(tmp / "submission.jsonl")
        # Should not raise — just skip the bad pair
        await generator.generate(test_pairs_path=pairs_path, output_path=output_path)

        with open(output_path) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 0


@pytest.mark.asyncio
async def test_load_expanded_dataset_with_real_data(context_store: ContextStore):
    """Verify load_expanded_dataset works with the actual expanded dataset."""
    expanded_dir = "dataset/expanded"
    if not Path(expanded_dir).is_dir():
        pytest.skip("Expanded dataset not available")

    llm = LLMClient(provider="openai", api_key="test", model="test")
    composer = Composer(llm_client=llm, context_store=context_store)
    gen = SubmissionGenerator(context_store=context_store, composer=composer)

    await gen.load_expanded_dataset(expanded_dir)

    counts = context_store.count_by_scope()
    assert counts["category"] == 5
    assert counts["merchant"] == 50
    assert counts["customer"] == 200
    assert counts["trigger"] == 100
