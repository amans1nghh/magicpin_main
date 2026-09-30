"""Unit tests for the AntiPatternValidator class."""

from __future__ import annotations

import pytest

from src.models import ComposedMessage
from src.validators import AntiPatternValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(body: str = "Hello there", **overrides) -> ComposedMessage:
    """Create a ComposedMessage with sensible defaults."""
    defaults = dict(
        body=body,
        cta="binary_yes_no",
        send_as="vera",
        suppression_key="sk_1",
        rationale="test",
        template_name="test_template",
        template_params=[],
    )
    defaults.update(overrides)
    return ComposedMessage(**defaults)


def _category(
    vocab_taboo: list[str] | None = None,
    offer_catalog: list[dict] | None = None,
) -> dict:
    """Build a minimal category dict with voice and offer_catalog."""
    cat: dict = {
        "voice": {
            "vocab_taboo": vocab_taboo or [],
        },
    }
    if offer_catalog is not None:
        cat["offer_catalog"] = offer_catalog
    return cat


@pytest.fixture
def validator() -> AntiPatternValidator:
    return AntiPatternValidator()


# ---------------------------------------------------------------------------
# Check 1: Taboo vocabulary
# ---------------------------------------------------------------------------


class TestTabooVocabulary:
    def test_detects_taboo_word(self, validator: AntiPatternValidator) -> None:
        msg = _msg("This is guaranteed to work!")
        cat = _category(vocab_taboo=["guaranteed"])
        violations = validator.validate(msg, cat, set())
        assert any(v.startswith("taboo_vocabulary:") for v in violations)

    def test_taboo_case_insensitive(self, validator: AntiPatternValidator) -> None:
        msg = _msg("This is GUARANTEED to work!")
        cat = _category(vocab_taboo=["guaranteed"])
        violations = validator.validate(msg, cat, set())
        assert any("guaranteed" in v for v in violations)

    def test_taboo_substring_match(self, validator: AntiPatternValidator) -> None:
        msg = _msg("This is 100% safe for everyone")
        cat = _category(vocab_taboo=["100% safe"])
        violations = validator.validate(msg, cat, set())
        assert any("100% safe" in v for v in violations)

    def test_no_taboo_when_absent(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Dental Cleaning @ ₹299 available now")
        cat = _category(vocab_taboo=["guaranteed", "miracle"])
        violations = validator.validate(msg, cat, set())
        assert not any(v.startswith("taboo_vocabulary:") for v in violations)

    def test_empty_taboo_list(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Anything goes here guaranteed miracle")
        cat = _category(vocab_taboo=[])
        violations = validator.validate(msg, cat, set())
        assert not any(v.startswith("taboo_vocabulary:") for v in violations)


# ---------------------------------------------------------------------------
# Check 2: Multiple CTAs
# ---------------------------------------------------------------------------


class TestMultipleCTAs:
    def test_detects_multiple_ctas(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Reply to confirm. Want me to draft it? Should I proceed?")
        violations = validator.validate(msg, _category(), set())
        assert "multiple_ctas" in violations

    def test_two_cta_patterns(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Reply YES to start. Should I send the details?")
        violations = validator.validate(msg, _category(), set())
        assert "multiple_ctas" in violations

    def test_single_cta_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Here is the plan. Reply YES to confirm.")
        violations = validator.validate(msg, _category(), set())
        assert "multiple_ctas" not in violations

    def test_no_cta_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Dental Cleaning @ ₹299 is now available.")
        violations = validator.validate(msg, _category(), set())
        assert "multiple_ctas" not in violations


# ---------------------------------------------------------------------------
# Check 3: CTA not at end
# ---------------------------------------------------------------------------


class TestCTAPosition:
    def test_cta_at_end_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Great news about your clinic. Reply YES to confirm.")
        violations = validator.validate(msg, _category(), set())
        assert "cta_not_at_end" not in violations

    def test_cta_not_at_end(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Reply YES to confirm. Here are the details of the plan.")
        violations = validator.validate(msg, _category(), set())
        assert "cta_not_at_end" in violations

    def test_no_cta_no_violation(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Here is the information you requested.")
        violations = validator.validate(msg, _category(), set())
        assert "cta_not_at_end" not in violations


# ---------------------------------------------------------------------------
# Check 4: URLs in body
# ---------------------------------------------------------------------------


class TestURLsInBody:
    def test_detects_http_url(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Check http://example.com for details.")
        violations = validator.validate(msg, _category(), set())
        assert "url_in_body" in violations

    def test_detects_https_url(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Visit https://magicpin.in/offers for more.")
        violations = validator.validate(msg, _category(), set())
        assert "url_in_body" in violations

    def test_detects_www_url(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Go to www.example.com for info.")
        violations = validator.validate(msg, _category(), set())
        assert "url_in_body" in violations

    def test_no_url_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Dental Cleaning @ ₹299 available now.")
        violations = validator.validate(msg, _category(), set())
        assert "url_in_body" not in violations


# ---------------------------------------------------------------------------
# Check 5: Long preambles
# ---------------------------------------------------------------------------


class TestLongPreambles:
    def test_detects_hope_doing_well(self, validator: AntiPatternValidator) -> None:
        msg = _msg("I hope you're doing well. Here is the update.")
        violations = validator.validate(msg, _category(), set())
        assert any(v.startswith("long_preamble:") for v in violations)

    def test_detects_good_morning(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Good morning! Here is your report.")
        violations = validator.validate(msg, _category(), set())
        assert any(v.startswith("long_preamble:") for v in violations)

    def test_detects_reaching_out(self, validator: AntiPatternValidator) -> None:
        msg = _msg("I'm reaching out today to share some news.")
        violations = validator.validate(msg, _category(), set())
        assert any(v.startswith("long_preamble:") for v in violations)

    def test_no_preamble_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Your clinic's views dropped 18% this week.")
        violations = validator.validate(msg, _category(), set())
        assert not any(v.startswith("long_preamble:") for v in violations)


# ---------------------------------------------------------------------------
# Check 6: Duplicate body
# ---------------------------------------------------------------------------


class TestDuplicateBody:
    def test_detects_duplicate(self, validator: AntiPatternValidator) -> None:
        body = "Dental Cleaning @ ₹299 available now."
        msg = _msg(body)
        sent = {body}
        violations = validator.validate(msg, _category(), sent)
        assert "duplicate_body" in violations

    def test_no_duplicate_ok(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Dental Cleaning @ ₹299 available now.")
        sent = {"Some other message."}
        violations = validator.validate(msg, _category(), sent)
        assert "duplicate_body" not in violations

    def test_empty_sent_bodies(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Any message here.")
        violations = validator.validate(msg, _category(), set())
        assert "duplicate_body" not in violations


# ---------------------------------------------------------------------------
# Check 7: Generic discount format
# ---------------------------------------------------------------------------


class TestGenericDiscount:
    def test_detects_flat_percent_off(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Get flat 30% off on all services!")
        cat = _category(
            offer_catalog=[
                {"id": "o1", "title": "Cleaning @ ₹299", "type": "service_at_price"}
            ]
        )
        violations = validator.validate(msg, cat, set())
        assert "generic_discount" in violations

    def test_detects_percent_discount(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Enjoy 20% discount on your next visit.")
        cat = _category(
            offer_catalog=[
                {"id": "o1", "title": "Cleaning @ ₹299", "type": "service_at_price"}
            ]
        )
        violations = validator.validate(msg, cat, set())
        assert "generic_discount" in violations

    def test_no_violation_without_service_at_price(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg("Get flat 30% off on all services!")
        cat = _category(
            offer_catalog=[
                {"id": "o1", "title": "Free Consultation", "type": "free_service"}
            ]
        )
        violations = validator.validate(msg, cat, set())
        assert "generic_discount" not in violations

    def test_no_violation_without_generic_discount(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg("Dental Cleaning @ ₹299 — book your slot.")
        cat = _category(
            offer_catalog=[
                {"id": "o1", "title": "Cleaning @ ₹299", "type": "service_at_price"}
            ]
        )
        violations = validator.validate(msg, cat, set())
        assert "generic_discount" not in violations

    def test_no_offer_catalog(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Get flat 30% off on all services!")
        cat = _category()  # no offer_catalog key
        violations = validator.validate(msg, cat, set())
        assert "generic_discount" not in violations


# ---------------------------------------------------------------------------
# Fix method
# ---------------------------------------------------------------------------


class TestFix:
    def test_fix_removes_urls(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Check https://example.com for details. Book now.")
        fixed = validator.fix(msg, ["url_in_body"])
        assert "https://example.com" not in fixed.body
        assert "Book now" in fixed.body

    def test_fix_removes_www_url(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Visit www.example.com today. Great offers.")
        fixed = validator.fix(msg, ["url_in_body"])
        assert "www.example.com" not in fixed.body

    def test_fix_trims_preamble(self, validator: AntiPatternValidator) -> None:
        msg = _msg("I hope you're doing well. Your clinic views dropped 18%.")
        fixed = validator.fix(msg, ["long_preamble:I hope you're doing well"])
        assert "hope you're doing well" not in fixed.body.lower()
        assert "clinic views" in fixed.body

    def test_fix_trims_good_morning(self, validator: AntiPatternValidator) -> None:
        msg = _msg("Good morning. Here is your weekly report.")
        fixed = validator.fix(msg, ["long_preamble:Good morning"])
        assert "good morning" not in fixed.body.lower()
        assert "weekly report" in fixed.body

    def test_fix_returns_same_for_unfixable(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg("This is guaranteed to work!")
        fixed = validator.fix(msg, ["taboo_vocabulary:guaranteed"])
        assert fixed.body == msg.body

    def test_fix_returns_same_for_no_violations(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg("Clean message here.")
        fixed = validator.fix(msg, [])
        assert fixed.body == msg.body

    def test_fix_handles_multiple_violations(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg(
            "I hope you're doing well. Visit https://example.com for details."
        )
        fixed = validator.fix(
            msg,
            ["long_preamble:I hope you're doing well", "url_in_body"],
        )
        assert "hope you're doing well" not in fixed.body.lower()
        assert "https://example.com" not in fixed.body


# ---------------------------------------------------------------------------
# Clean message — no violations
# ---------------------------------------------------------------------------


class TestCleanMessage:
    def test_clean_message_passes_all_checks(
        self, validator: AntiPatternValidator
    ) -> None:
        msg = _msg("Dental Cleaning @ ₹299 — your slot is ready. Reply YES to book.")
        cat = _category(
            vocab_taboo=["guaranteed", "miracle"],
            offer_catalog=[
                {"id": "o1", "title": "Cleaning @ ₹299", "type": "service_at_price"}
            ],
        )
        violations = validator.validate(msg, cat, set())
        assert violations == []
