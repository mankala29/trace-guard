"""
tests/test_context_builder.py
Unit tests for context_builder.py — pure functions, no API calls.
"""

import pytest

from context_builder import (
    _detect_signals,
    _dialogue_to_text,
    build_all_contexts,
    build_bad_context,
    build_good_context,
    build_medium_context,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

FRAUD_TRANSCRIPT = {
    "id": "T001",
    "label": "fraud",
    "fraud_type": "bank_impersonation",
    "dialogue": [
        {"speaker": "Caller", "text": "This is Chase Bank. We detected suspicious activity."},
        {"speaker": "Recipient", "text": "Oh no, what happened?"},
        {"speaker": "Caller", "text": "Please verify your account password immediately to stop the transfer."},
        {"speaker": "Recipient", "text": "Okay my password is..."},
        {"speaker": "Caller", "text": "Also read me the one-time code we just sent."},
    ],
}

BENIGN_TRANSCRIPT = {
    "id": "T016",
    "label": "not_fraud",
    "fraud_type": None,
    "dialogue": [
        {"speaker": "Caller", "text": "Hi, this is a reminder for your appointment tomorrow at 10 AM."},
        {"speaker": "Recipient", "text": "Thanks, I'll be there."},
        {"speaker": "Caller", "text": "See you then. Goodbye."},
    ],
}


# ── _dialogue_to_text ──────────────────────────────────────────────────────────

class TestDialogueToText:
    def test_format(self):
        dialogue = [{"speaker": "Caller", "text": "Hello."}, {"speaker": "Recipient", "text": "Hi."}]
        result = _dialogue_to_text(dialogue)
        assert result == "Caller: Hello.\nRecipient: Hi."

    def test_single_turn(self):
        dialogue = [{"speaker": "Caller", "text": "Testing."}]
        assert _dialogue_to_text(dialogue) == "Caller: Testing."


# ── build_bad_context ──────────────────────────────────────────────────────────

class TestBuildBadContext:
    def test_truncates_to_three_turns(self):
        result = build_bad_context(FRAUD_TRANSCRIPT)
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_starts_with_caller(self):
        result = build_bad_context(FRAUD_TRANSCRIPT)
        assert result.startswith("Caller:")

    def test_short_transcript_not_truncated(self):
        result = build_bad_context(BENIGN_TRANSCRIPT)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # exactly 3 turns in BENIGN_TRANSCRIPT


# ── build_medium_context ───────────────────────────────────────────────────────

class TestBuildMediumContext:
    def test_includes_all_turns(self):
        result = build_medium_context(FRAUD_TRANSCRIPT)
        lines = result.strip().split("\n")
        assert len(lines) == len(FRAUD_TRANSCRIPT["dialogue"])

    def test_contains_full_dialogue(self):
        result = build_medium_context(FRAUD_TRANSCRIPT)
        assert "one-time code" in result

    def test_bad_is_subset_of_medium(self):
        bad = build_bad_context(FRAUD_TRANSCRIPT)
        medium = build_medium_context(FRAUD_TRANSCRIPT)
        assert medium.startswith(bad)


# ── build_good_context ─────────────────────────────────────────────────────────

class TestBuildGoodContext:
    def test_contains_required_sections(self):
        result = build_good_context(FRAUD_TRANSCRIPT)
        assert "INTENT:" in result
        assert "KEY SIGNALS:" in result
        assert "FULL TRANSCRIPT:" in result
        assert "SUMMARY:" in result

    def test_full_transcript_included(self):
        result = build_good_context(FRAUD_TRANSCRIPT)
        assert "one-time code" in result

    def test_signals_non_empty(self):
        result = build_good_context(FRAUD_TRANSCRIPT)
        # At least one bullet point signal
        assert "  - " in result

    def test_benign_has_no_strong_signals(self):
        result = build_good_context(BENIGN_TRANSCRIPT)
        assert "No strong fraud signals detected" in result


# ── _detect_signals ────────────────────────────────────────────────────────────

class TestDetectSignals:
    def test_detects_credential_request(self):
        dialogue = [{"speaker": "Caller", "text": "Please confirm your account password."}]
        signals = _detect_signals(dialogue)
        assert any("credentials" in s.lower() or "account" in s.lower() for s in signals)

    def test_detects_untraceable_payment(self):
        dialogue = [{"speaker": "Caller", "text": "Pay via Zelle immediately."}]
        signals = _detect_signals(dialogue)
        assert any("payment" in s.lower() for s in signals)

    def test_detects_urgency(self):
        dialogue = [{"speaker": "Caller", "text": "You must act now or your account expires today."}]
        signals = _detect_signals(dialogue)
        assert any("urgency" in s.lower() or "pressure" in s.lower() for s in signals)

    def test_detects_isolation(self):
        dialogue = [{"speaker": "Caller", "text": "Don't tell anyone about this call."}]
        signals = _detect_signals(dialogue)
        assert any("secret" in s.lower() or "isolation" in s.lower() for s in signals)

    def test_detects_2fa_interception(self):
        dialogue = [{"speaker": "Caller", "text": "Please read me the one-time code we sent."}]
        signals = _detect_signals(dialogue)
        assert any("authentication" in s.lower() or "code" in s.lower() for s in signals)

    def test_no_signals_benign(self):
        dialogue = [{"speaker": "Caller", "text": "Your appointment is tomorrow at 10 AM."}]
        signals = _detect_signals(dialogue)
        assert signals == ["No strong fraud signals detected"]


# ── build_all_contexts ─────────────────────────────────────────────────────────

class TestBuildAllContexts:
    def test_returns_all_three_keys(self):
        result = build_all_contexts(FRAUD_TRANSCRIPT)
        assert set(result.keys()) == {"bad_context", "medium_context", "good_context"}

    def test_each_value_is_nonempty_string(self):
        result = build_all_contexts(FRAUD_TRANSCRIPT)
        for ctx_type, ctx_text in result.items():
            assert isinstance(ctx_text, str) and len(ctx_text) > 0, f"{ctx_type} is empty"

    def test_good_longer_than_medium(self):
        result = build_all_contexts(FRAUD_TRANSCRIPT)
        assert len(result["good_context"]) > len(result["medium_context"])

    def test_medium_longer_than_bad(self):
        result = build_all_contexts(FRAUD_TRANSCRIPT)
        assert len(result["medium_context"]) >= len(result["bad_context"])
