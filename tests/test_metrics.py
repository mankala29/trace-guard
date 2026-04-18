"""
tests/test_metrics.py
Unit tests for the three custom DeepEval metrics.
"""

import json

import pytest
from deepeval.test_case import LLMTestCase

from metrics.deepeval_metrics import (
    ContextSensitivityMetric,
    FraudAccuracyMetric,
    ReasoningFidelityMetric,
)


def _case(actual: str, expected: str = "FRAUD") -> LLMTestCase:
    return LLMTestCase(input="transcript", actual_output=actual, expected_output=expected)


# ── FraudAccuracyMetric ────────────────────────────────────────────────────────

class TestFraudAccuracyMetric:
    def test_correct_fraud(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: Caller demanded gift cards.", "FRAUD"))
        assert m.score == 1.0
        assert m.is_successful()

    def test_correct_not_fraud(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: NOT_FRAUD\nREASON: Routine reminder.", "NOT_FRAUD"))
        assert m.score == 1.0

    def test_wrong_prediction(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: NOT_FRAUD\nREASON: Seems fine.", "FRAUD"))
        assert m.score == 0.0
        assert not m.is_successful()

    def test_false_negative(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: Suspicious urgency.", "NOT_FRAUD"))
        assert m.score == 0.0

    def test_case_insensitive_match(self):
        m = FraudAccuracyMetric()
        m.measure(_case("prediction: fraud\nreason: something.", "FRAUD"))
        assert m.score == 1.0

    def test_reason_populated_on_success(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: Wire transfer.", "FRAUD"))
        assert "FRAUD" in m.reason

    def test_reason_populated_on_failure(self):
        m = FraudAccuracyMetric()
        m.measure(_case("PREDICTION: NOT_FRAUD\nREASON: Benign.", "FRAUD"))
        assert m.reason != ""


# ── ReasoningFidelityMetric ────────────────────────────────────────────────────

class TestReasoningFidelityMetric:
    def test_keyword_present(self):
        m = ReasoningFidelityMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: The caller created urgency and threatened arrest."))
        assert m.score == 1.0
        assert m.is_successful()

    def test_no_keyword(self):
        m = ReasoningFidelityMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: This looks like a bad call."))
        assert m.score == 0.0
        assert not m.is_successful()

    @pytest.mark.parametrize("keyword", [
        "threat", "urgency", "verify", "account", "pressure", "suspicious",
        "wire", "gift card", "credential", "impersonat", "isolation",
        "fee", "arrest", "password", "code", "intercept", "pretense", "fake",
    ])
    def test_each_signal_keyword_triggers(self, keyword):
        m = ReasoningFidelityMetric()
        m.measure(_case(f"PREDICTION: FRAUD\nREASON: The call involved {keyword} behavior."))
        assert m.score == 1.0, f"Keyword '{keyword}' should trigger ReasoningFidelityMetric"

    def test_matched_keywords_in_reason(self):
        m = ReasoningFidelityMetric()
        m.measure(_case("PREDICTION: FRAUD\nREASON: Caller applied pressure and threatened arrest."))
        assert "pressure" in m.reason or "arrest" in m.reason


# ── ContextSensitivityMetric ───────────────────────────────────────────────────

def _sensitivity_case(preds: dict, expected: str = "FRAUD") -> LLMTestCase:
    return LLMTestCase(
        input="Transcript T001",
        actual_output=json.dumps(preds),
        expected_output=expected,
    )


class TestContextSensitivityMetric:
    def test_all_agree_score_1(self):
        m = ContextSensitivityMetric()
        m.measure(_sensitivity_case({"bad": "FRAUD", "medium": "FRAUD", "good": "FRAUD"}))
        assert m.score == 1.0
        assert m.is_successful()

    def test_two_agree_score_half(self):
        m = ContextSensitivityMetric()
        m.measure(_sensitivity_case({"bad": "FRAUD", "medium": "NOT_FRAUD", "good": "FRAUD"}))
        assert m.score == 0.5
        assert not m.is_successful()

    def test_minority_context_named_in_reason(self):
        m = ContextSensitivityMetric()
        m.measure(_sensitivity_case({"bad": "FRAUD", "medium": "NOT_FRAUD", "good": "FRAUD"}))
        assert "medium" in m.reason

    def test_all_not_fraud_agree(self):
        m = ContextSensitivityMetric()
        m.measure(_sensitivity_case(
            {"bad": "NOT_FRAUD", "medium": "NOT_FRAUD", "good": "NOT_FRAUD"},
            expected="NOT_FRAUD",
        ))
        assert m.score == 1.0

    def test_invalid_json_score_0(self):
        m = ContextSensitivityMetric()
        tc = LLMTestCase(input="T001", actual_output="not json", expected_output="FRAUD")
        m.measure(tc)
        assert m.score == 0.0

    def test_empty_output_score_0(self):
        m = ContextSensitivityMetric()
        tc = LLMTestCase(input="T001", actual_output="", expected_output="FRAUD")
        m.measure(tc)
        assert m.score == 0.0

    def test_custom_threshold(self):
        m = ContextSensitivityMetric(threshold=0.5)
        m.measure(_sensitivity_case({"bad": "FRAUD", "medium": "NOT_FRAUD", "good": "FRAUD"}))
        assert m.is_successful()  # score=0.5 meets threshold=0.5
