"""
deepeval_metrics.py
Custom DeepEval metrics that mirror the ContextTrace eval layer:

  FraudAccuracyMetric       — exact-match prediction vs ground-truth label
  ReasoningFidelityMetric   — keyword check: does the reason cite a fraud signal?
  ContextSensitivityMetric  — consistency across bad/medium/good context variants

Each is a deepeval BaseMetric so results plug into deepeval.evaluate() and
appear in the DeepEval dashboard / CLI report.
"""

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

REASONING_KEYWORDS = [
    "threat", "urgency", "verify", "account", "pressure", "suspicious",
    "wire", "gift card", "credential", "impersonat", "isolation", "coer",
    "fee", "arrest", "password", "code", "intercept", "pretense", "fake",
]


# ── 1. Fraud Accuracy ──────────────────────────────────────────────────────────

class FraudAccuracyMetric(BaseMetric):
    """
    Score 1.0 if the model's prediction matches the ground-truth label,
    0.0 otherwise.

    LLMTestCase contract:
        actual_output  — "PREDICTION: FRAUD"  or  "PREDICTION: NOT_FRAUD"
        expected_output — "FRAUD"  or  "NOT_FRAUD"
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = "Fraud Accuracy"
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raw = (test_case.actual_output or "").upper()
        expected = (test_case.expected_output or "").upper().strip()

        # Extract exact prediction token — avoids "FRAUD" matching inside "NOT_FRAUD"
        match = re.search(r"PREDICTION:\s*(NOT_FRAUD|FRAUD)", raw)
        predicted = match.group(1) if match else raw.strip()

        if predicted == expected:
            self.score = 1.0
            self.reason = f"Prediction matches ground truth: {expected}"
        else:
            self.score = 0.0
            self.reason = f"Predicted {predicted} but expected {expected}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# ── 2. Reasoning Fidelity ──────────────────────────────────────────────────────

class ReasoningFidelityMetric(BaseMetric):
    """
    Score 1.0 if the model's REASON field mentions at least one recognised
    fraud-signal keyword, 0.0 otherwise.

    LLMTestCase contract:
        actual_output — full model response containing "REASON: <text>"
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = "Reasoning Fidelity"
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        output = (test_case.actual_output or "").lower()

        matched = [kw for kw in REASONING_KEYWORDS if kw in output]
        if matched:
            self.score = 1.0
            self.reason = f"Reason cites signal keyword(s): {', '.join(matched[:3])}"
        else:
            self.score = 0.0
            self.reason = "Reason does not mention any recognised fraud signal"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# ── 3. Context Sensitivity ─────────────────────────────────────────────────────

class ContextSensitivityMetric(BaseMetric):
    """
    Measures whether a model's prediction is *stable* across context quality
    levels (bad / medium / good) for the same transcript.

    Score 1.0  → all three variants agree (consistent)
    Score 0.5  → two of three agree
    Score 0.0  → all three disagree (impossible with binary labels, kept for
                 extensibility if labels expand)

    LLMTestCase contract:
        actual_output — JSON string: {"bad": "FRAUD", "medium": "FRAUD", "good": "NOT_FRAUD"}
        expected_output — the ground-truth label (used only for logging)
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = "Context Sensitivity"
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            preds: dict = json.loads(test_case.actual_output or "{}")
        except (json.JSONDecodeError, TypeError):
            self.score = 0.0
            self.reason = "Could not parse predictions JSON"
            self.success = False
            return self.score

        values = list(preds.values())
        unique = set(values)

        if len(unique) == 1:
            self.score = 1.0
            self.reason = f"All context variants agree: {values[0]}"
        elif len(unique) == 2:
            majority = max(unique, key=values.count)
            minority_ctx = [k for k, v in preds.items() if v != majority]
            self.score = 0.5
            self.reason = (
                f"Majority prediction: {majority}. "
                f"Context variant(s) that flipped: {', '.join(minority_ctx)}"
            )
        else:
            self.score = 0.0
            self.reason = "All context variants disagree"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


# ── 4. Grounded Accuracy ───────────────────────────────────────────────────────

class GroundedAccuracyMetric(BaseMetric):
    """
    Score 1.0 only if the prediction is correct AND the reason cites at least
    one recognised fraud-signal keyword.

    Separates "right for the right reason" from "lucky correct" — a model that
    predicts correctly but reasons poorly will fail here even if FraudAccuracyMetric passes.

    Scores:
        1.0 — correct prediction + grounded reasoning
        0.5 — correct prediction but no signal keyword (ungrounded correct)
        0.0 — wrong prediction

    LLMTestCase contract:
        actual_output  — full model response with PREDICTION and REASON
        expected_output — "FRAUD" or "NOT_FRAUD"
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = "Grounded Accuracy"
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        raw = (test_case.actual_output or "").upper()
        expected = (test_case.expected_output or "").upper().strip()

        pred_match = re.search(r"PREDICTION:\s*(NOT_FRAUD|FRAUD)", raw)
        predicted = pred_match.group(1) if pred_match else ""
        correct = predicted == expected

        output_lower = (test_case.actual_output or "").lower()
        matched = [kw for kw in REASONING_KEYWORDS if kw in output_lower]

        if correct and matched:
            self.score = 1.0
            self.reason = f"Correct ({expected}) + signal(s): {', '.join(matched[:3])}"
        elif correct and not matched:
            self.score = 0.5
            self.reason = f"Correct ({expected}) but ungrounded — no fraud signal cited"
        else:
            self.score = 0.0
            self.reason = f"Wrong: predicted {predicted}, expected {expected}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

