"""
tests/test_pipeline.py
End-to-end pipeline tests with mocked classify_pass3.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluate import (
    CSV_COLUMNS,
    CONTEXT_TYPES,
    MODELS,
    _normalise_label,
    _reasoning_quality,
    run_pipeline,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPTS = [
    {
        "id": "T001",
        "label": "fraud",
        "fraud_type": "bank_impersonation",
        "dialogue": [
            {"speaker": "Caller", "text": "This is Chase Bank. Verify your password immediately."},
            {"speaker": "Recipient", "text": "My password is abc123."},
            {"speaker": "Caller", "text": "Now read me the one-time code."},
        ],
    },
    {
        "id": "T002",
        "label": "not_fraud",
        "fraud_type": None,
        "dialogue": [
            {"speaker": "Caller", "text": "Reminder: your appointment is tomorrow at 10 AM."},
            {"speaker": "Recipient", "text": "Thanks."},
        ],
    },
]

def _make_pass3_result(prediction: str, pass3: bool = True, reason: str = "Test reason."):
    trials = [{"prediction": prediction, "reason": reason}] * 3
    return {
        "prediction": prediction,
        "reason": reason,
        "raw_response": f"PREDICTION: {prediction}\nREASON: {reason}",
        "trials": trials,
        "pass3": pass3,
    }


# ── _normalise_label ───────────────────────────────────────────────────────────

class TestNormaliseLabel:
    def test_fraud(self):
        assert _normalise_label("fraud") == "FRAUD"

    def test_not_fraud(self):
        assert _normalise_label("not_fraud") == "NOT_FRAUD"

    def test_already_upper(self):
        assert _normalise_label("FRAUD") == "FRAUD"


# ── _reasoning_quality ─────────────────────────────────────────────────────────

class TestReasoningQuality:
    def test_keyword_match(self):
        assert _reasoning_quality("The caller created urgency and threatened arrest.") is True

    def test_no_match(self):
        assert _reasoning_quality("This seems like a normal call.") is False

    def test_case_insensitive(self):
        assert _reasoning_quality("THREAT detected in the call.") is True

    @pytest.mark.parametrize("kw", ["threat", "urgency", "verify", "account", "wire", "arrest", "password"])
    def test_each_keyword(self, kw):
        assert _reasoning_quality(f"The call showed {kw} behavior.") is True


# ── run_pipeline ───────────────────────────────────────────────────────────────

SINGLE_MODEL = ["open-mistral-nemo"]


@patch("evaluate.classify_pass3")
class TestRunPipeline:
    def test_row_count_single_model(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert len(rows) == len(SAMPLE_TRANSCRIPTS) * len(CONTEXT_TYPES)

    def test_row_count_multi_model(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        two_models = ["open-mistral-nemo", "mistral-small-2503"]
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=two_models)
        assert len(rows) == len(SAMPLE_TRANSCRIPTS) * len(CONTEXT_TYPES) * len(two_models)

    def test_csv_columns_present(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        for row in rows:
            for col in CSV_COLUMNS:
                assert col in row, f"Missing column: {col}"

    def test_model_column_populated(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert all(r["model"] == "open-mistral-nemo" for r in rows)

    def test_correct_true_when_prediction_matches_label(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        fraud_rows = [r for r in rows if r["transcript_id"] == "T001"]
        assert all(r["correct"] is True for r in fraud_rows)

    def test_correct_false_on_mismatch(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        not_fraud_rows = [r for r in rows if r["transcript_id"] == "T002"]
        assert all(r["correct"] is False for r in not_fraud_rows)

    def test_pass3_propagated(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD", pass3=True)
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert all(r["pass3"] is True for r in rows)

    def test_pass3_false_propagated(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD", pass3=False)
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert all(r["pass3"] is False for r in rows)

    def test_consistency_true_when_all_agree(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        t1_rows = [r for r in rows if r["transcript_id"] == "T001"]
        assert all(r["consistent"] is True for r in t1_rows)

    def test_consistency_false_when_predictions_differ(self, mock_classify):
        # 3 context types per transcript, alternating predictions
        mock_classify.side_effect = [
            _make_pass3_result("FRAUD"),
            _make_pass3_result("NOT_FRAUD"),
            _make_pass3_result("FRAUD"),
            _make_pass3_result("NOT_FRAUD"),
            _make_pass3_result("FRAUD"),
            _make_pass3_result("NOT_FRAUD"),
        ]
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        t1_rows = [r for r in rows if r["transcript_id"] == "T001"]
        assert all(r["consistent"] is False for r in t1_rows)

    def test_consistency_scoped_per_model(self, mock_classify):
        # model A: all agree; model B: disagree — consistency should differ per model
        two_models = ["open-mistral-nemo", "mistral-small-2503"]
        # T001: model A → F/F/F (consistent), model B → F/N/F (not)
        # T002: model A → N/N/N (consistent), model B → N/F/N (not)
        mock_classify.side_effect = [
            _make_pass3_result("FRAUD"),      # T001 model A bad
            _make_pass3_result("FRAUD"),      # T001 model A medium
            _make_pass3_result("FRAUD"),      # T001 model A good
            _make_pass3_result("NOT_FRAUD"),  # T002 model A bad
            _make_pass3_result("NOT_FRAUD"),  # T002 model A medium
            _make_pass3_result("NOT_FRAUD"),  # T002 model A good
            _make_pass3_result("FRAUD"),      # T001 model B bad
            _make_pass3_result("NOT_FRAUD"),  # T001 model B medium
            _make_pass3_result("FRAUD"),      # T001 model B good
            _make_pass3_result("NOT_FRAUD"),  # T002 model B bad
            _make_pass3_result("FRAUD"),      # T002 model B medium
            _make_pass3_result("NOT_FRAUD"),  # T002 model B good
        ]
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=two_models)
        model_a_rows = [r for r in rows if r["model"] == "open-mistral-nemo" and r["transcript_id"] == "T001"]
        model_b_rows = [r for r in rows if r["model"] == "mistral-small-2503" and r["transcript_id"] == "T001"]
        assert all(r["consistent"] is True for r in model_a_rows)
        assert all(r["consistent"] is False for r in model_b_rows)

    def test_per_context_cases_count(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        _, per_context, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert len(per_context) == len(SAMPLE_TRANSCRIPTS) * len(CONTEXT_TYPES)

    def test_per_transcript_cases_count(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        _, _, per_transcript = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        assert len(per_transcript) == len(SAMPLE_TRANSCRIPTS)

    def test_per_transcript_case_has_all_predictions(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        _, _, per_transcript = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        for tc in per_transcript:
            preds = json.loads(tc.actual_output)
            assert set(preds.keys()) == {"bad", "medium", "good"}

    def test_per_context_metadata_includes_model(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        _, per_context, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        for tc in per_context:
            assert tc.additional_metadata["model"] == "open-mistral-nemo"

    def test_notes_uses_fraud_type(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        t1_rows = [r for r in rows if r["transcript_id"] == "T001"]
        assert all(r["notes"] == "bank_impersonation" for r in t1_rows)

    def test_notes_empty_for_none_fraud_type(self, mock_classify):
        mock_classify.return_value = _make_pass3_result("NOT_FRAUD")
        rows, _, _ = run_pipeline(SAMPLE_TRANSCRIPTS, models=SINGLE_MODEL)
        t2_rows = [r for r in rows if r["transcript_id"] == "T002"]
        assert all(r["notes"] == "" for r in t2_rows)
