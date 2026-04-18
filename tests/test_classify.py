"""
tests/test_classify.py
Unit + mocked API tests for classify.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.classify import _parse_response, classify, classify_pass3


# ── _parse_response ────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_fraud(self):
        raw = "PREDICTION: FRAUD\nREASON: Caller demanded gift cards under threat of arrest."
        result = _parse_response(raw)
        assert result["prediction"] == "FRAUD"
        assert "gift cards" in result["reason"]

    def test_not_fraud(self):
        raw = "PREDICTION: NOT_FRAUD\nREASON: Routine appointment reminder, no financial ask."
        result = _parse_response(raw)
        assert result["prediction"] == "NOT_FRAUD"

    def test_case_insensitive(self):
        raw = "prediction: fraud\nreason: Suspicious."
        result = _parse_response(raw)
        assert result["prediction"] == "FRAUD"

    def test_missing_prediction(self):
        raw = "I think this might be suspicious."
        result = _parse_response(raw)
        assert result["prediction"] == "UNKNOWN"

    def test_missing_reason(self):
        raw = "PREDICTION: FRAUD"
        result = _parse_response(raw)
        assert result["prediction"] == "FRAUD"
        assert result["reason"] == ""

    def test_multiline_reason(self):
        raw = "PREDICTION: NOT_FRAUD\nREASON: This is a legitimate call.\nNo red flags detected."
        result = _parse_response(raw)
        assert "legitimate" in result["reason"]

    def test_raw_response_preserved(self):
        raw = "PREDICTION: FRAUD\nREASON: Urgency and wire transfer request."
        result = _parse_response(raw)
        assert result["raw_response"] == raw


# ── classify (mocked Mistral) ──────────────────────────────────────────────────

def _make_mock_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("core.classify.time.sleep")
@patch("core.classify._get_client")
class TestClassifyRetry:
    def _make_429(self):
        from mistralai.client.errors.sdkerror import SDKError
        err = SDKError("API error occurred", MagicMock(status_code=429), "Rate limit exceeded: Status 429")
        return err

    def _make_503(self):
        from mistralai.client.errors.sdkerror import SDKError
        err = SDKError("API error occurred", MagicMock(status_code=503), "Server error: Status 503")
        return err

    def _ok_response(self):
        return _make_mock_response("PREDICTION: FRAUD\nREASON: Suspicious urgency.")

    def test_retries_on_429(self, mock_get_client, mock_sleep):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.side_effect = [
            self._make_429(),
            self._ok_response(),
        ]
        result = classify("ctx")
        assert result["prediction"] == "FRAUD"
        assert mock_client.chat.complete.call_count == 2

    def test_retries_on_503(self, mock_get_client, mock_sleep):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.side_effect = [
            self._make_503(),
            self._ok_response(),
        ]
        result = classify("ctx")
        assert result["prediction"] == "FRAUD"
        assert mock_client.chat.complete.call_count == 2

    def test_sleeps_on_retry(self, mock_get_client, mock_sleep):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.side_effect = [
            self._make_429(),
            self._ok_response(),
        ]
        classify("ctx")
        # First sleep is CALL_DELAY, second is backoff wait
        assert mock_sleep.call_count >= 2

    def test_raises_after_max_retries(self, mock_get_client, mock_sleep):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.side_effect = self._make_429()
        with pytest.raises(Exception, match="429"):
            classify("ctx", _retries=2)

    def test_non_retryable_error_raises_immediately(self, mock_get_client, mock_sleep):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.side_effect = ValueError("Bad input")
        with pytest.raises(ValueError):
            classify("ctx")
        assert mock_client.chat.complete.call_count == 1


@patch("core.classify._get_client")
class TestClassify:
    def test_returns_fraud(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.return_value = _make_mock_response(
            "PREDICTION: FRAUD\nREASON: Caller requested account password."
        )

        result = classify("some context")
        assert result["prediction"] == "FRAUD"
        assert result["reason"] != ""

    def test_returns_not_fraud(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.return_value = _make_mock_response(
            "PREDICTION: NOT_FRAUD\nREASON: Appointment reminder, nothing suspicious."
        )

        result = classify("some context")
        assert result["prediction"] == "NOT_FRAUD"

    def test_uses_temperature_zero(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.complete.return_value = _make_mock_response(
            "PREDICTION: FRAUD\nREASON: Threat."
        )

        classify("ctx")
        call_kwargs = mock_client.chat.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    def test_missing_api_key_raises(self, mock_get_client):
        mock_get_client.side_effect = EnvironmentError("MISTRAL_API_KEY not set")
        with pytest.raises(EnvironmentError):
            classify("ctx")


# ── classify_pass3 (mocked) ────────────────────────────────────────────────────

@patch("core.classify.classify")
class TestClassifyPass3:
    def _fraud_result(self, reason="Caller requested credentials."):
        return {"prediction": "FRAUD", "reason": reason, "raw_response": f"PREDICTION: FRAUD\nREASON: {reason}"}

    def _not_fraud_result(self, reason="Routine call."):
        return {"prediction": "NOT_FRAUD", "reason": reason, "raw_response": f"PREDICTION: NOT_FRAUD\nREASON: {reason}"}

    def test_all_agree_pass3_true(self, mock_classify):
        mock_classify.return_value = self._fraud_result()
        result = classify_pass3("ctx", n_trials=3)
        assert result["pass3"] is True
        assert result["prediction"] == "FRAUD"
        assert len(result["trials"]) == 3

    def test_disagreement_pass3_false(self, mock_classify):
        mock_classify.side_effect = [
            self._fraud_result(),
            self._not_fraud_result(),
            self._fraud_result(),
        ]
        result = classify_pass3("ctx", n_trials=3)
        assert result["pass3"] is False
        assert result["prediction"] == "FRAUD"  # majority

    def test_majority_vote(self, mock_classify):
        mock_classify.side_effect = [
            self._not_fraud_result(),
            self._not_fraud_result(),
            self._fraud_result(),
        ]
        result = classify_pass3("ctx", n_trials=3)
        assert result["prediction"] == "NOT_FRAUD"
        assert result["pass3"] is False

    def test_n_trials_respected(self, mock_classify):
        mock_classify.return_value = self._fraud_result()
        classify_pass3("ctx", n_trials=5)
        assert mock_classify.call_count == 5

    def test_first_trial_reason_used(self, mock_classify):
        mock_classify.side_effect = [
            self._fraud_result(reason="First reason."),
            self._fraud_result(reason="Second reason."),
            self._fraud_result(reason="Third reason."),
        ]
        result = classify_pass3("ctx", n_trials=3)
        assert result["reason"] == "First reason."
