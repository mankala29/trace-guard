"""
classify.py
Classification calls to open-mistral-nemo via Mistral API.
Supports Pass³: run n_trials times and require unanimous agreement.
"""

import os
import re
import time

from mistralai.client.sdk import Mistral

PROMPT_TEMPLATE = """You are a fraud detection assistant analyzing phone call transcripts.

Classify the call as FRAUD or NOT_FRAUD. Provide a brief reason (1-2 sentences).

{context}

Respond exactly:
PREDICTION: FRAUD or NOT_FRAUD
REASON: <your reasoning>"""

_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY environment variable is not set.")
        _client = Mistral(api_key=api_key)
    return _client


CALL_DELAY = 1.5  # seconds between calls — respects free-tier rate limits


def classify(
    context: str,
    model: str = "open-mistral-nemo",
    temperature: float = 0,
    _retries: int = 5,
) -> dict:
    """
    Single classification call with a proactive delay + exponential backoff on 429s.
    Returns:
        {"prediction": "FRAUD"|"NOT_FRAUD"|"UNKNOWN", "reason": str, "raw_response": str}
    """
    prompt = PROMPT_TEMPLATE.format(context=context)
    client = _get_client()

    for attempt in range(_retries):
        try:
            time.sleep(CALL_DELAY)
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_response(raw)
        except Exception as e:
            err = str(e)
            retryable = ("429" in err or "503" in err) and attempt < _retries - 1
            if retryable:
                wait = 2 ** (attempt + 2)  # 4s, 8s, 16s, 32s
                code = "rate limit" if "429" in err else "server error"
                print(f" [{code}, retrying in {wait}s]", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise


def classify_pass3(
    context: str,
    model: str = "open-mistral-nemo",
    n_trials: int = 3,
    temperature: float = 0,
) -> dict:
    """
    Run n_trials independent classifications (Pass³ methodology from Claw-Eval).
    A result is considered stable only if all trials agree.

    Use temperature > 0 to test semantic stability under sampling.

    Returns:
        {
            "prediction": majority-vote prediction,
            "reason": reason from the first trial,
            "raw_response": raw_response from the first trial,
            "trials": [{"prediction": ..., "reason": ...}, ...],
            "pass3": bool  — True if all trials returned the same prediction,
        }
    """
    trials = [classify(context, model=model, temperature=temperature) for _ in range(n_trials)]
    predictions = [t["prediction"] for t in trials]

    # Majority vote (handles UNKNOWN gracefully)
    majority = max(set(predictions), key=predictions.count)
    all_agree = len(set(predictions)) == 1

    return {
        "prediction": majority,
        "reason": trials[0]["reason"],
        "raw_response": trials[0]["raw_response"],
        "trials": [{"prediction": t["prediction"], "reason": t["reason"]} for t in trials],
        "pass3": all_agree,
    }


def _parse_response(raw: str) -> dict:
    prediction = "UNKNOWN"
    reason = ""

    pred_match = re.search(r"PREDICTION:\s*(FRAUD|NOT_FRAUD)", raw, re.IGNORECASE)
    if pred_match:
        prediction = pred_match.group(1).upper()

    reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()

    return {"prediction": prediction, "reason": reason, "raw_response": raw}
