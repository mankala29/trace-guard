"""
llm_judge.py
LLM-as-judge eval layer: use mistral-large to assess the quality of
smaller models' reasoning.

For each (transcript, context_type, model) result, the judge reads the
full model response and scores it on two dimensions:

  reasoning_score (0–3)
    0 — no reasoning, just a label
    1 — generic reason unrelated to transcript content
    2 — mentions a signal but is vague or partially correct
    3 — cites a specific, transcript-grounded fraud signal

  verdict  — "strong" | "weak" | "hallucinated"
    strong       — reasoning is grounded and accurate
    weak         — reasoning is present but unspecific / generic
    hallucinated — reasoning cites something not in the transcript

The judge always explains its verdict.

Usage:
    python llm_judge.py               # reads results/results.csv + data/transcripts.json
    python llm_judge.py --model nemo  # filter to one model ("nemo"|"small"|"large")
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client.sdk import Mistral

load_dotenv()

JUDGE_MODEL = "mistral-large-latest"
RESULTS_CSV = Path("results/results.csv")
TRANSCRIPTS_PATH = Path("data/transcripts.json")
JUDGE_CSV = Path("results/judge_results.csv")
CALL_DELAY = 1.5

JUDGE_PROMPT = """\
You are an expert fraud detection evaluator. Your job is to assess whether \
an AI model's reasoning about a phone call transcript is grounded and specific.

## Transcript
{transcript}

## Model's Response
{response}

## Evaluation Task
Score the model's REASON field on this rubric:

  3 — Cites a specific fraud signal actually present in the transcript \
(e.g. names the exact tactic, quotes or paraphrases a telling phrase)
  2 — Mentions a relevant fraud category but stays vague or generic
  1 — Reason exists but does not connect to this transcript's content
  0 — No reasoning provided, or only repeats the prediction label

Also assign one verdict:
  strong       — reasoning is grounded and accurate
  weak         — reasoning present but unspecific / generic
  hallucinated — reasoning cites a detail NOT present in the transcript

Respond exactly:
REASONING_SCORE: <0|1|2|3>
VERDICT: <strong|weak|hallucinated>
EXPLANATION: <one sentence>
"""

_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY not set")
        _client = Mistral(api_key=api_key)
    return _client


def _dialogue_to_text(dialogue: list[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in dialogue)


def judge_response(transcript_text: str, model_response: str, _retries: int = 5) -> dict:
    """Ask mistral-large to evaluate a model's reasoning. Returns parsed judge output."""
    prompt = JUDGE_PROMPT.format(transcript=transcript_text, response=model_response)
    client = _get_client()

    for attempt in range(_retries):
        try:
            time.sleep(CALL_DELAY)
            resp = client.chat.complete(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_judge(raw)
        except Exception as e:
            err = str(e)
            retryable = ("429" in err or "503" in err) and attempt < _retries - 1
            if retryable:
                wait = 2 ** (attempt + 2)
                code = "rate limit" if "429" in err else "server error"
                print(f" [{code}, retrying in {wait}s]", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise


def _parse_judge(raw: str) -> dict:
    score_match = re.search(r"REASONING_SCORE:\s*([0-3])", raw, re.IGNORECASE)
    verdict_match = re.search(r"VERDICT:\s*(strong|weak|hallucinated)", raw, re.IGNORECASE)
    expl_match = re.search(r"EXPLANATION:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)

    return {
        "reasoning_score": int(score_match.group(1)) if score_match else -1,
        "verdict": verdict_match.group(1).lower() if verdict_match else "unknown",
        "explanation": expl_match.group(1).strip() if expl_match else "",
        "raw": raw,
    }


JUDGE_COLUMNS = [
    "transcript_id", "model", "context_type",
    "prediction", "correct",
    "reasoning_score", "verdict", "explanation",
]


def _init_judge_csv(path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=JUDGE_COLUMNS).writeheader()


def _append_judge_row(path: Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=JUDGE_COLUMNS).writerow(row)


def run(model_filter: str | None = None) -> list[dict]:
    """
    Read results.csv + transcripts.json, run judge on each row, save judge_results.csv.
    model_filter: short name fragment e.g. "nemo", "small", "large" — filters rows.
    """
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} not found — run evaluate.py first")
    if not TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(f"{TRANSCRIPTS_PATH} not found")

    with open(TRANSCRIPTS_PATH, encoding="utf-8") as f:
        transcripts = {t["id"]: t for t in json.load(f)}

    with open(RESULTS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if model_filter:
        rows = [r for r in rows if model_filter in r["model"]]

    print(f"Judging {len(rows)} result rows with {JUDGE_MODEL}...\n")
    _init_judge_csv(JUDGE_CSV)

    judge_rows: list[dict] = []

    for i, row in enumerate(rows, 1):
        tid = row["transcript_id"]
        model = row["model"]
        ctx_type = row["context_type"]
        pred = row["prediction"]
        correct = row["correct"]
        raw_response = row.get("raw_response") or row.get("prediction", "")

        transcript = transcripts.get(tid)
        if not transcript:
            print(f"  [{tid}] transcript not found, skipping")
            continue

        transcript_text = _dialogue_to_text(transcript["dialogue"])

        # The raw_response column may not exist in older CSVs — fall back to prediction
        model_output = raw_response if raw_response else f"PREDICTION: {pred}\nREASON: (not recorded)"

        print(f"  [{i:>3}/{len(rows)}] {tid} | {model.split('-')[1]:<6} | {ctx_type:<15} ...", end=" ", flush=True)
        result = judge_response(transcript_text, model_output)
        print(f"score={result['reasoning_score']}  verdict={result['verdict']}")

        jrow = {
            "transcript_id": tid,
            "model": model,
            "context_type": ctx_type,
            "prediction": pred,
            "correct": correct,
            "reasoning_score": result["reasoning_score"],
            "verdict": result["verdict"],
            "explanation": result["explanation"],
        }
        judge_rows.append(jrow)
        _append_judge_row(JUDGE_CSV, jrow)

    _print_judge_summary(judge_rows)
    print(f"\nJudge results saved to {JUDGE_CSV}")
    return judge_rows


def _print_judge_summary(rows: list[dict]) -> None:
    if not rows:
        return

    print("\n" + "=" * 65)
    print("LLM-AS-JUDGE SUMMARY")
    print("=" * 65)

    models = list(dict.fromkeys(r["model"] for r in rows))

    print("\nMean reasoning score per model (0–3 scale):")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        valid = [r for r in subset if r["reasoning_score"] >= 0]
        if not valid:
            continue
        mean = sum(r["reasoning_score"] for r in valid) / len(valid)
        strong = sum(1 for r in valid if r["verdict"] == "strong")
        weak = sum(1 for r in valid if r["verdict"] == "weak")
        halluc = sum(1 for r in valid if r["verdict"] == "hallucinated")
        print(f"  {model:<30} mean={mean:.2f}  strong={strong}  weak={weak}  hallucinated={halluc}")

    print("\nMean reasoning score per model × context type:")
    ctx_types = ["bad_context", "medium_context", "good_context"]
    header = f"  {'context':<20}" + "".join(f"  {m.split('-')[1]:<10}" for m in models)
    print(header)
    for ctx in ctx_types:
        line = f"  {ctx:<20}"
        for model in models:
            subset = [r for r in rows if r["model"] == model and r["context_type"] == ctx and r["reasoning_score"] >= 0]
            if subset:
                mean = sum(r["reasoning_score"] for r in subset) / len(subset)
                line += f"  {mean:.1f}/{len(subset)}"
            else:
                line += "  —"
        print(line)

    print("\nHallucination cases:")
    halluc_rows = [r for r in rows if r["verdict"] == "hallucinated"]
    if halluc_rows:
        for r in halluc_rows:
            print(f"  {r['transcript_id']} | {r['model'].split('-')[1]:<6} | {r['context_type']:<15} | {r['explanation'][:80]}")
    else:
        print("  None detected")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge eval for TraceGuard")
    parser.add_argument("--model", default=None, help="Filter by model fragment: nemo|small|large")
    args = parser.parse_args()
    run(model_filter=args.model)
