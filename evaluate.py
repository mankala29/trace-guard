"""
evaluate.py
Full evaluation pipeline: load → build contexts → classify → score → report.

Runs across multiple Mistral models so context quality × model size can be compared.

Evaluation layers:
  1. Custom CSV pipeline  — accuracy, consistency, pass3, reasoning quality per row
  2. DeepEval layer       — FraudAccuracyMetric, ReasoningFidelityMetric,
                            ContextSensitivityMetric via deepeval.evaluate()

Results saved to results/results.csv.
"""

import csv
import json
import sys
from pathlib import Path

from deepeval import evaluate as deepeval_evaluate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from classify import classify_pass3
from context_builder import build_all_contexts
from deepeval_metrics import (
    ContextSensitivityMetric,
    FraudAccuracyMetric,
    GroundedAccuracyMetric,
    ReasoningFidelityMetric,
)

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "results.csv"
TRANSCRIPTS_PATH = Path("data/transcripts.json")

MODELS = [
    "open-mistral-nemo",
    "mistral-small-2503",
    "mistral-large-latest",
]

CONTEXT_TYPES = ["bad_context", "medium_context", "good_context"]

REASONING_SIGNAL_KEYWORDS = [
    "threat", "urgency", "verify", "account", "pressure", "suspicious",
    "wire", "gift card", "credential", "impersonat", "isolation", "coer",
    "fee", "arrest", "password", "code", "intercept", "pretense", "fake",
]

CSV_COLUMNS = [
    "transcript_id",
    "label",
    "model",
    "context_type",
    "prediction",
    "correct",
    "pass3",
    "consistent",
    "reasoning_quality",
    "grounded_correct",
    "error_type",        # "tp" | "tn" | "fp" | "fn" | ""
    "notes",
    "raw_response",
]


def _reasoning_quality(reason: str) -> bool:
    lower = reason.lower()
    return any(kw in lower for kw in REASONING_SIGNAL_KEYWORDS)


def _error_type(prediction: str, label: str) -> str:
    if prediction == "FRAUD" and label == "FRAUD":
        return "tp"
    if prediction == "NOT_FRAUD" and label == "NOT_FRAUD":
        return "tn"
    if prediction == "FRAUD" and label == "NOT_FRAUD":
        return "fp"
    if prediction == "NOT_FRAUD" and label == "FRAUD":
        return "fn"
    return ""


def _normalise_label(label: str) -> str:
    return label.upper().replace("-", "_")


def _init_csv(path: Path) -> None:
    """Write CSV header. Called once before run_pipeline."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def _append_row(path: Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def run_pipeline(
    transcripts: list[dict],
    models: list[str] = MODELS,
    output_csv: Path | None = None,
) -> tuple[list[dict], list[LLMTestCase], list[LLMTestCase]]:
    """
    Returns:
        rows                 — CSV row dicts (one per transcript × model × context type)
        per_context_cases    — LLMTestCases for FraudAccuracy + ReasoningFidelity
        per_transcript_cases — LLMTestCases for ContextSensitivity (one per transcript × model)

    If output_csv is provided, each row is written immediately — crash-safe.
    """
    rows: list[dict] = []
    per_context_cases: list[LLMTestCase] = []
    per_transcript_cases: list[LLMTestCase] = []

    for model in models:
        print(f"\n── Model: {model} {'─' * (40 - len(model))}")

        for t in transcripts:
            tid = t["id"]
            true_label = _normalise_label(t["label"])
            contexts = build_all_contexts(t)
            predictions: dict[str, str] = {}
            raw_outputs: dict[str, str] = {}

            for ctx_type in CONTEXT_TYPES:
                ctx_text = contexts[ctx_type]
                print(f"  [{tid}] {ctx_type} ...", end=" ", flush=True)

                result = classify_pass3(ctx_text, model=model)
                pred = result["prediction"]
                reason = result["reason"]
                raw = result["raw_response"]
                p3 = result["pass3"]
                correct = pred == true_label
                rq = _reasoning_quality(reason)

                predictions[ctx_type] = pred
                raw_outputs[ctx_type] = raw

                trial_summary = "/".join(tr["prediction"][0] for tr in result["trials"])
                print(f"{pred} ({'✓' if correct else '✗'})  pass3={'✓' if p3 else '✗'}  [{trial_summary}]")

                row = {
                    "transcript_id": tid,
                    "label": true_label,
                    "model": model,
                    "context_type": ctx_type,
                    "prediction": pred,
                    "correct": correct,
                    "pass3": p3,
                    "consistent": None,  # filled below
                    "reasoning_quality": rq,
                    "grounded_correct": correct and rq,
                    "error_type": _error_type(pred, true_label),
                    "notes": t.get("fraud_type") or "",
                    "raw_response": raw,
                }
                rows.append(row)
                if output_csv:
                    _append_row(output_csv, row)

                per_context_cases.append(
                    LLMTestCase(
                        input=ctx_text,
                        actual_output=raw,
                        expected_output=true_label,
                        additional_metadata={
                            "transcript_id": tid,
                            "model": model,
                            "context_type": ctx_type,
                            "fraud_type": t.get("fraud_type"),
                        },
                    )
                )

            # Consistency: all 3 context types agree for this transcript × model
            is_consistent = len(set(predictions.values())) == 1
            for row in rows:
                if row["transcript_id"] == tid and row["model"] == model:
                    row["consistent"] = is_consistent

            ctx_key_map = {
                "bad_context": "bad",
                "medium_context": "medium",
                "good_context": "good",
            }
            per_transcript_cases.append(
                LLMTestCase(
                    input=f"Transcript {tid}",
                    actual_output=json.dumps({ctx_key_map[k]: v for k, v in predictions.items()}),
                    expected_output=true_label,
                    additional_metadata={"transcript_id": tid, "model": model},
                )
            )

    return rows, per_context_cases, per_transcript_cases


def run_deepeval(
    per_context_cases: list[LLMTestCase],
    per_transcript_cases: list[LLMTestCase],
) -> None:
    print("\nRunning DeepEval metrics...\n")

    deepeval_evaluate(
        test_cases=per_context_cases,
        metrics=[FraudAccuracyMetric(), ReasoningFidelityMetric(), GroundedAccuracyMetric()],
    )
    deepeval_evaluate(
        test_cases=per_transcript_cases,
        metrics=[ContextSensitivityMetric()],
    )

    print("\nDeepEval metric scores (mean across test cases):\n")
    for label, cases, metric_cls in [
        ("FraudAccuracy      ", per_context_cases, FraudAccuracyMetric),
        ("ReasoningFidelity  ", per_context_cases, ReasoningFidelityMetric),
        ("GroundedAccuracy   ", per_context_cases, GroundedAccuracyMetric),
        ("ContextSensitivity ", per_transcript_cases, ContextSensitivityMetric),
    ]:
        metric = metric_cls()
        scores = [metric.measure(tc) for tc in cases]
        mean = sum(scores) / len(scores) if scores else 0
        passed = sum(1 for s in scores if s >= metric.threshold)
        print(f"  {label}  mean={mean:.2f}  ({passed}/{len(scores)} passed)")


def save_csv(rows: list[dict]) -> None:
    """Final rewrite of CSV — resolves consistency values filled in after the fact."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {RESULTS_CSV}")


def print_summary(rows: list[dict], subtle_ids: set[str] | None = None) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    models = list(dict.fromkeys(r["model"] for r in rows))  # preserve order

    # Accuracy per model × context type
    print("\nAccuracy — model × context type:")
    header = f"  {'context':<20}" + "".join(f"  {m.split('-')[1]:<10}" for m in models)
    print(header)
    for ctx in CONTEXT_TYPES:
        line = f"  {ctx:<20}"
        for model in models:
            subset = [r for r in rows if r["context_type"] == ctx and r["model"] == model]
            if subset:
                correct = sum(1 for r in subset if r["correct"])
                line += f"  {correct}/{len(subset)} ({100*correct//len(subset):>2}%)"
        print(line)

    # Overall accuracy per model
    print("\nOverall accuracy per model:")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        correct = sum(1 for r in subset if r["correct"])
        pct = 100 * correct / len(subset)
        print(f"  {model:<30} {correct:>2}/{len(subset)}  ({pct:.0f}%)")

    # Pass³ per model
    print("\nPass³ rate per model:")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        p3 = sum(1 for r in subset if r["pass3"])
        pct = 100 * p3 / len(subset)
        print(f"  {model:<30} {p3:>2}/{len(subset)}  ({pct:.0f}%)")

    # Consistency per model
    print("\nConsistency (all 3 context types agree) per model:")
    for model in models:
        # one value per transcript per model — use bad_context rows as index
        subset = [r for r in rows if r["model"] == model and r["context_type"] == "bad_context"]
        consistent = sum(1 for r in subset if r["consistent"])
        pct = 100 * consistent / len(subset) if subset else 0
        print(f"  {model:<30} {consistent:>2}/{len(subset)}  ({pct:.0f}%)")

    # Reasoning quality per model
    print("\nReasoning quality per model:")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        rq = sum(1 for r in subset if r["reasoning_quality"])
        pct = 100 * rq / len(subset)
        print(f"  {model:<30} {rq:>2}/{len(subset)}  ({pct:.0f}%)")

    # Subtle vs obvious fraud detection
    if subtle_ids is not None:
        print("\nSubtle vs obvious fraud detection:")
        for model in models:
            fraud_rows = [r for r in rows if r["model"] == model and r["label"] == "FRAUD"]
            subtle_rows = [r for r in fraud_rows if r["transcript_id"] in subtle_ids]
            obvious_rows = [r for r in fraud_rows if r["transcript_id"] not in subtle_ids]
            s_correct = sum(1 for r in subtle_rows if r["correct"])
            s_total = len(subtle_rows)
            o_correct = sum(1 for r in obvious_rows if r["correct"])
            o_total = len(obvious_rows)
            s_pct = 100 * s_correct // s_total if s_total else 0
            o_pct = 100 * o_correct // o_total if o_total else 0
            print(f"  {model:<30} subtle={s_correct}/{s_total} ({s_pct}%)  obvious={o_correct}/{o_total} ({o_pct}%)")

    # Grounded accuracy per model (correct AND reasoning quality)
    print("\nGrounded accuracy (correct + cites signal) per model:")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        gc = sum(1 for r in subset if r["grounded_correct"])
        acc = sum(1 for r in subset if r["correct"])
        pct = 100 * gc / len(subset)
        print(f"  {model:<30} {gc:>2}/{len(subset)}  ({pct:.0f}%)  vs accuracy {100*acc//len(subset)}%")

    # Error direction per model
    print("\nError direction per model (fn=missed fraud, fp=false alarm):")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        fn = sum(1 for r in subset if r["error_type"] == "fn")
        fp = sum(1 for r in subset if r["error_type"] == "fp")
        print(f"  {model:<30} fn={fn}  fp={fp}")

    # Context sensitivity direction per model
    print("\nContext sensitivity direction (bad → good context):")
    for model in models:
        rescued = degraded = robust = failed = 0
        tids = list(dict.fromkeys(r["transcript_id"] for r in rows))
        for tid in tids:
            bad = next((r for r in rows if r["transcript_id"] == tid and r["model"] == model and r["context_type"] == "bad_context"), None)
            good = next((r for r in rows if r["transcript_id"] == tid and r["model"] == model and r["context_type"] == "good_context"), None)
            if not bad or not good:
                continue
            b, g = bad["correct"], good["correct"]
            if not b and g:
                rescued += 1
            elif b and not g:
                degraded += 1
            elif b and g:
                robust += 1
            else:
                failed += 1
        print(f"  {model:<30} robust={robust}  rescued={rescued}  degraded={degraded}  failed={failed}")

    print("=" * 70)


def main() -> None:
    if not TRANSCRIPTS_PATH.exists():
        sys.exit(f"Transcripts file not found: {TRANSCRIPTS_PATH}")

    with open(TRANSCRIPTS_PATH, encoding="utf-8") as f:
        transcripts = json.load(f)

    total_calls = len(transcripts) * len(CONTEXT_TYPES) * len(MODELS)
    print(f"Loaded {len(transcripts)} transcripts.")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Total classification calls: {total_calls} ({total_calls * 3} with Pass³ trials)\n")

    subtle_ids = {t["id"] for t in transcripts if t.get("subtle")}

    _init_csv(RESULTS_CSV)
    rows, per_context_cases, per_transcript_cases = run_pipeline(
        transcripts, output_csv=RESULTS_CSV
    )
    save_csv(rows)  # final rewrite — resolves consistency column
    print_summary(rows, subtle_ids=subtle_ids)
    run_deepeval(per_context_cases, per_transcript_cases)


if __name__ == "__main__":
    main()
