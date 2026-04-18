"""
counterfactual_eval.py
Counterfactual sensitivity eval: strip the key fraud signals from a transcript
and re-classify. If the model was truly reading the signals, the prediction
should flip from FRAUD to NOT_FRAUD.

Method
------
For each fraud transcript in the subset:
  1. Generate a neutralised variant by replacing / removing explicit fraud
     signals (urgency language, gift-card demands, credential requests,
     arrest threats, isolation instructions).
  2. Classify both the original and the neutralised transcript at
     bad/medium/good context quality.
  3. Record whether the prediction flipped.

Neutralisation is rule-based (no extra LLM call) so it is deterministic
and cheap. The rules target the most common surface-level fraud markers.
Subtle transcripts are included to show that the model is less sensitive
there — which is the interesting finding.

Subset: T001–T006 (obvious), T007–T011 (subtle fraud) — 11 transcripts.
Models: all three (nemo, small, large).
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from core.classify import classify_pass3
from core.context_builder import build_all_contexts
from evaluate import MODELS, CONTEXT_TYPES, _normalise_label

load_dotenv()

TRANSCRIPTS_PATH = Path("data/transcripts.json")
RESULTS_PATH = Path("results/counterfactual_results.json")

# Subset: obvious fraud + subtle fraud (exclude non-fraud — nothing to neutralise)
SUBSET_IDS = {
    "T001", "T002", "T003", "T004", "T005", "T006",  # obvious
    "T007", "T008", "T009", "T010", "T011",           # subtle
}

# ── Neutralisation rules ────────────────────────────────────────────────────────

# Each rule: (regex_pattern, replacement)
# Applied to every dialogue turn's text.
NEUTRALISATION_RULES: list[tuple[str, str]] = [
    # Urgency + time pressure (more specific patterns first)
    (r"\bin the next \d+ minutes?\b", "when you get a chance"),
    (r"\b(immediately|right now|right away|at once|this instant)\b", "soon"),
    (r"\byou (must|have to|need to) act now\b", "you can take action"),
    (r"\bif you (don't|do not) act\b", "if you'd like to proceed"),
    (r"\byour account will be (frozen|suspended|closed)\b", "your account is being reviewed"),
    (r"\b(or )?you will be (arrested|charged|detained)\b", ""),
    # Arrest / legal threats
    (r"\b(arrest warrant|federal warrant|warrant (has been|was) issued)\b", "notice was sent"),
    (r"\b(arrest|arrested|jail|detention)\b", "inconvenience"),
    (r"\b(IRS|Internal Revenue Service)\b", "billing department"),
    (r"\bfederal (officer|matter|agent|agents)\b", "customer service representative"),
    (r"\bdo not contact (a )?(lawyer|attorney)\b", ""),
    # Gift card / wire transfer demands (specific phrases before generic)
    (r"\b(Google Play|iTunes|Amazon|Steam) gift cards?\b", "standard payment"),
    (r"\bgift[- ]cards?\b", "payment method"),
    (r"\bredemption codes?\b", "payment reference"),
    (r"\b(wire transfer|wire the funds?|Western Union|MoneyGram)\b", "bank transfer"),
    # Credential requests (compound form first to avoid double-replacement)
    (r"\bonline banking (password|credentials)\b", "account details"),
    (r"\b(password|PIN|passcode)\b", "verification code"),
    (r"\bone-time (code|password|pin|OTP)\b", "confirmation code"),
    (r"\b(online banking|banking portal)\b", "account portal"),
    # Isolation / secrecy instructions
    (r"\b(don't tell|do not tell)\b", ""),
    (r"\bkeep this (between us|confidential|secret)\b", ""),
    (r"\b(hang up|hang-up) (and )?call (back )?(us|me|this number)\b", "contact us"),
    # Remote access
    (r"\b(AnyDesk|TeamViewer)\b", "our website"),
    (r"\bremote (access|desktop|session)\b", "online support"),
    (r"\bdownload .{0,30}(tool|software|app)\b", "visit our website"),
    # Surveillance / monitoring language (subtle fraud)
    (r"\b(surveillance|being watched|being monitored)\b", "being followed up with"),
    (r"\b(isolated|isolation|cut off from)\b", "separated from"),
]


def neutralise_dialogue(dialogue: list[dict]) -> list[dict]:
    """
    Return a new dialogue list with fraud signals stripped from each turn.
    """
    neutralised = []
    for turn in dialogue:
        text = turn["text"]
        for pattern, replacement in NEUTRALISATION_RULES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Collapse multiple spaces / trailing punctuation artifacts
        text = re.sub(r" {2,}", " ", text).strip()
        neutralised.append({"speaker": turn["speaker"], "text": text})
    return neutralised


def build_neutralised_transcript(t: dict) -> dict:
    """Return a copy of transcript t with its dialogue neutralised."""
    return {**t, "dialogue": neutralise_dialogue(t["dialogue"])}


# ── Run ─────────────────────────────────────────────────────────────────────────

def run(transcripts: list[dict]) -> list[dict]:
    subset = [t for t in transcripts if t["id"] in SUBSET_IDS]
    total = len(subset) * len(MODELS) * len(CONTEXT_TYPES) * 2  # ×2 for orig + neutral

    print(f"Counterfactual sensitivity eval")
    print(f"Subset: {len(subset)} fraud transcripts  |  {len(MODELS)} models  |  {len(CONTEXT_TYPES)} context types")
    print(f"Total classification calls: {total} ({total * 3} with Pass³ trials)\n")

    records: list[dict] = []

    for model in MODELS:
        print(f"── {model}")
        for t in subset:
            tid = t["id"]
            true_label = _normalise_label(t["label"])
            subtle = t.get("subtle", False)

            orig_contexts = build_all_contexts(t)
            neutral_t = build_neutralised_transcript(t)
            neutral_contexts = build_all_contexts(neutral_t)

            for ctx_type in CONTEXT_TYPES:
                print(f"  [{tid}{'*' if subtle else ' '}] {ctx_type}", end=" ", flush=True)

                orig_result = classify_pass3(orig_contexts[ctx_type], model=model)
                orig_pred = orig_result["prediction"]

                neutral_result = classify_pass3(neutral_contexts[ctx_type], model=model)
                neutral_pred = neutral_result["prediction"]

                flipped = orig_pred == "FRAUD" and neutral_pred == "NOT_FRAUD"
                held = orig_pred == "FRAUD" and neutral_pred == "FRAUD"
                status = "FLIPPED" if flipped else ("HELD" if held else f"{orig_pred}→{neutral_pred}")

                print(f"  orig={orig_pred[0]}  neutral={neutral_pred[0]}  [{status}]")

                records.append({
                    "transcript_id": tid,
                    "subtle": subtle,
                    "fraud_type": t.get("fraud_type", ""),
                    "model": model,
                    "context_type": ctx_type,
                    "original_prediction": orig_pred,
                    "neutral_prediction": neutral_pred,
                    "flipped": flipped,  # fraud → not_fraud after neutralisation
                    "held": held,        # still fraud after neutralisation
                })

    _print_summary(records)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nCounterfactual results saved to {RESULTS_PATH}")
    return records


def _print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("COUNTERFACTUAL SENSITIVITY SUMMARY")
    print("=" * 65)

    models = list(dict.fromkeys(r["model"] for r in records))

    # Overall flip rate per model
    print("\nFlip rate (FRAUD→NOT_FRAUD after neutralisation) per model:")
    for model in models:
        subset = [r for r in records if r["model"] == model]
        flipped = sum(1 for r in subset if r["flipped"])
        total = len(subset)
        pct = 100 * flipped // total if total else 0
        held = sum(1 for r in subset if r["held"])
        print(f"  {model:<30} flipped={flipped}/{total} ({pct}%)  held={held}")

    # Obvious vs subtle breakdown
    print("\nFlip rate: obvious vs subtle fraud per model:")
    for model in models:
        obvious = [r for r in records if r["model"] == model and not r["subtle"]]
        subtle  = [r for r in records if r["model"] == model and r["subtle"]]
        o_flip  = sum(1 for r in obvious if r["flipped"])
        s_flip  = sum(1 for r in subtle  if r["flipped"])
        o_pct   = 100 * o_flip // len(obvious) if obvious else 0
        s_pct   = 100 * s_flip // len(subtle)  if subtle  else 0
        print(f"  {model:<30} obvious={o_flip}/{len(obvious)} ({o_pct}%)  subtle={s_flip}/{len(subtle)} ({s_pct}%)")

    # Per context type
    print("\nFlip rate per context type:")
    ctx_types = ["bad_context", "medium_context", "good_context"]
    header = f"  {'context':<20}" + "".join(f"  {m.split('-')[1]:<10}" for m in models)
    print(header)
    for ctx in ctx_types:
        line = f"  {ctx:<20}"
        for model in models:
            subset = [r for r in records if r["model"] == model and r["context_type"] == ctx]
            if subset:
                flipped = sum(1 for r in subset if r["flipped"])
                pct = 100 * flipped // len(subset)
                line += f"  {flipped}/{len(subset)} ({pct:>2}%)"
        print(line)

    # Transcripts that never flipped (model insensitive to signal removal)
    print("\nTranscripts that NEVER flipped across any model/context:")
    tids = list(dict.fromkeys(r["transcript_id"] for r in records))
    never_flipped = []
    for tid in tids:
        tid_records = [r for r in records if r["transcript_id"] == tid]
        if not any(r["flipped"] for r in tid_records):
            ftype = tid_records[0]["fraud_type"] if tid_records else ""
            subtle = tid_records[0]["subtle"] if tid_records else False
            never_flipped.append(f"{tid} ({ftype}{'*subtle' if subtle else ''})")
    if never_flipped:
        for nf in never_flipped:
            print(f"  {nf}")
    else:
        print("  All transcripts flipped in at least one condition")

    print("=" * 65)


if __name__ == "__main__":
    transcripts = json.loads(TRANSCRIPTS_PATH.read_text())
    run(transcripts)
