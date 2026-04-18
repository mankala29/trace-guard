"""
temperature_pass3.py
Focused Pass³ analysis at temperature=0.7 to test semantic stability under sampling.

Compares pass3 rate at temp=0 (from main run) vs temp=0.7 (this script) across
all 3 models on a representative subset of 10 transcripts.

Subset: 5 subtle fraud + 5 non-fraud (the harder cases).
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from classify import classify_pass3
from context_builder import build_all_contexts
from evaluate import MODELS, CONTEXT_TYPES, _normalise_label

load_dotenv()

TEMPERATURE = 0.7
N_TRIALS = 3

# Subtle fraud: T007-T015 — non-fraud: T016, T018, T019, T020, T025
SUBSET_IDS = {"T007", "T008", "T009", "T010", "T011", "T016", "T018", "T019", "T020", "T025"}


def run(transcripts: list[dict]) -> None:
    subset = [t for t in transcripts if t["id"] in SUBSET_IDS]

    print(f"Pass³ at temperature={TEMPERATURE}  ({N_TRIALS} trials per call)")
    print(f"Subset: {len(subset)} transcripts  |  {len(MODELS)} models  |  {len(CONTEXT_TYPES)} context types")
    print(f"Total calls: {len(subset) * len(MODELS) * len(CONTEXT_TYPES) * N_TRIALS}\n")

    # results[model][context_type] = {"pass3": int, "total": int, "flips": list}
    results: dict[str, dict[str, dict]] = {
        m: {c: {"pass3": 0, "total": 0, "flips": []} for c in CONTEXT_TYPES}
        for m in MODELS
    }

    for model in MODELS:
        print(f"── {model}")
        for t in subset:
            tid = t["id"]
            true_label = _normalise_label(t["label"])
            contexts = build_all_contexts(t)

            for ctx_type in CONTEXT_TYPES:
                print(f"  [{tid}] {ctx_type} ...", end=" ", flush=True)

                result = classify_pass3(
                    contexts[ctx_type],
                    model=model,
                    n_trials=N_TRIALS,
                    temperature=TEMPERATURE,
                )
                p3 = result["pass3"]
                trial_preds = [tr["prediction"][0] for tr in result["trials"]]
                summary = "/".join(trial_preds)

                results[model][ctx_type]["total"] += 1
                if p3:
                    results[model][ctx_type]["pass3"] += 1
                else:
                    results[model][ctx_type]["flips"].append(
                        f"{tid}({true_label[0]}) [{summary}]"
                    )

                print(f"{'✓' if p3 else '✗'}  [{summary}]")

    # Summary
    print("\n" + "=" * 65)
    print(f"PASS³ RATE AT TEMPERATURE={TEMPERATURE}")
    print("=" * 65)

    header = f"  {'context':<20}" + "".join(f"  {m.split('-')[1]:<12}" for m in MODELS)
    print("\n" + header)
    for ctx in CONTEXT_TYPES:
        line = f"  {ctx:<20}"
        for model in MODELS:
            r = results[model][ctx]
            pct = 100 * r["pass3"] // r["total"] if r["total"] else 0
            line += f"  {r['pass3']}/{r['total']} ({pct:>2}%)"
        print(line)

    print("\nFlips (transcripts where trials disagreed):")
    for model in MODELS:
        all_flips = []
        for ctx in CONTEXT_TYPES:
            for flip in results[model][ctx]["flips"]:
                all_flips.append(f"{ctx.split('_')[0]}:{flip}")
        if all_flips:
            print(f"  {model}: {', '.join(all_flips)}")
        else:
            print(f"  {model}: none")

    print("=" * 65)


if __name__ == "__main__":
    transcripts = json.loads(Path("data/transcripts.json").read_text())
    run(transcripts)
