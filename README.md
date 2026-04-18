# TraceGuard

A personal experiment measuring how **context quality** affects LLM fraud detection — and how well different model sizes actually read the signals they claim to read.

Built around 25 synthetic phone call transcripts (15 fraud, 10 non-fraud) evaluated across three Mistral AI models with four layers of eval.

---

## About This Project

TraceGuard is an independent educational research project conceived, designed, and authored solely by Sweta Mankala. It was created to explore publicly available AI evaluation methodologies and is intended for publication as a research article.

All intellectual property in this project — including its experimental design, evaluation framework, methodology, and implementation — is the original work of and exclusively owned by Sweta Mankala.

This project was developed independently:
- Conceived and built on personal time using personal equipment and resources only
- Based entirely on publicly available tools, libraries, and research literature
- No proprietary information, confidential data, or trade secrets of any employer or third party were used or incorporated at any stage
- All transcripts are fully synthetic, created solely for this research, and do not represent any real individuals, organisations, or events

This project is not affiliated with, sponsored by, or connected to any employer, past or present. Any future use, adaptation, or commercialisation of the ideas, methods, or code in this repository by any third party requires the explicit written consent of Sweta Mankala.

© 2026 Sweta Mankala. Released under the [MIT License](LICENSE).

---

## Motivation

Fraud calls are won or lost on context. A raw transcript stripped of framing looks different from one annotated with intent and key signals. This experiment asks: does giving a model better context actually help, or does model size matter more? And when a model says "this is fraud because of X" — is it actually reading X?

---

## Eval Layers

| Layer | What it measures | Where |
|---|---|---|
| **Accuracy + Pass³** | Prediction correctness; stability across 3 independent trials | `evaluate.py` |
| **Grounded accuracy** | Correct prediction *and* reason cites a fraud signal | `deepeval_metrics.py` |
| **LLM-as-judge** | mistral-large scores smaller models' reasoning on 0–3 scale | `llm_judge.py` |
| **Counterfactual sensitivity** | Does stripping fraud signals flip the prediction? | `counterfactual_eval.py` |
| **Temperature Pass³** | Semantic stability at temp=0.7 on a harder subset | `temperature_pass3.py` |

Custom [DeepEval](https://github.com/confident-ai/deepeval) metrics: `FraudAccuracyMetric`, `ReasoningFidelityMetric`, `ContextSensitivityMetric`, `GroundedAccuracyMetric`.

---

## Context Variants

Each transcript is classified three ways:

- **bad_context** — first 3 dialogue turns only (truncated, no resolution)
- **medium_context** — full dialogue as plain text
- **good_context** — structured block with inferred INTENT, KEY SIGNALS bullet list, full transcript, and a one-sentence SUMMARY (heuristic, no extra LLM call)

---

## Models

- `open-mistral-nemo` — small, fast
- `mistral-small-2503` — mid-size
- `mistral-large-latest` — frontier

---

## Dataset

`data/transcripts.json` — 25 synthetic transcripts:

- **Obvious fraud** (T001–T006): bank impersonation, IRS threat, tech support, grandparent scam, prize scam, utility shutoff
- **Subtle fraud** (T007–T015): bank refund verification, delivery customs fee, insurance overpayment, medical billing, job equipment fee, charity fraud, psychological isolation + surveillance, 2FA manipulation, investment pitch
- **Non-fraud** (T016–T025): real bank fraud alert, doctor reminder, customer service, family call, insurance renewal, utility outage, school call, vendor follow-up, friend call, real estate agent

---

## Setup

```bash
git clone https://github.com/mankala29/trace-guard
cd trace-guard

pyenv local 3.11.5          # deepeval requires 3.10+
pip install -r requirements.txt

cp .env.example .env        # add your MISTRAL_API_KEY
```

`.env`:
```
MISTRAL_API_KEY=your_key_here
```

---

## Running

**Full evaluation** (225 classifications × 3 Pass³ trials = 675 API calls):
```bash
python evaluate.py
# → results/results.csv
# → summary table printed to stdout
```

**LLM-as-judge** (reads results.csv, judges reasoning quality with mistral-large):
```bash
python llm_judge.py
python llm_judge.py --model nemo   # filter to one model
# → results/judge_results.csv
```

**Counterfactual sensitivity** (strips fraud signals, re-classifies):
```bash
python counterfactual_eval.py
# → results/counterfactual_results.json
```

**Temperature Pass³** (stability at temp=0.7 on harder subset):
```bash
python temperature_pass3.py
```

**Tests**:
```bash
pytest tests/
```

---

## Key Findings

- **Model size beats context quality**: mistral-large at 96% accuracy on bad_context outperforms nemo on good_context
- **Grounded accuracy gap**: nemo correct 93% of the time but only grounded 52% — often right for the wrong reasons
- **Counterfactual sensitivity**: larger models are more sensitive to signal removal (higher flip rate when fraud markers are stripped), confirming they actually read them
- **False negatives**: nemo and small both miss subtle fraud cases that large catches
- **Temperature stability**: nemo's Pass³ rate *drops* on good_context at temp=0.7 — richer context introduces sampling variance for smaller models

---

## Project Structure

```
trace-guard/
├── data/
│   └── transcripts.json        # 25 synthetic transcripts
├── tests/
│   ├── test_classify.py
│   ├── test_context_builder.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── classify.py                 # Mistral API calls + Pass³ + retry logic
├── context_builder.py          # bad / medium / good context variants
├── counterfactual_eval.py      # signal-stripping + re-classification
├── deepeval_metrics.py         # custom DeepEval BaseMetric subclasses
├── evaluate.py                 # main pipeline + summary reporting
├── llm_judge.py                # LLM-as-judge scoring
├── temperature_pass3.py        # temp=0.7 stability analysis
└── requirements.txt
```
