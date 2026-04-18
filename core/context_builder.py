"""
context_builder.py
Builds bad / medium / good context variants for each transcript.
No LLM calls — good_context uses keyword-based heuristics only.
"""

SIGNAL_KEYWORDS = [
    "verify", "account", "password", "urgent", "immediately", "threat",
    "arrest", "wire", "gift card", "code", "transfer", "suspended",
    "frozen", "limited time", "expires", "don't tell", "secret", "isolated",
    "pressure", "suspicious", "fee", "deposit", "refund", "remote access",
]

URGENCY_KEYWORDS = [
    "now", "immediately", "today", "minutes", "hours", "expires",
    "deadline", "urgent", "right away", "don't wait", "act fast",
]

ISOLATION_KEYWORDS = [
    "don't tell", "don't contact", "keep this between", "no one", "secret",
    "don't call", "watching", "lawyer will complicate", "can't tell anyone",
]


def _dialogue_to_text(dialogue: list[dict]) -> str:
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in dialogue)


def _detect_signals(dialogue: list[dict]) -> list[str]:
    signals = []
    full_text = " ".join(t["text"].lower() for t in dialogue)

    if any(k in full_text for k in ["password", "account number", "routing number", "pin", "ssn", "social security"]):
        signals.append("Requests sensitive credentials or account details")
    if any(k in full_text for k in ["gift card", "zelle", "venmo", "wire", "moneypack", "cryptocurrency", "bitcoin"]):
        signals.append("Requests untraceable payment method")
    if any(k in full_text for k in URGENCY_KEYWORDS):
        signals.append("Creates artificial urgency or time pressure")
    if any(k in full_text for k in ISOLATION_KEYWORDS):
        signals.append("Instructs victim to keep conversation secret / avoid outside help")
    if any(k in full_text for k in ["arrest", "warrant", "federal", "irs", "lawsuit", "legal action"]):
        signals.append("Uses legal threats or impersonates law enforcement")
    if any(k in full_text for k in ["verify", "confirm", "update your", "security check"]):
        signals.append("Solicits verification of personal information under a pretense")
    if any(k in full_text for k in ["code", "one-time", "2fa", "otp", "authentication code"]):
        signals.append("Attempts to intercept one-time authentication codes")
    if any(k in full_text for k in ["prize", "won", "winner", "congratulations", "selected"]):
        signals.append("Unsolicited prize or lottery claim")
    if any(k in full_text for k in ["fee", "deposit", "processing", "customs", "insurance fee"]):
        signals.append("Upfront fee required before receiving funds or prize")

    return signals if signals else ["No strong fraud signals detected"]


def _infer_intent(dialogue: list[dict]) -> str:
    full_text = " ".join(t["text"].lower() for t in dialogue)
    caller_text = " ".join(t["text"].lower() for t in dialogue if t["speaker"] == "Caller")

    if any(k in caller_text for k in ["arrest", "warrant", "irs", "tax debt"]):
        return "Coerce immediate payment using legal threats and fear of arrest"
    if any(k in caller_text for k in ["password", "account number", "routing number"]):
        return "Extract account credentials under the guise of security or refund processing"
    if any(k in caller_text for k in ["gift card", "zelle", "venmo", "wire"]):
        return "Obtain untraceable payment from victim"
    if any(k in caller_text for k in ["one-time code", "verification code", "otp", "2fa"]):
        return "Intercept authentication codes to hijack victim's account"
    if any(k in caller_text for k in ["don't tell", "don't contact", "keep this between", "watching"]):
        return "Isolate victim from support network to prevent intervention"
    if any(k in caller_text for k in ["remind", "appointment", "scheduled", "confirmation"]):
        return "Provide routine service notification — no financial ask"
    if any(k in caller_text for k in ["refund", "overcharged", "overpayment"]):
        return "Obtain banking details under the pretense of delivering a refund"
    if any(k in caller_text for k in ["won", "prize", "winner", "selected"]):
        return "Collect upfront fee by falsely claiming victim has won a prize"
    if any(k in caller_text for k in ["remote", "download", "anydesk", "virus", "infected"]):
        return "Gain remote access to victim's device under a tech support pretext"
    if any(k in full_text for k in ["coming over", "game saturday", "see you", "are you home"]):
        return "Personal or social communication with no financial component"

    return "Unclear — requires full transcript review"


def _one_line_summary(dialogue: list[dict], label_hint: str = "") -> str:
    caller_lines = [t["text"] for t in dialogue if t["speaker"] == "Caller"]
    first = caller_lines[0] if caller_lines else ""
    last_caller = caller_lines[-1] if caller_lines else ""

    if "arrest" in first.lower() or "warrant" in first.lower():
        return "Caller threatens arrest over alleged tax debt and demands immediate gift-card payment."
    if "bank" in first.lower() and "fraud" in first.lower():
        return "Caller impersonates bank fraud team and requests account credentials to 'reverse' transfers."
    if "microsoft" in first.lower() or "virus" in first.lower():
        return "Caller impersonates tech support and attempts to gain remote device access for a fee."
    if "won" in first.lower() or "prize" in first.lower():
        return "Caller falsely informs recipient of a prize win and demands an upfront processing fee."
    if "grandma" in first.lower() or "grandpa" in first.lower() or "jail" in last_caller.lower():
        return "Caller impersonates grandchild claiming to need emergency bail money."
    if "refund" in first.lower() or "overcharged" in first.lower():
        return "Caller claims to offer a refund but requests full account details to 'process' it."
    if "don't tell" in " ".join(t["text"].lower() for t in dialogue):
        return "Caller uses emotional manipulation and isolation tactics to extract a large wire transfer."
    if "one-time code" in " ".join(t["text"].lower() for t in dialogue).lower() or "verification code" in " ".join(t["text"].lower() for t in dialogue).lower():
        return "Caller impersonates bank security and extracts 2FA codes to compromise the victim's account."
    if "appointment" in first.lower() or "remind" in first.lower():
        return "Routine appointment reminder call with no financial request."
    if "game" in first.lower() or "coming over" in first.lower():
        return "Casual personal call about social plans."
    return "Caller initiates contact and engages recipient — full review recommended."


# ── Public API ─────────────────────────────────────────────────────────────────

def build_bad_context(transcript: dict) -> str:
    """First 3 turns of dialogue only (truncated, missing resolution)."""
    dialogue = transcript["dialogue"][:3]
    return _dialogue_to_text(dialogue)


def build_medium_context(transcript: dict) -> str:
    """Full transcript as plain dialogue text."""
    return _dialogue_to_text(transcript["dialogue"])


def build_good_context(transcript: dict) -> str:
    """
    Structured block with inferred intent, key signals, full transcript,
    and a 1-sentence summary. Generated by heuristics — no LLM.
    """
    dialogue = transcript["dialogue"]
    intent = _infer_intent(dialogue)
    signals = _detect_signals(dialogue)
    full_text = _dialogue_to_text(dialogue)
    summary = _one_line_summary(dialogue)

    signal_lines = "\n".join(f"  - {s}" for s in signals)

    return (
        f"INTENT: {intent}\n"
        f"KEY SIGNALS:\n{signal_lines}\n\n"
        f"FULL TRANSCRIPT:\n{full_text}\n\n"
        f"SUMMARY: {summary}"
    )


def build_all_contexts(transcript: dict) -> dict[str, str]:
    return {
        "bad_context": build_bad_context(transcript),
        "medium_context": build_medium_context(transcript),
        "good_context": build_good_context(transcript),
    }
