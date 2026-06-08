"""Runtime enforcement callbacks — the optimization, wired into ADK itself.

This module turns the GenerativeBoundary from a *documented contract* into a
*runtime-enforced* guarantee. It attaches three ADK callbacks to the optimized
agent:

  • after_tool_callback   — records every engine (tool) result on the session,
                            so later stages know what the deterministic layer
                            actually decided. This is the "ground truth" the
                            model is not allowed to contradict.

  • before_model_callback — observability + contract reinforcement. Stamps the
                            boundary's presence into agent state so it shows up
                            in ADK traces.

  • after_model_callback  — THE HARD GATE. Inspects the model's drafted text and:
                              1. rejects fabricated citations (CitationValidator),
                              2. rejects a positive recommendation issued over an
                                 engine refusal (override_recommendation), and
                              3. on violation, REPLACES the unsafe text with a
                                 named, safe refusal instead of letting it through.

The key property: even if the LLM ignores its system instruction and tries to
emit an unsafe recommendation or a hallucinated PMID, the after_model_callback
rewrites the response before the user ever sees it. The guardrail does not rely
on the model's cooperation.

All callbacks are pure-Python and require no network access, so they run
identically in `adk web`, `adk eval`, and the offline harness.
"""

from __future__ import annotations

import re
from typing import Any

from .boundary import (
    GenerativeBoundaryViolation,
    allowed_citations,
    validate_citations,
)
from .rules import is_refusal, label

# Session-state keys (namespaced to avoid collisions with user state).
_STATE_TOOL_RESULTS = "anukriti:tool_results"
_STATE_BOUNDARY_STAMP = "anukriti:boundary_active"
_STATE_ENFORCEMENT_LOG = "anukriti:enforcement_log"

# Phrases that count as the model asserting a positive recommendation. Kept
# conservative: we only flip to a refusal when the model both (a) recommends and
# (b) the engine refused — never on a legitimate engine-approved recommendation.
_RECOMMEND_PATTERNS = re.compile(
    r"\b(is safe|you can safely|recommend(?:ed|ing)?|go ahead|prescribe|"
    r"standard dose is fine|no concerns|safe to (?:take|use|prescribe)|"
    r"proceed with)\b",
    re.IGNORECASE,
)

# Phrases that indicate the model is correctly refusing / hedging. If present,
# we do not treat the turn as an unsafe recommendation even if a recommend-word
# also appears (e.g. "I cannot recommend").
_REFUSAL_SIGNALS = re.compile(
    r"\b(cannot recommend|can't recommend|insufficient|uncertain|downgrad|"
    r"refus|escalat|do not have|don't have|not enough evidence|R\d{1,2}\b|"
    r"V\d{1,2}\b|U\d\b)",
    re.IGNORECASE,
)


def _state(callback_context: Any) -> Any:
    """Best-effort access to the mutable session state across ADK versions."""
    state = getattr(callback_context, "state", None)
    return state if state is not None else {}


def _record_tool_result(callback_context: Any, result: dict) -> None:
    state = _state(callback_context)
    try:
        existing = list(state.get(_STATE_TOOL_RESULTS, []))
    except Exception:
        existing = []
    existing.append(result)
    try:
        state[_STATE_TOOL_RESULTS] = existing
    except Exception:
        pass


def _get_tool_results(callback_context: Any) -> list[dict]:
    state = _state(callback_context)
    try:
        return list(state.get(_STATE_TOOL_RESULTS, []))
    except Exception:
        return []


def _append_enforcement_log(callback_context: Any, entry: dict) -> None:
    state = _state(callback_context)
    try:
        log = list(state.get(_STATE_ENFORCEMENT_LOG, []))
    except Exception:
        log = []
    log.append(entry)
    try:
        state[_STATE_ENFORCEMENT_LOG] = log
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pure decision functions (unit-testable without ADK objects).
# ---------------------------------------------------------------------------

def model_text_asserts_recommendation(text: str) -> bool:
    """Heuristic: does the model's text assert a positive recommendation?

    Returns False if the text contains an explicit refusal/hedge signal, so we
    never misclassify a correct named refusal as an unsafe recommendation.
    """
    if not text:
        return False
    if _REFUSAL_SIGNALS.search(text):
        return False
    return bool(_RECOMMEND_PATTERNS.search(text))


def latest_refusal(tool_results: list[dict]) -> dict | None:
    """Return the most recent engine tool result that was a refusal, if any."""
    for r in reversed(tool_results or []):
        if r.get("ok") and r.get("is_refusal"):
            return r
    return None


def safe_refusal_text(tool_results: list[dict]) -> str:
    """Build a named, safe refusal message from the engine's own decision.

    This is what replaces an unsafe model draft. It cites the exact rule the
    engine returned, so the rewritten answer is still grounded and honest.
    """
    refusal = latest_refusal(tool_results)
    if refusal is None:
        return (
            "I can't make a recommendation here: I don't have a sufficient, "
            "evidence-grounded result from the deterministic engine for this "
            "case. Always confirm with a healthcare professional before making "
            "any prescription decisions."
        )
    decision = refusal.get("sufficiency_decision", "R-?")
    drug = refusal.get("drug", "this drug")
    pop = refusal.get("population", "this ancestry")
    finding = refusal.get("finding", "")
    return (
        f"I can't recommend {drug} for a {pop} patient as safe or unsafe. The "
        f"deterministic engine returned **{decision} — {label(decision)}**. "
        f"{finding} What would resolve it: ancestry-specific evidence for this "
        f"(drug, population) pair. Always confirm with a healthcare professional "
        f"before making any prescription decisions."
    )


def evaluate_model_output(
    text: str,
    tool_results: list[dict],
) -> dict:
    """Pure enforcement decision over a model output + the engine results.

    Returns a dict:
        {
          "action": "pass" | "rewrite",
          "violations": [<action_name>, ...],
          "fabricated": [<citation_id>, ...],
          "replacement": <str|None>,   # set when action == "rewrite"
        }

    This is the single source of truth for the after_model gate and is unit
    tested directly, so the enforcement logic is verified without an LLM.
    """
    violations: list[str] = []
    replacement: str | None = None

    # 1. Citation validation (fabricate_claim guard).
    cite_check = validate_citations(text, tool_results)
    fabricated = cite_check["fabricated"]
    if fabricated:
        violations.append("fabricate_claim")

    # 2. Recommendation-over-refusal (override_recommendation guard).
    refusal = latest_refusal(tool_results)
    if refusal is not None and model_text_asserts_recommendation(text):
        violations.append("override_recommendation")

    if violations:
        replacement = safe_refusal_text(tool_results)
        return {
            "action": "rewrite",
            "violations": violations,
            "fabricated": fabricated,
            "replacement": replacement,
        }

    return {
        "action": "pass",
        "violations": [],
        "fabricated": [],
        "replacement": None,
    }


# ---------------------------------------------------------------------------
# ADK callbacks.
# ---------------------------------------------------------------------------

def after_tool_callback(tool, args, tool_context, tool_response):  # noqa: ANN001
    """Record the engine's structured result so later stages can enforce it.

    ADK passes the tool's return value as `tool_response`. We stash every
    structured engine result on the session state. Returning None keeps the
    original tool response unchanged.
    """
    if isinstance(tool_response, dict):
        _record_tool_result(tool_context, tool_response)
    return None


def before_model_callback(callback_context, llm_request):  # noqa: ANN001
    """Stamp the boundary's presence into state (observability) and proceed."""
    state = _state(callback_context)
    try:
        state[_STATE_BOUNDARY_STAMP] = True
    except Exception:
        pass
    return None


def after_model_callback(callback_context, llm_response):  # noqa: ANN001
    """THE HARD GATE: rewrite unsafe model output before the user sees it.

    If the model drafted a fabricated citation or a recommendation that
    contradicts an engine refusal, we replace the text with a named, safe
    refusal derived from the engine's own decision and log the violation.

    Returns a modified LlmResponse to override the model output, or None to
    let the original response pass.
    """
    # Extract the drafted text from the response content.
    content = getattr(llm_response, "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None) or []
    text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text.strip():
        return None

    tool_results = _get_tool_results(callback_context)
    decision = evaluate_model_output(text, tool_results)

    if decision["action"] == "pass":
        return None

    # Violation detected — rewrite the response.
    _append_enforcement_log(callback_context, {
        "violations": decision["violations"],
        "fabricated": decision["fabricated"],
        "original_text_prefix": text[:160],
    })

    replacement = decision["replacement"] or safe_refusal_text(tool_results)

    # Build a new LlmResponse with the safe text. Import locally so the module
    # imports cleanly even in the offline harness path.
    try:
        from google.adk.models import LlmResponse
        from google.genai import types

        new_content = types.Content(
            role="model",
            parts=[types.Part(text=replacement)],
        )
        return LlmResponse(content=new_content)
    except Exception:
        # If we cannot construct a replacement response, fail closed by raising
        # the boundary violation rather than letting unsafe text through.
        raise GenerativeBoundaryViolation(
            decision["violations"][0],
            detail="after_model_callback could not rewrite; failing closed.",
        )


def enforcement_summary(callback_context: Any) -> dict:
    """Expose what the guardrail did this session (for traces / debugging)."""
    state = _state(callback_context)
    return {
        "boundary_active": bool(state.get(_STATE_BOUNDARY_STAMP, False)),
        "tool_results_recorded": len(_get_tool_results(callback_context)),
        "enforcement_log": list(state.get(_STATE_ENFORCEMENT_LOG, [])),
    }
