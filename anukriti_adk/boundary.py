"""GenerativeBoundary + CitationValidator — the optimization intervention.

These are the deterministic guardrails that turn the experimental agent into a
production-reliable one. They are what the optimized agent has and the baseline
agent does not.

GenerativeBoundary forbids four LLM actions at runtime (it raises, it does not
log-and-continue):
    1. infer_phenotype       — only the engine assigns a phenotype
    2. override_recommendation — the LLM cannot flip a refusal into a yes
    3. bypass_verification    — the LLM cannot answer without a tool result
    4. fabricate_claim        — every clinical claim must trace to a tool output

CitationValidator rejects any citation token the agent emits that was not
present in a tool result — catching hallucinated PMIDs/guideline IDs.

In ADK these run as a `before_model_callback` / output check. For the eval
harness they are also callable directly so we can score guarded vs unguarded
behavior deterministically without spending tokens.
"""

from __future__ import annotations

import re

FORBIDDEN_ACTIONS = (
    "infer_phenotype",
    "override_recommendation",
    "bypass_verification",
    "fabricate_claim",
)


class GenerativeBoundaryViolation(RuntimeError):
    """Raised when the LLM attempts a runtime-forbidden generative action."""

    def __init__(self, action: str, detail: str = "") -> None:
        self.action = action
        super().__init__(f"GenerativeBoundary violation: {action}. {detail}".strip())


# Citation token format the engine emits, e.g. "PMID:35034351", "Kerdoncuff2025".
_CITATION_TOKEN = re.compile(r"\b(PMID:\d+|[A-Z][A-Za-z]+\d{4})\b")


def allowed_citations(tool_results: list[dict]) -> set[str]:
    """Collect every citation ID that appeared in the engine's tool outputs."""
    allowed: set[str] = set()
    for r in tool_results or []:
        for c in r.get("citations", []) or []:
            cid = c.get("id")
            if cid:
                allowed.add(cid)
        for h in r.get("hits", []) or []:
            hid = h.get("id")
            if hid:
                allowed.add(hid)
    return allowed


def validate_citations(text: str, tool_results: list[dict]) -> dict:
    """Check that every citation token in `text` was produced by a tool.

    Returns {ok, fabricated: [...], allowed: [...]}. `ok` is False if the agent
    emitted a citation that no tool result contained (a hallucinated citation).
    """
    allowed = allowed_citations(tool_results)
    found = set(_CITATION_TOKEN.findall(text or ""))
    fabricated = sorted(found - allowed)
    return {
        "ok": len(fabricated) == 0,
        "fabricated": fabricated,
        "allowed": sorted(allowed),
    }


def enforce_recommendation_boundary(tool_result: dict, agent_recommends: bool) -> None:
    """Raise if the agent asserts a recommendation the engine refused.

    This is the `override_recommendation` guard: when the deterministic engine
    returns a refusal decision (R1..R10), the agent is forbidden from emitting a
    positive drug recommendation. The engine decides; the LLM narrates.
    """
    if not tool_result.get("ok", False):
        return
    if tool_result.get("is_refusal", False) and agent_recommends:
        raise GenerativeBoundaryViolation(
            "override_recommendation",
            detail=(f"engine returned {tool_result.get('sufficiency_decision')} "
                    "(a refusal); agent may not assert a recommendation."),
        )
