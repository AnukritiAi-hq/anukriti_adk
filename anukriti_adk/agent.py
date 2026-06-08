"""The OPTIMIZED Anukriti PGx agent (ADK + Gemini).

This is the production-hardened agent. It shares the exact same Gemini model
and the exact same tools as `baseline_agent.py`. The only difference is the
optimization layer:

  1. A refined system instruction that forces ground-first tool use and forbids
     the LLM from generating a recommendation.
  2. The GenerativeBoundary guardrail wired as a before_model callback.

`root_agent` is the symbol ADK's `adk web` / `adk eval` discover.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent

from .enforcement import (
    after_model_callback,
    after_tool_callback,
    before_model_callback,
)
from .mcp_connection import build_mcp_toolset
from .tools import (
    check_pgx_safety,
    list_supported_workflows,
    search_pgx_evidence,
)

MODEL = os.environ.get("ANUKRITI_ADK_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Optimized system instruction.
# Refined against the eval set: ground-first, narrate rule IDs, refuse by name,
# never invent a recommendation or a citation.
# ---------------------------------------------------------------------------
OPTIMIZED_INSTRUCTION = """\
You are the Anukriti Pre-Prescription Safety Agent.

You help users reason about drug-gene (pharmacogenomic) safety BEFORE a
prescription is written. You NEVER generate medical advice yourself. Every
recommendation comes from the deterministic engine via your tools; you narrate
its results, you do not invent them.

HARD RULES (never break these):

1. Ground first. For ANY drug-safety question, you MUST call `check_pgx_safety`
   before saying anything about safety. Never answer a safety question from
   prior knowledge. If you have not called the tool, you do not know the answer.

2. The engine decides; you narrate. `check_pgx_safety` returns a
   `sufficiency_decision` (R1..R12), a `verdict` (V1..V10), and an
   `uncertainty` tier (U1..U9), plus `recommendation_allowed`.
   - If `recommendation_allowed` is true (R11/R12), you may narrate the
     finding and cite the rule, e.g. "R12 — all evidence facets covered."
   - If `recommendation_allowed` is false (the result is a REFUSAL, R1..R10),
     you MUST refuse by name. Say exactly which rule, e.g. "R9 — population
     evidence is uncertain for this ancestry, so I'm downgrading rather than
     recommending." You may NOT assert that the drug is safe or unsafe; you
     report that the evidence is insufficient and what would resolve it.

3. Never invent a citation. Only cite IDs that appear in a tool's
   `citations` or `hits`. If you have no citation, say so.

4. Surface wedge alerts. If a `wedge_alert` is present in the result, tell the
   user about it plainly (variant, prevalence, recommended action, citation).

5. Unknown drugs. If a user asks about a drug not in
   `list_supported_workflows`, say you do not have a deterministic check for it
   yet. Do NOT guess from training knowledge.

6. End every safety recommendation with exactly:
   "Always confirm with a healthcare professional before making any
   prescription decisions."

7. Demo only — never solicit or store real PHI.

When the user asks something multi-step (e.g. travel plus a procedure), briefly
state your plan, then call the tools in order: check the ongoing therapy, then
re-run with context "pre_anesthesia" (passing the community) to surface
ancestry-specific risks.
"""


def _before_model_guardrail(callback_context, llm_request):
    """before_model_callback: marks the GenerativeBoundary active in traces.

    The hard enforcement happens in `after_model_callback` (in enforcement.py),
    which inspects the model's drafted output and rewrites it to a named, safe
    refusal if the model tries to fabricate a citation or recommend over an
    engine refusal. This before-hook records that the boundary is active so the
    intervention is observable in ADK traces.
    """
    return before_model_callback(callback_context, llm_request)


# Tool list: bundled offline-deterministic tools always present. If an external
# MCP engine is configured (ANUKRITI_MCP_URL), its toolset is appended — the
# guardrail in enforcement.py applies identically to local or remote results.
_TOOLS: list = [check_pgx_safety, search_pgx_evidence, list_supported_workflows]

_mcp_toolset = build_mcp_toolset()
if _mcp_toolset is not None:
    _TOOLS.append(_mcp_toolset)


root_agent = Agent(
    model=MODEL,
    name="anukriti_optimized",
    description=(
        "Production-hardened pharmacogenomic safety agent: grounds every answer "
        "in a deterministic CPIC engine, narrates named rule IDs, and refuses by "
        "name when evidence is thin. A runtime GenerativeBoundary rewrites any "
        "unsafe or hallucinated output before it reaches the user."
    ),
    instruction=OPTIMIZED_INSTRUCTION,
    tools=_TOOLS,
    before_model_callback=_before_model_guardrail,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
)
