"""The BASELINE (naive) Anukriti PGx agent — the 'before optimization' artifact.

Same Gemini model. Same tools available. The ONLY difference from the optimized
agent is the system instruction: this one is a typical first-draft prompt that
is helpful and conversational but does NOT enforce ground-first tool use, does
NOT forbid the LLM from recommending, and does NOT constrain citations.

This is what most teams ship to a sandbox. It works on easy questions and
hallucinates confident answers on the hard edge cases — which is exactly the
failure the optimization removes, and what the eval set measures.

Exposed as `baseline_agent` and consumed by the optimization harness.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent

from .tools import (
    check_pgx_safety,
    list_supported_workflows,
    search_pgx_evidence,
)

MODEL = os.environ.get("ANUKRITI_ADK_MODEL", "gemini-2.5-flash")

# A naive, well-meaning first-draft instruction. Note what's MISSING: no
# ground-first mandate, no refusal contract, no citation discipline, no
# "the engine decides" rule.
BASELINE_INSTRUCTION = """\
You are a helpful pharmacogenomics assistant. Answer the user's questions about
whether a drug is safe given their genetics and ancestry. You have tools that
can look up drug-gene safety information if you find them useful. Be helpful,
clear, and reassuring, and give the user a concrete recommendation so they know
what to do next.
"""

baseline_agent = Agent(
    model=MODEL,
    name="anukriti_baseline",
    description="Naive first-draft pharmacogenomic assistant (pre-optimization).",
    instruction=BASELINE_INSTRUCTION,
    tools=[check_pgx_safety, search_pgx_evidence, list_supported_workflows],
)

# Allow `adk web .` / `adk eval` to target the baseline too, if pointed here.
root_agent = baseline_agent
