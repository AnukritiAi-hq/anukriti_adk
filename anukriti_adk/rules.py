"""Named-rule taxonomy (R/V/U) surfaced by the deterministic PGx engine.

These are the human-readable labels for the closed-enum refusal/sufficiency
vocabulary produced by the Anukriti evidence-sufficiency layer. The agent
narrates these IDs verbatim; it never invents a recommendation. Mirrors the
taxonomy in anukriti-swarm/core/evidence_sufficiency.
"""

from __future__ import annotations

# Sufficiency decisions (R1..R12) -> the decision engine's verdict.
RULE_LABELS: dict[str, str] = {
    # Sufficiency rules (R1..R12)
    "R1": "blocked: hard conflict in evidence",
    "R2": "blocked: phenotype evidence missing",
    "R3": "blocked: recommendation evidence missing",
    "R4": "abstain: provenance incomplete",
    "R5": "escalate: population evidence missing",
    "R6": "request more: CPIC evidence missing",
    "R7": "request more: allele evidence missing",
    "R8": "downgrade: recommendation uncertain",
    "R9": "downgrade: population evidence uncertain",
    "R10": "downgrade: evidence uncertain",
    "R11": "pass with caveat: conflict evidence soft",
    "R12": "sufficient: all facets covered, no conflict",
    # Set-level verdicts (V1..V10)
    "V1": "refuted: named avoid/use conflict",
    "V2": "conflicting: hard conflict",
    "V3": "insufficient: phenotype missing",
    "V4": "insufficient: recommendation missing",
    "V5": "insufficient: facet missing",
    "V6": "uncertain: empty knowledge graph paths",
    "V7": "uncertain: population evidence uncertain",
    "V8": "uncertain: facet evidence uncertain",
    "V9": "uncertain: conflict evidence soft",
    "V10": "supported: all evidence aligned",
    # Uncertainty tiers (U1..U9)
    "U1": "unsafe: hard conflict",
    "U2": "high: missing facet",
    "U3": "high: population evidence uncertain",
    "U4": "high: empty knowledge graph paths",
    "U5": "high: multiple uncertain facets",
    "U6": "moderate: conflict evidence soft",
    "U7": "moderate: one non-core facet uncertain",
    "U8": "moderate: single knowledge-graph path",
    "U9": "low: all evidence aligned",
}

# Decisions that mean "the agent must NOT emit a positive recommendation".
REFUSAL_DECISIONS: frozenset[str] = frozenset(
    {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"}
)

# Decisions that allow a recommendation (possibly with a caveat).
RECOMMEND_DECISIONS: frozenset[str] = frozenset({"R11", "R12"})


def label(rule_id: str) -> str:
    """Return the plain-English label for a rule ID, or the ID itself."""
    return RULE_LABELS.get(rule_id, rule_id)


def is_refusal(decision: str) -> bool:
    """True if this sufficiency decision forbids a positive recommendation."""
    return decision in REFUSAL_DECISIONS
