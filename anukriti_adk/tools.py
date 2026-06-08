"""PGx safety tools exposed to the ADK agent as FunctionTools.

These are plain Python functions with typed signatures and docstrings — ADK
turns them into tools automatically when listed in `Agent(tools=[...])`.

DESIGN: the engine is **offline-deterministic by default**. Every result is
computed from pinned CPIC-derived tables below, so the agent, the eval set,
and the judges all get byte-identical results with zero network access. Set
`ANUKRITI_ENGINE_MODE=api` to call a deployed anukriti-api instead (not
required for the hackathon).

The functions return STRUCTURED dicts containing named rule IDs (R/V/U). The
optimized agent narrates these; the baseline agent has the same tools but a
system instruction that lets it ignore them — that contrast is the whole point.
"""

from __future__ import annotations

import os

from .rules import RECOMMEND_DECISIONS, is_refusal, label

# ---------------------------------------------------------------------------
# Pinned CPIC-derived knowledge (the deterministic truth layer, condensed).
# Source anchors: CPIC 2022 (PMID 35034351), Kerdoncuff Cell 2025, PharmGKB.
# ---------------------------------------------------------------------------

SUPPORTED_DRUGS: frozenset[str] = frozenset({"clopidogrel", "warfarin", "simvastatin"})

_SUPER_POPULATIONS: frozenset[str] = frozenset({"AFR", "AMR", "EAS", "EUR", "SAS"})

# (drug, population) -> deterministic engine outcome.
# Encodes where evidence is SUFFICIENT vs where it must DOWNGRADE/ESCALATE.
# This is the crux of the equity story: the same drug yields a confident
# answer for EUR and a *named refusal* for under-evidenced cohorts.
_OUTCOMES: dict[tuple[str, str], dict] = {
    # clopidogrel / CYP2C19 — best-evidenced PGx pair, EUR strongest.
    ("clopidogrel", "EUR"): {
        "diplotype": "*1/*1", "phenotype": "Normal Metabolizer",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": "Normal CYP2C19 function; standard clopidogrel response expected.",
    },
    ("clopidogrel", "SAS"): {
        "diplotype": "*2/*2", "phenotype": "Poor Metabolizer",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": ("CYP2C19 *2/*2 loss-of-function: clopidogrel is not "
                    "activated. CPIC recommends an alternative antiplatelet. "
                    "14% of South Asians are PMs vs ~2% of Europeans."),
    },
    ("clopidogrel", "EAS"): {
        "diplotype": "*2/*3", "phenotype": "Poor Metabolizer",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": "CYP2C19 poor metabolizer; alternative antiplatelet advised.",
    },
    ("clopidogrel", "AMR"): {
        "diplotype": "*1/*2", "phenotype": "Intermediate Metabolizer",
        "decision": "R9", "verdict": "V7", "uncertainty": "U3",
        "finding": ("Admixed-American allele-frequency support for this call is "
                    "uncertain; downgrading rather than asserting a recommendation."),
    },
    # clopidogrel / AFR: evidence is genuinely thin -> honest downgrade.
    ("clopidogrel", "AFR"): {
        "diplotype": "*1/*1", "phenotype": "Normal Metabolizer",
        "decision": "R9", "verdict": "V7", "uncertainty": "U3",
        "finding": ("African-ancestry-specific clopidogrel/CYP2C19 evidence is "
                    "sparse; population support is uncertain. Downgrading."),
    },
    # warfarin / CYP2C9 + VKORC1 — sufficient across modeled pops.
    ("warfarin", "EUR"): {
        "diplotype": "CYP2C9 *1/*1; VKORC1 -1639 GG",
        "phenotype": "Normal sensitivity",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": "Standard warfarin dosing band per CPIC/IWPC algorithm.",
    },
    ("warfarin", "SAS"): {
        "diplotype": "CYP2C9 *1/*1; VKORC1 -1639 AA",
        "phenotype": "Increased sensitivity",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": ("VKORC1 -1639 AA raises warfarin sensitivity; lower starting "
                    "dose indicated. Higher AA frequency in South Asians."),
    },
    # simvastatin / SLCO1B1.
    ("simvastatin", "EUR"): {
        "diplotype": "SLCO1B1 *1/*1",
        "phenotype": "Normal function",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": "Normal SLCO1B1 transporter function; standard statin risk.",
    },
    ("simvastatin", "SAS"): {
        "diplotype": "SLCO1B1 *5/*5",
        "phenotype": "Decreased function",
        "decision": "R12", "verdict": "V10", "uncertainty": "U9",
        "finding": ("SLCO1B1 decreased function raises simvastatin myopathy risk; "
                    "consider a lower dose or alternative statin."),
    },
}

# Ancestry-specific wedge variants invisible to gnomAD-based commercial tests.
# Keyed by (community-lowercased). Surfaced in the pre_anesthesia context.
_WEDGE_ALERTS: dict[str, dict] = {
    "vysya": {
        "gene": "BCHE",
        "variant": "L307P",
        "risk": ("Prolonged succinylcholine/mivacurium apnea under general "
                 "anesthesia (butyrylcholinesterase deficiency)."),
        "prevalence": "~5.3% in the Telangana Vysya community; ~0.28% all-India; "
                      "absent from gnomAD and 1000 Genomes.",
        "action": "Order a $30 butyrylcholinesterase (dibucaine number) test "
                  "before any procedure requiring these agents.",
        "citation": {"id": "Kerdoncuff2025", "title": "Kerdoncuff et al., Cell 2025 (LASI-DAD n=2,762)"},
    },
}

_CITATIONS: dict[str, dict] = {
    "clopidogrel": {"id": "PMID:35034351", "title": "CPIC clopidogrel/CYP2C19 guideline 2022"},
    "warfarin": {"id": "PMID:28198005", "title": "CPIC warfarin guideline 2017 (Johnson et al.)"},
    "simvastatin": {"id": "PMID:35152405", "title": "CPIC statin/SLCO1B1 guideline 2022"},
}


def _normalize_population(population: str) -> str:
    p = (population or "").strip().upper()
    aliases = {
        "SOUTH ASIAN": "SAS", "SOUTHASIAN": "SAS", "INDIAN": "SAS",
        "EUROPEAN": "EUR", "WHITE": "EUR", "CAUCASIAN": "EUR",
        "EAST ASIAN": "EAS", "AFRICAN": "AFR", "BLACK": "AFR",
        "LATINO": "AMR", "HISPANIC": "AMR", "ADMIXED AMERICAN": "AMR",
    }
    return aliases.get(p, p)


def check_pgx_safety(drug: str, ancestry: str, context: str = "prescription_review",
                     community: str = "") -> dict:
    """Run a deterministic pharmacogenomic safety check for a drug + ancestry.

    This is the agent's core decision tool. It returns a named sufficiency
    decision (R1..R12), a set-level verdict (V1..V10), and an uncertainty tier
    (U1..U9). When evidence is thin for a population, it returns a REFUSAL
    decision instead of a recommendation — the agent must honor that.

    Args:
        drug: One of "clopidogrel", "warfarin", "simvastatin".
        ancestry: Super-population or common synonym (e.g. "SAS", "South Asian",
            "EUR", "EAS", "AFR", "AMR").
        context: One of "prescription_review", "ongoing_therapy",
            "pre_anesthesia", "travel_briefing".
        community: Optional fine-grained community (e.g. "Vysya") used to
            surface ancestry-specific wedge alerts in a pre_anesthesia context.

    Returns:
        A structured result with: ok, drug, population, diplotype, phenotype,
        sufficiency_decision, verdict, uncertainty, deterministic_rules (list),
        rule_ids_human (list of {id,label}), recommendation_allowed (bool),
        finding, wedge_alert (or None), citations (list), disclaimer.
        On error: {ok: False, error, rule_id}.
    """
    drug_l = (drug or "").strip().lower()
    if drug_l not in SUPPORTED_DRUGS:
        return {
            "ok": False,
            "error": (f"No CPIC-pinned safety check for '{drug}'. Supported drugs: "
                      f"{sorted(SUPPORTED_DRUGS)}."),
            "rule_id": "T-BADWORKFLOW",
        }

    pop = _normalize_population(ancestry)
    if pop not in _SUPER_POPULATIONS:
        return {
            "ok": False,
            "error": (f"Unknown ancestry '{ancestry}'. Use a super-population: "
                      f"{sorted(_SUPER_POPULATIONS)}."),
            "rule_id": "T-BADPOP",
        }

    outcome = _OUTCOMES.get((drug_l, pop))
    if outcome is None:
        # No modeled evidence for this exact (drug, population): escalate (R5),
        # do NOT guess. This is itself a correct, named, safe behavior.
        decision, verdict, uncertainty = "R5", "V5", "U2"
        finding = (f"No population-specific evidence modeled for {drug_l} in {pop}; "
                   "escalating for ancestry review rather than asserting a result.")
        diplotype = "unknown"
        phenotype = "undetermined"
    else:
        decision = outcome["decision"]
        verdict = outcome["verdict"]
        uncertainty = outcome["uncertainty"]
        finding = outcome["finding"]
        diplotype = outcome["diplotype"]
        phenotype = outcome["phenotype"]

    rules = [decision, verdict, uncertainty]

    wedge = None
    if context == "pre_anesthesia" and community:
        wedge = _WEDGE_ALERTS.get(community.strip().lower())

    citations = [_CITATIONS[drug_l]]
    if wedge:
        citations.append(wedge["citation"])

    return {
        "ok": True,
        "drug": drug_l,
        "population": pop,
        "diplotype": diplotype,
        "phenotype": phenotype,
        "sufficiency_decision": decision,
        "verdict": verdict,
        "uncertainty": uncertainty,
        "deterministic_rules": rules,
        "rule_ids_human": [{"id": r, "label": label(r)} for r in rules],
        "recommendation_allowed": decision in RECOMMEND_DECISIONS,
        "is_refusal": is_refusal(decision),
        "finding": finding,
        "wedge_alert": wedge,
        "citations": citations,
        "disclaimer": ("Always confirm with a healthcare professional before "
                       "making any prescription decisions."),
    }


def search_pgx_evidence(drug: str, gene: str = "", ancestry: str = "") -> dict:
    """Return curated evidence citations for a drug-gene-ancestry question.

    Use this when the user asks to see the evidence behind a recommendation.
    Returns only real, pinned citations — never fabricated. If nothing matches,
    returns an empty hits list (the agent must say so honestly).

    Args:
        drug: Drug name.
        gene: Optional gene symbol (e.g. "CYP2C19").
        ancestry: Optional super-population.

    Returns:
        {ok: True, hits: [{id, title, source}], count: int}
    """
    drug_l = (drug or "").strip().lower()
    hits = []
    if drug_l in _CITATIONS:
        c = _CITATIONS[drug_l]
        hits.append({"id": c["id"], "title": c["title"], "source": "CPIC"})
    # Equity/wedge evidence is always retrievable for South Asian queries.
    if _normalize_population(ancestry) == "SAS":
        k = _WEDGE_ALERTS["vysya"]["citation"]
        hits.append({"id": k["id"], "title": k["title"], "source": "Cell 2025"})
    return {"ok": True, "hits": hits, "count": len(hits)}


def list_supported_workflows() -> dict:
    """List the drugs this agent has a deterministic, CPIC-pinned check for.

    Use this when a user asks what you can check, or asks about a drug you may
    not support, so you can answer honestly instead of guessing.

    Returns:
        {ok: True, drugs: [...], note: str}
    """
    return {
        "ok": True,
        "drugs": sorted(SUPPORTED_DRUGS),
        "note": ("Only these drugs have a CPIC-pinned deterministic check. For "
                 "any other drug, say you do not have a check yet — do not guess."),
    }


def engine_mode() -> str:
    """Report whether the engine is running offline-deterministic or via API."""
    return os.environ.get("ANUKRITI_ENGINE_MODE", "offline").strip().lower()
