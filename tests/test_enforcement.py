"""Tests for the runtime enforcement gate (the hard GenerativeBoundary).

These verify the after_model decision logic directly — the part that rewrites
an unsafe model draft into a named, safe refusal — without needing an LLM or a
network. This is what proves the guardrail does not depend on model cooperation.

Run: pytest tests/test_enforcement.py
"""

from __future__ import annotations

from anukriti_adk.enforcement import (
    evaluate_model_output,
    latest_refusal,
    model_text_asserts_recommendation,
    safe_refusal_text,
)
from anukriti_adk.tools import check_pgx_safety


# ---------------------------------------------------------------------------
# Recommendation detection.
# ---------------------------------------------------------------------------

def test_detects_positive_recommendation():
    assert model_text_asserts_recommendation(
        "Yes, clopidogrel is safe for this patient — go ahead and prescribe it."
    ) is True


def test_does_not_flag_a_named_refusal_as_recommendation():
    # Contains "recommend" but as an explicit refusal — must NOT be flagged.
    assert model_text_asserts_recommendation(
        "I cannot recommend this drug; the engine returned R9 (population "
        "evidence uncertain), so I'm downgrading."
    ) is False


def test_empty_text_is_not_a_recommendation():
    assert model_text_asserts_recommendation("") is False


# ---------------------------------------------------------------------------
# The hard gate: rewrite on violation.
# ---------------------------------------------------------------------------

def test_rewrites_recommendation_over_engine_refusal():
    refusal = check_pgx_safety("clopidogrel", "AFR")  # R9 refusal
    assert refusal["is_refusal"] is True
    unsafe_draft = "Clopidogrel is safe for this patient. You can go ahead."
    decision = evaluate_model_output(unsafe_draft, [refusal])
    assert decision["action"] == "rewrite"
    assert "override_recommendation" in decision["violations"]
    assert "R9" in decision["replacement"]
    assert "healthcare professional" in decision["replacement"]


def test_rewrites_fabricated_citation():
    ok_result = check_pgx_safety("clopidogrel", "SAS")  # R12, allowed
    # Model invents a PMID that no tool produced.
    draft = "Clopidogrel works for this patient [PMID:99999999]."
    decision = evaluate_model_output(draft, [ok_result])
    assert decision["action"] == "rewrite"
    assert "fabricate_claim" in decision["violations"]
    assert "PMID:99999999" in decision["fabricated"]


def test_passes_a_grounded_recommendation():
    ok_result = check_pgx_safety("clopidogrel", "EUR")  # R12, allowed, no refusal
    # A grounded narration citing only the real engine citation.
    cited = ok_result["citations"][0]["id"]
    draft = (
        f"The engine returned R12 (sufficient). Normal CYP2C19 function; "
        f"standard clopidogrel response expected [{cited}]."
    )
    decision = evaluate_model_output(draft, [ok_result])
    assert decision["action"] == "pass"
    assert decision["violations"] == []


def test_passes_a_correct_named_refusal():
    refusal = check_pgx_safety("clopidogrel", "AFR")
    draft = (
        "I cannot recommend clopidogrel here. The engine returned R9 "
        "(population evidence uncertain)."
    )
    decision = evaluate_model_output(draft, [refusal])
    assert decision["action"] == "pass"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def test_latest_refusal_finds_the_refusal():
    ok = check_pgx_safety("clopidogrel", "EUR")
    refusal = check_pgx_safety("clopidogrel", "AFR")
    assert latest_refusal([ok, refusal]) is refusal
    assert latest_refusal([ok]) is None


def test_safe_refusal_text_names_the_rule():
    refusal = check_pgx_safety("clopidogrel", "AFR")
    txt = safe_refusal_text([refusal])
    assert "R9" in txt
    assert "clopidogrel" in txt
