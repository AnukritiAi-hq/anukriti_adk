"""Tests for the instruction optimizer and the MCP connection seam.

Locks in the Track-2 claims:
  • the programmatic refinement loop actually selects the full-contract
    instruction as the winner (not by hand), and
  • the MCP seam is off-by-default and reports its config truthfully.

Run: pytest tests/test_optimizer.py
"""

from __future__ import annotations

import os

from anukriti_adk import mcp_connection
from harness.optimize_instructions import VARIANTS, optimize


# ---------------------------------------------------------------------------
# Instruction optimizer.
# ---------------------------------------------------------------------------

def test_optimizer_selects_full_contract_winner():
    scores = optimize()
    assert scores[0].name == "v6_full_contract"


def test_optimizer_winner_has_zero_unsafe_recommendations():
    scores = optimize()
    assert scores[0].unsafe_recommendations == 0


def test_naive_variant_is_worst_and_unsafe():
    scores = optimize()
    naive = next(s for s in scores if s.name == "v1_naive")
    assert naive.unsafe_recommendations >= 1
    # Naive must rank strictly below the winner.
    assert naive.composite < scores[0].composite


def test_composite_is_monotonic_in_added_clauses():
    """Each added clause should not reduce the composite score (refinement)."""
    scores = {s.name: s.composite for s in optimize()}
    order = [v.name for v in VARIANTS]  # v1..v6 in clause-addition order
    composites = [scores[n] for n in order]
    for earlier, later in zip(composites, composites[1:]):
        assert later >= earlier - 1e-9


# ---------------------------------------------------------------------------
# MCP seam.
# ---------------------------------------------------------------------------

def test_mcp_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ANUKRITI_MCP_URL", raising=False)
    assert mcp_connection.mcp_enabled() is False
    assert mcp_connection.build_mcp_toolset() is None


def test_mcp_config_reports_truthfully(monkeypatch):
    monkeypatch.setenv("ANUKRITI_MCP_URL", "https://example.invalid/mcp")
    monkeypatch.delenv("ANUKRITI_MCP_TOKEN", raising=False)
    cfg = mcp_connection.mcp_config()
    assert cfg["enabled"] is True
    assert cfg["auth"] == "none"
    monkeypatch.setenv("ANUKRITI_MCP_TOKEN", "secret")
    assert mcp_connection.mcp_config()["auth"] == "header-bearer"
