"""Tests for the hosted testing surface (serve.py).

Verifies auth enforcement and that the /v1/guard endpoint actually rewrites an
unsafe draft via the runtime guardrail — the headline judge-facing demo.

Run: pytest tests/test_serve.py
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANUKRITI_API_KEY", "test-key")
    monkeypatch.delenv("ANUKRITI_ALLOW_NOAUTH", raising=False)
    from anukriti_adk import serve
    importlib.reload(serve)
    return TestClient(serve.app)


def test_healthz_no_auth(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_card_no_auth(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "anukriti-adk"


def test_check_requires_auth(client):
    r = client.post("/v1/check", json={"drug": "clopidogrel", "ancestry": "SAS"})
    assert r.status_code == 401


def test_check_with_auth(client):
    r = client.post(
        "/v1/check",
        json={"drug": "clopidogrel", "ancestry": "SAS"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sufficiency_decision"] == "R12"
    assert body["recommendation_allowed"] is True


def test_guard_rewrites_unsafe_draft(client):
    r = client.post(
        "/v1/guard",
        json={
            "drug": "clopidogrel",
            "ancestry": "AFR",
            "draft_answer": "Clopidogrel is safe — go ahead and prescribe.",
        },
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["was_rewritten"] is True
    assert "R9" in body["final_answer"]
    assert "override_recommendation" in body["guardrail"]["violations"]


def test_guard_passes_safe_draft(client):
    r = client.post(
        "/v1/guard",
        json={
            "drug": "clopidogrel",
            "ancestry": "AFR",
            "draft_answer": "I cannot recommend this; the engine returned R9 (population uncertain).",
        },
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert r.json()["was_rewritten"] is False


def test_bearer_auth_also_accepted(client):
    r = client.get("/v1/workflows", headers={"Authorization": "Bearer test-key"})
    assert r.status_code == 200
    assert "clopidogrel" in r.json()["drugs"]
