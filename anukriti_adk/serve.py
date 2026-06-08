"""serve.py — a minimal authenticated HTTP surface for judges to test the agent.

This is the "Testing access" deliverable: a hosted endpoint judges can call to
exercise the optimized agent's *decision + guardrail* path without provisioning
anything or spending tokens. It wraps the same deterministic engine + the same
runtime enforcement gate (`enforcement.evaluate_model_output`) the ADK agent
uses, so what you test here is exactly what the agent does.

Endpoints
---------
  GET  /healthz                      — liveness (no auth)
  GET  /                             — service card + how to call (no auth)
  POST /v1/check                     — run a PGx safety check (auth)
  POST /v1/guard                     — submit a draft answer; see the guardrail
                                       rewrite it if it's unsafe (auth)
  GET  /v1/workflows                 — list supported drugs (auth)

Auth
----
Set `ANUKRITI_API_KEY` in the environment. Requests must send it as
`X-API-Key: <key>` (or `Authorization: Bearer <key>`). If `ANUKRITI_API_KEY` is
unset the service refuses to start in a network context — we never expose an
unauthenticated PGx endpoint by default. For a purely local demo you may set
`ANUKRITI_ALLOW_NOAUTH=1` to bind to 127.0.0.1 without a key.

Run locally:
    export ANUKRITI_API_KEY=dev-local-key
    uvicorn anukriti_adk.serve:app --host 127.0.0.1 --port 8080

Deploy: see DEPLOYMENT.md (Cloud Run).
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .enforcement import evaluate_model_output, safe_refusal_text
from .mcp_connection import mcp_config
from .tools import check_pgx_safety, engine_mode, list_supported_workflows

API_KEY_ENV = "ANUKRITI_API_KEY"
ALLOW_NOAUTH = os.environ.get("ANUKRITI_ALLOW_NOAUTH", "").strip() == "1"


def _expected_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Dependency: enforce the API key unless explicitly running no-auth local."""
    expected = _expected_key()
    if expected is None:
        if ALLOW_NOAUTH:
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "Server misconfigured: ANUKRITI_API_KEY is not set. Refusing to "
                "serve an unauthenticated PGx endpoint. Set ANUKRITI_API_KEY, or "
                "ANUKRITI_ALLOW_NOAUTH=1 for a local-only demo."
            ),
        )
    presented = x_api_key
    if presented is None and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if presented != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


app = FastAPI(
    title="Anukriti ADK — PGx Safety Agent (testing surface)",
    description=(
        "Hosted, authenticated test surface for the Track-2 optimized agent. "
        "Same deterministic engine + same runtime guardrail as the ADK agent."
    ),
    version="0.1.0",
)


class CheckRequest(BaseModel):
    drug: str = Field(..., examples=["clopidogrel"])
    ancestry: str = Field(..., examples=["SAS", "South Asian", "AFR"])
    context: str = Field("prescription_review", examples=["pre_anesthesia"])
    community: str = Field("", examples=["Vysya"])


class GuardRequest(BaseModel):
    drug: str = Field(..., examples=["clopidogrel"])
    ancestry: str = Field(..., examples=["AFR"])
    draft_answer: str = Field(
        ...,
        description="A proposed agent answer to run through the guardrail.",
        examples=["Clopidogrel is safe for this patient — go ahead and prescribe."],
    )
    context: str = "prescription_review"
    community: str = ""


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "anukriti-adk",
        "what": (
            "Track-2 (Optimize) PGx safety agent. Deterministic engine decides; "
            "Gemini narrates; a runtime GenerativeBoundary rewrites unsafe output."
        ),
        "engine_mode": engine_mode(),
        "mcp": mcp_config(),
        "auth": "X-API-Key header required" if _expected_key() else (
            "NO-AUTH local mode" if ALLOW_NOAUTH else "MISCONFIGURED (no key set)"
        ),
        "endpoints": {
            "GET /healthz": "liveness",
            "POST /v1/check": "run a PGx safety check",
            "POST /v1/guard": "see the guardrail rewrite an unsafe draft",
            "GET /v1/workflows": "list supported drugs",
        },
        "disclaimer": "Demo only. Not for clinical use. Never enter real PHI.",
    }


@app.get("/v1/workflows", dependencies=[Depends(require_api_key)])
def workflows() -> dict:
    return list_supported_workflows()


@app.post("/v1/check", dependencies=[Depends(require_api_key)])
def check(req: CheckRequest) -> dict:
    return check_pgx_safety(
        drug=req.drug, ancestry=req.ancestry,
        context=req.context, community=req.community,
    )


@app.post("/v1/guard", dependencies=[Depends(require_api_key)])
def guard(req: GuardRequest) -> dict:
    """Run the engine, then run a draft answer through the runtime guardrail.

    This is the headline demo: submit an UNSAFE draft (a confident
    recommendation over an engine refusal, or a fabricated citation) and watch
    the GenerativeBoundary rewrite it into a named, safe refusal — exactly what
    the ADK after_model_callback does in production.
    """
    engine = check_pgx_safety(
        drug=req.drug, ancestry=req.ancestry,
        context=req.context, community=req.community,
    )
    decision = evaluate_model_output(req.draft_answer, [engine])
    return {
        "engine_result": engine,
        "draft_answer": req.draft_answer,
        "guardrail": decision,
        "final_answer": (
            decision["replacement"] if decision["action"] == "rewrite"
            else req.draft_answer
        ),
        "was_rewritten": decision["action"] == "rewrite",
    }
