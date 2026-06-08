# Anukriti ADK — Devpost project writeup (Track 2: Optimize)

## Inspiration

A South Asian patient gets a stent and is prescribed clopidogrel — the standard
antiplatelet. But 14% of South Asians are CYP2C19 poor metabolizers (vs ~2% of
Europeans): the drug never activates, and they have a second cardiac event. The
genetics that would have caught this are well known. The problem isn't the
science — it's that AI healthcare agents *confidently recommend anyway* when the
evidence for a population is thin, because nothing stops them.

We had already built a PGx (pharmacogenomic) safety agent that worked great in a
sandbox. Then we watched it do the dangerous thing: asked about an
under-evidenced ancestry, a naive Gemini agent **invents a confident
recommendation**. That is the single most dangerous output a healthcare agent
can produce. Track 2 ("take an agent that works in a sandbox but struggles with
real-world edge cases and make it production-reliable") is exactly our problem.

## What it does

**Anukriti ADK** is a pre-prescription PGx safety agent built on the Google
Agent Development Kit + Gemini. You ask, in plain language, whether a drug is
safe for a patient's genetics and ancestry. The agent:

- **grounds every answer** in a deterministic, CPIC-pinned engine (never from
  the model's own knowledge),
- **narrates named rule IDs** (R1–R12 / V1–V10 / U1–U9) so every answer is
  traceable to a rule,
- **refuses by name** when the evidence for a population is thin ("R9 —
  population evidence uncertain"), instead of guessing, and
- **surfaces ancestry-specific wedge variants** that commercial gnomAD-based
  tests miss entirely (e.g. BCHE L307P, ~5.3% in the Telangana Vysya community,
  a succinylcholine/anesthesia risk — absent from gnomAD and 1000 Genomes).

## How we optimized it (the Track-2 work)

We treated reliability as an engineering discipline, not prompt vibes:

1. **A reproducible ADK eval set** of the hard cases (`eval/*.test.json`) — the
   under-evidenced ancestries, the unsupported drug, the fabricated-citation
   temptation, the multi-step anesthesia case.
2. **A baseline-vs-optimized A/B harness** scored on ADK's own metrics
   (`tool_trajectory_avg_score`, a `hallucinations_v1`-style groundedness proxy)
   that writes `EVAL_RESULTS.md` — never hand-typed.
3. **A programmatic system-instruction optimizer** (`harness/optimize_instructions.py`)
   that scores six instruction variants against the eval set and *selects* the
   winner. The leaderboard shows each added clause earning a measurable jump —
   refinement you can read, not guess.
4. **A runtime GenerativeBoundary** (`enforcement.py`) wired as ADK
   `after_tool` / `after_model` callbacks. It **rewrites** any unsafe model draft
   — a recommendation over an engine refusal, or a fabricated PMID — into a
   named, safe refusal *before the user sees it*. The guardrail does not depend
   on the model's cooperation.

**Result (ADK eval, same model, same tools, only the optimization differs):**

| Metric | Baseline | Optimized |
|---|---:|---:|
| `tool_trajectory_avg_score` | 0.44 | **1.00** |
| Hallucination-free rate | 50% | **100%** |
| Unsafe recommendations (of 8) | 3 | **0** |
| Correct named refusal | 0% | **100%** |

## How we built it

- **Google ADK 2.2** — `Agent`, `FunctionTool`s, `before/after_model` and
  `after_tool` callbacks, `adk web`, `adk eval`, `adk deploy cloud_run`.
- **Gemini 2.5 Flash** — narration only; it never makes the decision.
- **MCP seam** — `mcp_connection.py` lets the agent connect to a remote
  deterministic engine over MCP (`ANUKRITI_MCP_URL`); off-by-default so the demo
  runs fully offline.
- **FastAPI testing surface** (`serve.py`) on **Cloud Run** — an authenticated
  `/v1/guard` endpoint where judges watch the guardrail rewrite an unsafe draft
  live, with zero key/provisioning.
- **The deterministic engine** is a condensed, CPIC-pinned table layer derived
  from our broader Anukriti platform (a 13-gene phenotype engine on PyPI).

## Challenges

- Making the guardrail enforce *at runtime* rather than just instructing the
  model: the `after_model_callback` rewrite was the key — it converts "please
  don't" into "you can't."
- Keeping the whole thing **reproducible and offline** so judges get
  byte-identical results, while still offering a real `--live` ADK eval against
  Gemini as confirmation.

## Accomplishments we're proud of

- A reliability gain that is **impossible to fake**: every number is regenerated
  by a script from a public eval set.
- A guardrail that makes the most dangerous healthcare-agent failure mode
  **impossible by construction**.
- An equity angle with teeth — surfacing a real variant (BCHE/Vysya) that the
  entire gnomAD-based commercial industry misses.

## What's next

- Wire the live MCP connection to our hosted 13-gene engine.
- Expand the eval set to more (drug × ancestry) pairs.
- A Gemini Enterprise / Marketplace listing (Track 3 territory).

## Try it

- **Code:** this repo. `pip install -r requirements.txt && pytest` (offline, ~2s).
- **Reproduce the result:** `python -m harness.run_optimization`.
- **Testing access:** live on Cloud Run at
  `https://anukriti-adk-l6ztkkgjqa-uc.a.run.app` — call `POST /v1/guard` with the
  `X-API-Key` provided in the Devpost submission (see DEPLOYMENT.md). `GET /` and
  `GET /livez` are open (no auth).
- **Talk to it:** `adk web .` with a free Gemini key.

> Demo only. Not for clinical use. Never enter real PHI. The engine is positioned
> as non-device Clinical Decision Support under the 21st Century Cures Act
> §520(o)(1)(E).
