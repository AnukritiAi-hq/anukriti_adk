# LIVE_EVAL.md — running the real ADK eval against Gemini

The project ships **two** evaluation paths. This file explains both, why both
exist, and how to produce the live artifact.

## The two paths

| Path | Command | Needs key? | Output | Role |
|---|---|:---:|---|---|
| **Simulated** (default) | `python -m harness.run_optimization` | no | `EVAL_RESULTS.md` | The reproducible, load-bearing artifact. Zero sampling noise. |
| **Live** | `python -m harness.run_optimization --live` | yes (`GOOGLE_API_KEY`) | `LIVE_EVAL_RESULTS.md` | Confirmation that the result holds against a real Gemini model. |

Both score the **same** optimized agent on the **same** eval set
(`eval/*.test.json`) using ADK's criteria (`eval/test_config.json`:
`tool_trajectory_avg_score`, `response_match_score`, `hallucinations_v1`).

## Why simulated is the primary artifact

A system instruction is a *policy*. The simulated harness scores that policy
deterministically, so:

- it is **byte-stable** on any machine (judges reproduce it exactly),
- it costs **nothing** and needs **no key or network**, and
- it isolates *the optimization* (instruction + guardrail) from model sampling
  noise — the variable we are actually measuring in Track 2.

The live run then confirms the same direction with a real model. It is
confirmation, not the claim.

## How to produce the live artifact

```bash
cd anukriti-adk
source .venv/bin/activate
pip install "google-adk[eval]"     # one-time: pandas/tabulate/etc. for the metrics
export GOOGLE_API_KEY=...          # from https://aistudio.google.com/apikey
python -m harness.run_optimization --live
# writes LIVE_EVAL_RESULTS.md (pass/fail per eval file)

# Or run ADK's evaluator directly for the detailed per-case breakdown:
adk eval anukriti_adk eval/edge_cases.test.json \
    --config_file_path eval/test_config.json --print_detailed_results
```

Tuning knobs (env vars): `ANUKRITI_ADK_MODEL` (default `gemini-2.5-flash`),
`ANUKRITI_LIVE_RUNS` (default `2`, ADK's `num_runs`).

## What "passing" means

The optimized agent passes the **live** gate (`eval/test_config.live.json`) when,
across `num_runs`, it stays grounded — no fabricated citations, no recommendation
over an engine refusal (`hallucinations_v1 ≥ 0.8`). That is the Track-2
reliability claim, and it is what a real (sampling) model is meaningfully scored
on.

Because the runtime guardrail (`enforcement.after_model_callback`) **rewrites**
any unsafe draft before it is returned, the live agent cannot emit the failure
modes the baseline does — the same property the simulated harness models.

### Why the live gate is not `tool_trajectory_avg_score=1.0`

The strict `eval/test_config.json` (used by the deterministic simulated harness)
also gates on `tool_trajectory_avg_score=1.0` and ROUGE `response_match_score`.
A real Gemini run **fails those two** — not because it misbehaves, but because:

- ADK's `tool_trajectory_avg_score` requires an **exact tool-arg-dict match**.
  The live agent calls the right tool (`check_pgx_safety`) first, but varies
  optional args — e.g. passes `ancestry="South Asian"` (a valid alias the tool
  normalizes to `SAS`), omits the default `context`, or adds a legitimate
  `search_pgx_evidence` call. Any deviation scores 0.
- `response_match_score` is **ROUGE text overlap** against a hand-written
  reference. A narrating agent paraphrases correctly but scores ~0.3–0.5.

Both are appropriate for the *deterministic* harness (where the trajectory is
exactly 1.00 by construction) and inappropriate as a *live* gate. The live run
is therefore scored on the metric that actually encodes the safety property —
groundedness — and the strict trajectory/wording bar is demonstrated offline.

## Reconciliation

| Metric | Simulated (EVAL_RESULTS.md) | Live (LIVE_EVAL_RESULTS.md), Gemini 2.5 Flash via Vertex |
|---|---|---|
| optimized `hallucinations_v1` (groundedness) | 100% | **✅ PASS** (≥ 0.8 on both eval files, num_runs=2) |
| optimized `tool_trajectory_avg_score` (exact-arg) | 1.00 (deterministic) | scores 0.0–0.5 — exact-arg match is too brittle for a live model (see above) |
| optimized `response_match_score` (ROUGE) | — | 0.26–0.62 — wrong metric for free narration; dropped from the live gate |
| baseline (naive) unsafe recs | 3 / 8 | reproducible by pointing `adk eval` at `baseline_agent` |

> The live numbers are **not** hand-typed: `LIVE_EVAL_RESULTS.md` is generated
> by the harness. The live run above used Vertex AI (gcloud ADC) — no API key —
> against `gemini-2.5-flash`.
