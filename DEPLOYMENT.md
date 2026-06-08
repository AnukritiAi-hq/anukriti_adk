# DEPLOYMENT.md — running, deploying, and testing access for judges

## 1. Reproduce the result locally (no cloud, no API key)

```bash
cd anukriti-adk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # 32 tests, offline, ~2s
python -m harness.run_optimization       # regenerates EVAL_RESULTS.md
```

The engine is offline-deterministic, so the before/after table is byte-stable
on any machine. This is the primary "testing access" path for judges — nothing
to provision.

## 2. Talk to the agent (needs a free Gemini key)

```bash
cp .env.example .env
# put a key from https://aistudio.google.com/apikey into GOOGLE_API_KEY
adk web .                                # opens the ADK dev UI; pick anukriti_adk
```

Try: *"South Asian patient, post-PCI, on clopidogrel — anything I should know?"*
and *"Is aspirin safe for them?"* (the second must refuse to guess).

## 3. Run the real ADK eval against Gemini (confirmation of the simulated result)

```bash
pip install "google-adk[eval]"   # one-time: metric deps (pandas/tabulate/…)

adk eval anukriti_adk eval/edge_cases.test.json \
    --config_file_path eval/test_config.json --print_detailed_results

# or via the harness:
python -m harness.run_optimization --live
```

## 4. Deploy to Cloud Run (hosted testing surface for judges)

The `/v1/guard` testing surface (`anukriti_adk/serve.py`) deploys via the
bundled script, which builds the `Dockerfile` with Cloud Build and creates an
API-key-protected Cloud Run service:

```bash
GCP_PROJECT=<your-project> ANUKRITI_API_KEY=$(openssl rand -hex 16) ./deploy_cloud_run.sh
```

The script prints the hosted URL and the generated `X-API-Key`. The service
runs offline-deterministic (no Gemini key needed for `/v1/check` / `/v1/guard`).

> **Live deployment (this submission):**
> URL: `https://anukriti-adk-l6ztkkgjqa-uc.a.run.app`
> (project `project-885b39bc-9a80-45bf-bfd`, region `us-central1`, revision `…-00002`).
> The `X-API-Key` is **not** committed here — it is shared with judges out of
> band (Devpost "Testing access" field). Rotate it after judging by redeploying
> with a new `ANUKRITI_API_KEY`.

Smoke test (no PHI — demo only):

```bash
URL=https://anukriti-adk-l6ztkkgjqa-uc.a.run.app
curl "$URL/livez"                 # liveness (no auth). NOTE: /healthz is reserved
                                  # by the Cloud Run frontend; use /livez live.
curl "$URL/"                      # service card (no auth)

# Headline demo: an UNSAFE draft gets rewritten into a named R9 refusal.
curl -s -X POST "$URL/v1/guard" -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"drug":"clopidogrel","ancestry":"AFR","draft_answer":"Clopidogrel is safe — go ahead and prescribe."}' | python -m json.tool
```

### Alternative: deploy the ADK dev UI (talk-to-the-agent demo)

To host the conversational ADK agent itself (needs a Gemini key as a Cloud Run
env var) instead of the `/v1/guard` surface:

```bash
adk deploy cloud_run \
    --project="$GCP_PROJECT" --region=us-central1 \
    --service_name=anukriti-adk-web \
    anukriti_adk
```

---

## Credential safety — READ BEFORE EXPOSING ANY LIVE DEMO

This repo commits **no secrets**. `.env` is gitignored; only `.env.example`
(blank values) is tracked.

Carry-over action item from the prior hackathon build: the Elastic API key and
MongoDB Atlas password used by `anukriti-rapid-agent` were pasted into a chat
transcript. **Rotate both before pointing judges at any shared live service**:

- Elastic Cloud: revoke + reissue the deployment API key.
- MongoDB Atlas: rotate the database user password; update the deployment env.

This ADK submission does not depend on either (it runs offline-deterministic),
so it is safe to demo today; the rotation is only required if you reuse the
older hosted Elastic/Atlas endpoints in the video.

## Security note

This agent does not expose an authenticated network surface in the local/eval
path. If you deploy to Cloud Run for a public demo, restrict ingress or add IAM
auth — do not leave an unauthenticated PGx endpoint open. The agent is
demo-only and not for clinical use.
