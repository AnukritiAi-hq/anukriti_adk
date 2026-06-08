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
adk eval anukriti_adk eval/edge_cases.test.json \
    --config_file_path eval/test_config.json --print_detailed_results

# or via the harness:
python -m harness.run_optimization --live
```

## 4. Deploy to Cloud Run (optional, for a hosted demo URL)

ADK agents deploy to Cloud Run directly:

```bash
adk deploy cloud_run \
    --project="$GCP_PROJECT" --region=us-central1 \
    --service_name=anukriti-adk \
    anukriti_adk
```

Set `GOOGLE_API_KEY` (or Vertex AI service-account auth) as a Cloud Run env var.
The hosted URL goes in the Devpost "Testing access" field.

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
