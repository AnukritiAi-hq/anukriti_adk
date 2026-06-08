#!/usr/bin/env bash
# deploy_cloud_run.sh — deploy the Anukriti ADK testing surface to Cloud Run.
#
# Produces the hosted URL for the Devpost "Testing access" field. The service
# runs offline-deterministic (no Gemini key needed for /v1/check and /v1/guard),
# and is protected by an API key so it is never an open PGx endpoint.
#
# Prereqs: gcloud CLI authenticated, a GCP project with billing + Cloud Run +
# Cloud Build APIs enabled.
#
# Usage:
#   GCP_PROJECT=my-proj ANUKRITI_API_KEY=$(openssl rand -hex 16) ./deploy_cloud_run.sh
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-anukriti-adk}"
API_KEY="${ANUKRITI_API_KEY:-$(openssl rand -hex 16)}"

echo "==> Project:  $GCP_PROJECT"
echo "==> Region:   $REGION"
echo "==> Service:  $SERVICE"
echo "==> API key:  $API_KEY   (save this — send it to judges as X-API-Key)"

# Build + deploy from source (Cloud Build uses the Dockerfile in this dir).
gcloud run deploy "$SERVICE" \
  --project="$GCP_PROJECT" \
  --region="$REGION" \
  --source=. \
  --allow-unauthenticated \
  --set-env-vars="ANUKRITI_API_KEY=${API_KEY},ANUKRITI_ENGINE_MODE=offline" \
  --memory=512Mi \
  --cpu=1 \
  --max-instances=3 \
  --port=8080

URL="$(gcloud run services describe "$SERVICE" --project="$GCP_PROJECT" --region="$REGION" --format='value(status.url)')"

cat <<EOF

================================================================
Deployed. Testing-access URL (put this in Devpost):
  $URL

Smoke test (no PHI — demo only):
  curl "$URL/"                       # service card (no auth)
  curl "$URL/livez"                  # liveness (no auth; /healthz is reserved by Cloud Run)

  # The headline demo: an UNSAFE draft gets rewritten by the guardrail.
  curl -s -X POST "$URL/v1/guard" \\
    -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \\
    -d '{"drug":"clopidogrel","ancestry":"AFR","draft_answer":"Clopidogrel is safe — go ahead and prescribe."}' | python -m json.tool

Note: --allow-unauthenticated lets the URL be reachable; the API key enforces
access at the application layer. Rotate the key after judging.
================================================================
EOF
