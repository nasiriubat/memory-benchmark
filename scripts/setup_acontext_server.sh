#!/usr/bin/env bash
# Bring up the self-hosted AContext server stack (a separate compose project,
# ./acontext_server/), idempotently, so `make acontext` needs no manual steps.
#
# The benchmark's `acontext` container connects to this stack's API at
# host.docker.internal:8029. This script is a no-op if the stack is already up.
set -euo pipefail

cd "$(dirname "$0")/.."

AC_DIR="acontext_server"
AC_PROJECT="acontext-server"
AC_COMPOSE="${AC_DIR}/docker-compose.yml"
AC_ENV="${AC_DIR}/.env"

if [[ ! -f "${AC_ENV}" ]]; then
  echo ">> ERROR: ${AC_ENV} is missing (it holds the AContext secrets and is gitignored)." >&2
  echo "          Recreate it with OPENAI_API_KEY, ACONTEXT_ROOT_TOKEN, ROOT_SECRET_PEPPER." >&2
  echo "          See RUNBOOK 'AContext setup'." >&2
  exit 1
fi

acdc() { docker compose -p "${AC_PROJECT}" -f "${AC_COMPOSE}" --env-file "${AC_ENV}" "$@"; }

echo ">> Ensuring AContext server stack is up (project: ${AC_PROJECT})..."
acdc up -d

# AContext core fails to start until the S3 bucket exists; SeaweedFS does not
# auto-create it. Idempotent: creating an existing bucket is a harmless no-op.
echo ">> Ensuring S3 bucket 'acontext-assets' exists..."
acdc exec -T acontext-server-seaweedfs \
  sh -c 'echo "s3.bucket.create -name acontext-assets" | weed shell' >/dev/null 2>&1 || true
# Re-run up so core/api start now that the bucket is present.
acdc up -d

echo ">> Waiting for AContext API health (http://localhost:8029/health)..."
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8029/health >/dev/null 2>&1; then
    echo ">> AContext API is healthy."
    exit 0
  fi
  sleep 2
done
echo ">> WARNING: AContext API not healthy after ~60s; continuing (the adapter will" >&2
echo "            fall back to the local store and mark uses_local_fallback=true)." >&2
