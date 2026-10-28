#!/bin/sh
# Models live in a Docker volume (/models), downloaded once on first start and kept across app updates and rebuilds.
# Already-present models are skipped in seconds; the lock stops web and worker downloading the same files twice.
mkdir -p "${MEDREEL_MODELS:-/models}"
if ! flock "${MEDREEL_MODELS:-/models}/.lock" bash /app/tools/setup_models.sh; then
  echo "Model download incomplete (network?) - the app starts anyway and the download resumes on the next start."
fi
exec "$@"
