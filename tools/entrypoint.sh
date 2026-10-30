#!/bin/sh
# Models live in a Docker volume (/models), downloaded once on first start and kept across app updates and rebuilds.
# The download runs in the background so the website opens immediately; the render worker waits for
# /models/.ready before processing videos. The lock stops web and worker downloading the same files twice.
M="${MEDREEL_MODELS:-/models}"
mkdir -p "$M"
export MEDREEL_WAIT_MODELS=1
( if flock "$M/.lock" bash /app/tools/setup_models.sh; then touch "$M/.ready"
  else echo "Model download incomplete (network?) - it resumes on the next start."; fi ) &
exec "$@"
