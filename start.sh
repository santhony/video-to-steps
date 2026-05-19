#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source venv/bin/activate

# Load .env if present (don't fail if missing — env vars may come from elsewhere).
if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

# Job directories are created on demand by pipeline.storage.ensure_job_dir().

HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8090}"

# server.py is created in Phase 6; until then this will exit non-zero. That's fine.
python -m uvicorn server:app --host "$HOST" --port "$PORT" &
echo $! > .vts.pid
echo "Started on $HOST:$PORT (pid $(cat .vts.pid))"
