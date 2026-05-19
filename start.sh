#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source venv/bin/activate

# Load .env if present. Variables already set in the parent shell take
# precedence — this lets operators inject secrets at start time
# (`LLM_API_KEY=... ./start.sh`) without editing .env, while .env still
# supplies the default values for everything else.
if [ -f .env ]; then
  set -a
  while IFS= read -r _line; do
    # Skip blank lines and comment-only lines.
    [[ "$_line" =~ ^[[:space:]]*(#|$) ]] && continue
    # Skip lines that aren't VAR=value shape.
    [[ "$_line" != *=* ]] && continue
    _key="${_line%%=*}"
    _key="${_key// }"
    # Only assign if the var is currently unset in the parent env.
    if [ -z "${!_key+x}" ]; then
      eval "$_line"
    fi
  done < .env
  unset _line _key
  set +a
fi

# Job directories are created on demand by pipeline.storage.ensure_job_dir().

HOST="${APP_HOST:-127.0.0.1}"
PORT="${APP_PORT:-8090}"

# server.py is created in Phase 6; until then this will exit non-zero. That's fine.
python -m uvicorn server:app --host "$HOST" --port "$PORT" &
echo $! > .vts.pid
echo "Started on $HOST:$PORT (pid $(cat .vts.pid))"
