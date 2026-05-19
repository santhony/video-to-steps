#!/usr/bin/env bash
set -euo pipefail

if [ -f .vts.pid ]; then
  PID=$(cat .vts.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped pid $PID"
  else
    echo "Process $PID not running"
  fi
  rm -f .vts.pid
else
  echo ".vts.pid not found; nothing to stop"
fi
