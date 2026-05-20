#!/usr/bin/env bash
# launch.sh — opinionated wrapper around start.sh for the maintainer's
# specific dev environment.
#
# Wires up shell-side API key env vars (DEEPSEEK_API_KEY, TOGETHER_API_KEY)
# to the app-side names (LLM_API_KEY, VISION_API_KEY) before invoking
# ./start.sh. Reads from the current shell env first; falls back to
# grepping ~/.bashrc for `export VAR=value` lines so it works even when
# the parent shell hasn't sourced .bashrc (e.g., in a fresh non-interactive
# shell or under a process manager).
#
# This file contains zero literal secrets — only variable references.
# Safe to commit.
#
# Usage:
#   ./launch.sh           # binds 127.0.0.1 (default; loopback-only)
#   ./launch.sh --lan     # binds 0.0.0.0 for LAN access — no auth in v1, careful
#
# If your shell uses different env-var names for the same keys, edit the
# two mapping lines below. Don't put the actual key values here.
set -euo pipefail

# Read `export NAME=value` from ~/.bashrc if it isn't already in env.
# Handles plain values and ones wrapped in single or double quotes.
_from_bashrc() {
  local name="$1"
  grep -E "^export ${name}=" ~/.bashrc 2>/dev/null \
    | head -1 \
    | sed "s/^export ${name}=//; s/^[\"']//; s/[\"']\$//"
}

# Pull from shell env first, fall back to .bashrc.
: "${DEEPSEEK_API_KEY:=$(_from_bashrc DEEPSEEK_API_KEY)}"
: "${TOGETHER_API_KEY:=$(_from_bashrc TOGETHER_API_KEY)}"

[ -n "${DEEPSEEK_API_KEY:-}" ] || {
  echo "launch.sh: DEEPSEEK_API_KEY not set in env or ~/.bashrc" >&2
  echo "  (the script reads from your shell env first, then greps your .bashrc)" >&2
  exit 1
}
[ -n "${TOGETHER_API_KEY:-}" ] || {
  echo "launch.sh: TOGETHER_API_KEY not set in env or ~/.bashrc" >&2
  exit 1
}

# Map shell-side names → app-side names. Edit these if your provider
# choices differ (e.g., point VISION_API_KEY at $OPENAI_API_KEY instead).
export LLM_API_KEY="$DEEPSEEK_API_KEY"
export VISION_API_KEY="$TOGETHER_API_KEY"

# Default to loopback. Pass --lan to bind 0.0.0.0.
_app_host_default="127.0.0.1"
for arg in "$@"; do
  case "$arg" in
    --lan)  _app_host_default="0.0.0.0" ;;
    --help|-h)
      sed -n '/^# Usage:/,/^set/p' "$0" | sed 's/^# \?//; /^set/d'
      exit 0
      ;;
  esac
done
export APP_HOST="${APP_HOST:-$_app_host_default}"

exec ./start.sh
