#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv venv --python 3.11 venv
# shellcheck source=/dev/null
source venv/bin/activate
uv pip install -r requirements-dev.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: ffmpeg not on PATH. Install before running pipeline." >&2
fi

echo "Setup complete. Activate venv with: source venv/bin/activate"
