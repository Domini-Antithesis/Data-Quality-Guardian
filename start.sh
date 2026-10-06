#!/usr/bin/env bash
# One-command launcher for macOS / Linux:   ./start.sh
# Creates a virtual environment on first run, installs the app, starts the
# server and opens the browser. Press Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and run again." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment (first run only)..."
  "$PY" -m venv .venv
fi
VENV_PY=".venv/bin/python"
if ! "$VENV_PY" -c "import dq_agent, langgraph" >/dev/null 2>&1; then
  echo "Installing the app and its dependencies (first run only, 1-3 minutes)..."
  "$VENV_PY" -m pip install --upgrade pip --quiet
  "$VENV_PY" -m pip install -e ".[all-providers]" --quiet
fi

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example (optional - you can also use the Settings page)."
fi

echo
echo "Starting Data Quality Agent. Press Ctrl+C to stop."
echo "(If port 8100 is busy it picks the next free one and prints the address.)"
echo
if [ "${1:-}" = "--no-browser" ]; then
  exec "$VENV_PY" -m dq_agent.cli serve
else
  exec "$VENV_PY" -m dq_agent.cli serve --open
fi
