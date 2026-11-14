#!/usr/bin/env bash
# One-command setup and launch: creates the virtual environment, builds the UI if needed,
# and serves everything at http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
if [ ! -f frontend/dist/index.html ]; then
  (cd frontend && npm install && npm run build)
fi
echo "Open http://127.0.0.1:8000"
exec .venv/bin/uvicorn pinnsolver.api:app --host 127.0.0.1 --port 8000
