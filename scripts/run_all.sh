#!/usr/bin/env bash
# One-command reproduction: env -> data -> tests -> offline metrics -> serve
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/setup.sh
.venv/bin/python -m movielens.data.download
.venv/bin/python -m movielens.data.prepare
.venv/bin/python -m pytest
.venv/bin/python scripts/evaluate.py

echo ""
echo "[run_all] offline pipeline complete."
echo "[run_all] start the API with:"
echo "  .venv/bin/python -m uvicorn movielens.api.main:app --host 127.0.0.1 --port 8000"
echo "  then open http://127.0.0.1:8000/"

if [ "${1:-}" = "--serve" ]; then
  .venv/bin/python -m uvicorn movielens.api.main:app --host 127.0.0.1 --port 8000
fi
