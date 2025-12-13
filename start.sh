#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
