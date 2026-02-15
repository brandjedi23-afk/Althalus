#!/usr/bin/env bash
set -euo pipefail

# Si no viene PORT (Railway lo pone), usa 8000 por defecto
PORT="${PORT:-8000}"

# Arranca FastAPI
exec uvicorn server:app --host 0.0.0.0 --port "$PORT"