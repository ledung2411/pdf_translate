#!/usr/bin/env sh
# Khởi động web tool tại http://localhost:8000
cd "$(dirname "$0")"
exec python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
