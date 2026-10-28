#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -f run_api.sh ] && source run_api.sh      # hosted-model settings, if you have them
(sleep 4 && xdg-open http://127.0.0.1:8000) &
exec ./venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
