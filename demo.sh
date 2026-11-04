#!/usr/bin/env bash
# Starts the chatbot and a public tunnel together, then prints the phone address.
# Stop everything with Ctrl+C.
cd "$(dirname "$0")" || exit 1

PORT=8000
LOG=/tmp/embu_tunnel.log
: > "$LOG"

if [ ! -x ./cloudflared ]; then
  echo "cloudflared not found in $(pwd). Download it first:"
  echo "  curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared"
  exit 1
fi

source run_api.sh

cleanup() {
  echo; echo "Stopping..."
  [ -n "$TUN_PID" ] && kill "$TUN_PID" 2>/dev/null
  [ -n "$BOT_PID" ] && kill "$BOT_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM EXIT

echo "Starting the bot..."
./venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port "$PORT" &
BOT_PID=$!

# wait up to 2 minutes for the bot to answer
for i in $(seq 1 120); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then break; fi
  if ! kill -0 "$BOT_PID" 2>/dev/null; then echo "The bot stopped. See the messages above."; exit 1; fi
  sleep 1
done

echo "Bot is up. Starting the tunnel..."
./cloudflared tunnel --url "http://localhost:$PORT" > "$LOG" 2>&1 &
TUN_PID=$!

URL=""
for i in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1)
  [ -n "$URL" ] && break
  sleep 1
done

echo
if [ -n "$URL" ]; then
  echo "=============================================="
  echo " Open this on your phone:"
  echo "   $URL"
  echo "=============================================="
  echo "(It may take up to a minute to become reachable.)"
else
  echo "No tunnel address yet. Last lines of the tunnel log:"
  tail -n 15 "$LOG"
fi
echo "Press Ctrl+C to stop the bot and the tunnel."
wait
