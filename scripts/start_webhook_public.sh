#!/usr/bin/env bash
set -e

if [ -z "$TRADINGVIEW_WEBHOOK_SECRET" ]; then
  echo "ERROR: TRADINGVIEW_WEBHOOK_SECRET is not set."
  echo "Run first:"
  echo 'export TRADINGVIEW_WEBHOOK_SECRET="YOUR_PRIVATE_SECRET"'
  exit 1
fi

echo "Stopping old webhook receiver if running..."
pkill -f "webhook_receiver.py" || true

echo "Starting webhook receiver on port 8000..."
python live_monitor/webhook_receiver.py --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

sleep 3

echo "Setting Codespaces port 8000 to Public..."
gh codespace ports visibility 8000:public -c "$CODESPACE_NAME" || true

echo "Webhook receiver running. PID: $SERVER_PID"
echo "Health endpoint:"
echo "https://${CODESPACE_NAME}-8000.app.github.dev/health"
echo "Webhook endpoint:"
echo "https://${CODESPACE_NAME}-8000.app.github.dev/webhook"

wait $SERVER_PID
