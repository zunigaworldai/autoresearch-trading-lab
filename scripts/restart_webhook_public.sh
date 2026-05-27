#!/usr/bin/env bash
set -e

if [ -z "$TRADINGVIEW_WEBHOOK_SECRET" ]; then
  echo "ERROR: TRADINGVIEW_WEBHOOK_SECRET is not set."
  echo 'Run: export TRADINGVIEW_WEBHOOK_SECRET="YOUR_PRIVATE_SECRET"'
  exit 1
fi

mkdir -p logs

echo "Stopping old webhook receiver..."
pkill -f "live_monitor/webhook_receiver.py" || true
sleep 1

echo "Starting webhook receiver in background..."
nohup python live_monitor/webhook_receiver.py --host 0.0.0.0 --port 8000 > logs/webhook_receiver.out 2>&1 &

echo $! > logs/webhook_receiver.pid
sleep 2

echo "Setting port 8000 to Public..."
gh codespace ports visibility 8000:public -c "$CODESPACE_NAME" || gh codespace ports visibility 8000:public || true

echo "Receiver PID:"
cat logs/webhook_receiver.pid

echo "Health check:"
curl -s http://127.0.0.1:8000/health || true

echo
echo "Webhook URL:"
echo "https://${CODESPACE_NAME}-8000.app.github.dev/webhook"
