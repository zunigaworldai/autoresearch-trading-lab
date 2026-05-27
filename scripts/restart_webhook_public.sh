#!/usr/bin/env bash
set -e

if [ -z "$TRADINGVIEW_WEBHOOK_SECRET" ]; then
  echo "ERROR: TRADINGVIEW_WEBHOOK_SECRET is not set."
  echo 'Run first: export TRADINGVIEW_WEBHOOK_SECRET="YOUR_PRIVATE_SECRET"'
  exit 1
fi

mkdir -p logs

echo "Stopping old webhook receiver and risk watcher..."
pkill -f "live_monitor/webhook_receiver.py" || true
pkill -f "live_monitor/risk_watch.py" || true
sleep 1

echo "Starting webhook receiver in background..."
nohup python live_monitor/webhook_receiver.py --host 0.0.0.0 --port 8000 > logs/webhook_receiver.out 2>&1 &
echo $! > logs/webhook_receiver.pid

sleep 2

echo "Setting Codespaces port 8000 to Public..."
gh codespace ports visibility 8000:public -c "$CODESPACE_NAME" || gh codespace ports visibility 8000:public || true

echo "Starting risk watcher in background..."
nohup python live_monitor/risk_watch.py --watch --seconds 30 --max-orders-10min 8 --max-close-10min 4 --max-same-side-10min 6 > logs/risk_watch.out 2>&1 &
echo $! > logs/risk_watch.pid

echo "Webhook receiver PID:"
cat logs/webhook_receiver.pid

echo "Risk watcher PID:"
cat logs/risk_watch.pid

echo
echo "Health endpoint:"
echo "https://${CODESPACE_NAME}-8000.app.github.dev/health"

echo
echo "Webhook endpoint:"
echo "https://${CODESPACE_NAME}-8000.app.github.dev/webhook"

echo
echo "Local health check:"
curl -s http://127.0.0.1:8000/health || true

echo
echo "Startup complete."