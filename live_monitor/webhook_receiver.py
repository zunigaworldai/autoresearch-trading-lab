from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

import risk_guard


class WebhookHandler(BaseHTTPRequestHandler):
    server_version = "ZWTradingWebhook/0.1"

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "zw-trading-webhook",
                    "status": "running",
                    "mode": risk_guard.load_config().get("mode", "paper"),
                },
            )
            return

        self._send_json(404, {"ok": False, "reason": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/webhook":
            self._send_json(404, {"ok": False, "reason": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))

        if content_length <= 0:
            self._send_json(400, {"ok": False, "reason": "empty_body"})
            return

        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "reason": "invalid_json"})
            return

        status_code, response = risk_guard.process_payload(payload)
        self._send_json(status_code, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZW TradingView webhook receiver")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)

    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WebhookHandler)

    print(f"ZW Trading webhook receiver running on http://{args.host}:{args.port}")
    print("Health check: /health")
    print("Webhook endpoint: /webhook")
    print("Mode: paper/logging")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping webhook receiver...")
        server.server_close()


if __name__ == "__main__":
    main()