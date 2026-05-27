from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDERS_LOG = ROOT / "logs" / "orders.ndjson"
ALERTS_LOG = ROOT / "logs" / "risk_alerts.ndjson"


def read_ndjson(path: Path) -> list[dict]:
    if not path.exists():
        return []

    events = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return events


def build_summary() -> dict:
    orders = read_ndjson(ORDERS_LOG)
    alerts = read_ndjson(ALERTS_LOG)

    sides = Counter(o.get("side", "UNKNOWN") for o in orders)
    tickers = Counter(o.get("ticker", "UNKNOWN") for o in orders)

    rr_values = [float(o.get("rr", 0)) for o in orders if "rr" in o]
    avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0.0

    return {
        "orders_count": len(orders),
        "alerts_count": len(alerts),
        "sides": dict(sides),
        "tickers": dict(tickers),
        "avg_rr": avg_rr,
        "last_order": orders[-1] if orders else None,
        "last_alert": alerts[-1] if alerts else None,
    }


def print_summary() -> None:
    print(json.dumps(build_summary(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live webhook logs")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--seconds", type=int, default=30)

    args = parser.parse_args()

    if not args.watch:
        print_summary()
        return

    while True:
        print_summary()
        print("-" * 80)
        time.sleep(args.seconds)


if __name__ == "__main__":
    main()