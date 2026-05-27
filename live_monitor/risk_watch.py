from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDERS_LOG = ROOT / "logs" / "orders.ndjson"
ALERTS_LOG = ROOT / "logs" / "risk_alerts.ndjson"
PAUSED_FILE = ROOT / "logs" / "PAUSED"
LIVE_STATUS = ROOT / "logs" / "live_status.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def append_alert(reason: str, details: dict) -> None:
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "received_at": utc_now().isoformat(),
        "reason": reason,
        "details": details,
    }
    with ALERTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def parse_time(value: str):
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
    

def pause_system(reason: str, status: dict) -> None:
    paused = {
        "paused_at": utc_now().isoformat(),
        "reason": reason,
        "status": status,
    }
    write_json(PAUSED_FILE, paused)
    append_alert(reason, status)


def evaluate_risk(
    max_orders_10min: int,
    max_close_10min: int,
    max_same_side_10min: int,
) -> dict:
    orders = read_ndjson(ORDERS_LOG)
    now = utc_now()
    cutoff = now - timedelta(minutes=10)

    recent = []
    for order in orders:
        ts = parse_time(order.get("received_at", ""))
        if ts and ts >= cutoff:
            recent.append(order)

    buy_count = sum(1 for o in recent if o.get("side") == "BUY")
    sell_count = sum(1 for o in recent if o.get("side") == "SELL")
    close_count = sum(1 for o in recent if o.get("side") == "CLOSE")

    status = {
        "checked_at": now.isoformat(),
        "paused": PAUSED_FILE.exists(),
        "orders_last_10min": len(recent),
        "buy_last_10min": buy_count,
        "sell_last_10min": sell_count,
        "close_last_10min": close_count,
        "thresholds": {
            "max_orders_10min": max_orders_10min,
            "max_close_10min": max_close_10min,
            "max_same_side_10min": max_same_side_10min,
        },
        "last_order": orders[-1] if orders else None,
    }

    write_json(LIVE_STATUS, status)

    if PAUSED_FILE.exists():
        return status

    if len(recent) > max_orders_10min:
        pause_system("max_orders_10min_exceeded", status)
        status["paused"] = True

    elif close_count > max_close_10min:
        pause_system("max_close_10min_exceeded", status)
        status["paused"] = True

    elif buy_count > max_same_side_10min:
        pause_system("max_buy_10min_exceeded", status)
        status["paused"] = True

    elif sell_count > max_same_side_10min:
        pause_system("max_sell_10min_exceeded", status)
        status["paused"] = True

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="ZW live risk watcher")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--max-orders-10min", type=int, default=8)
    parser.add_argument("--max-close-10min", type=int, default=4)
    parser.add_argument("--max-same-side-10min", type=int, default=6)

    args = parser.parse_args()

    while True:
        status = evaluate_risk(
            max_orders_10min=args.max_orders_10min,
            max_close_10min=args.max_close_10min,
            max_same_side_10min=args.max_same_side_10min,
        )

        print(json.dumps(status, indent=2))

        if not args.watch:
            break

        print("-" * 80)
        time.sleep(args.seconds)


if __name__ == "__main__":
    main()