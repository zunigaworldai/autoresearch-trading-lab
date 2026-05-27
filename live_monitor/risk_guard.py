from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "trading_runtime.yaml"

DEFAULT_CONFIG = {
    "mode": "paper",
    "webhook_secret": "CHANGE_ME_LOCAL_ONLY",
    "orders_log": "logs/orders.ndjson",
    "alerts_log": "logs/risk_alerts.ndjson",
    "paused_file": "logs/PAUSED",
    "max_qty": 10,
    "allowed_sides": "BUY,SELL,CLOSE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_value(value: str):
    value = value.strip()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()

    if CONFIG_PATH.exists():
        for raw_line in CONFIG_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            key, value = line.split(":", 1)
            config[key.strip()] = parse_value(value)

    env_secret = os.getenv("TRADINGVIEW_WEBHOOK_SECRET")
    if env_secret:
        config["webhook_secret"] = env_secret

    return config


def root_path(relative_path: str) -> Path:
    return ROOT / relative_path


def append_ndjson(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def is_paused(config: dict) -> bool:
    return root_path(str(config["paused_file"])).exists()


def validate_secret(payload: dict, config: dict) -> tuple[bool, str]:
    expected = str(config.get("webhook_secret", "")).strip()

    if not expected or expected == "CHANGE_ME_LOCAL_ONLY":
        return True, "secret_check_disabled"

    received = str(payload.get("secret", "")).strip()

    if received != expected:
        return False, "invalid_webhook_secret"

    return True, "secret_ok"


def validate_payload(payload: dict, config: dict) -> tuple[bool, str, dict | None]:
    if not isinstance(payload, dict):
        return False, "payload_must_be_json_object", None

    if is_paused(config):
        return False, "system_paused", None

    secret_ok, secret_reason = validate_secret(payload, config)
    if not secret_ok:
        return False, secret_reason, None

    side = str(payload.get("side", "")).upper().strip()
    allowed_sides = {s.strip().upper() for s in str(config["allowed_sides"]).split(",")}

    if side not in allowed_sides:
        return False, f"invalid_side:{side}", None

    ticker = str(payload.get("ticker", "")).upper().strip()
    if not ticker:
        return False, "missing_ticker", None

    try:
        qty = float(payload.get("qty", 0))
    except (TypeError, ValueError):
        return False, "invalid_qty", None

    if qty <= 0:
        return False, "qty_must_be_positive", None

    if qty > float(config["max_qty"]):
        return False, "qty_exceeds_max_qty", None

        safe_payload = payload.copy()
    if "secret" in safe_payload:
        safe_payload["secret"] = "***REDACTED***"

    event = {
        "received_at": utc_now(),
        "mode": config["mode"],
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "raw": safe_payload,
    }

    if side == "CLOSE":
        return True, "accepted_close", event

    try:
        entry = float(payload["entry"])
        sl = float(payload["sl"])
        tp2 = float(payload["tp2"])
    except (KeyError, TypeError, ValueError):
        return False, "missing_or_invalid_entry_sl_tp2", None

    if side == "BUY" and not (sl < entry < tp2):
        return False, "invalid_buy_structure", None

    if side == "SELL" and not (tp2 < entry < sl):
        return False, "invalid_sell_structure", None

    risk = abs(entry - sl)
    reward = abs(tp2 - entry)

    event.update(
        {
            "entry": entry,
            "sl": sl,
            "tp2": tp2,
            "risk_per_unit": risk,
            "reward_per_unit": reward,
            "rr": reward / risk if risk > 0 else 0,
        }
    )

    return True, "accepted_order", event


def process_payload(payload: dict) -> tuple[int, dict]:
    config = load_config()
    ok, reason, event = validate_payload(payload, config)

    response = {
        "ok": ok,
        "reason": reason,
        "timestamp": utc_now(),
    }

    if ok and event:
        append_ndjson(root_path(str(config["orders_log"])), event)
        response["event"] = {
            "ticker": event["ticker"],
            "side": event["side"],
            "qty": event["qty"],
            "mode": event["mode"],
        }
        return 200, response

    append_ndjson(
        root_path(str(config["alerts_log"])),
        {
            "received_at": utc_now(),
            "reason": reason,
            "payload": payload,
        },
    )

    return 400, response