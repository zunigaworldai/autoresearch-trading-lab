from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

ORDERS_LOG = LOGS / "orders.ndjson"
ALERTS_LOG = LOGS / "risk_alerts.ndjson"
PAUSED_FILE = LOGS / "PAUSED"
LIVE_STATUS = LOGS / "live_status.json"
WEBHOOK_PID = LOGS / "webhook_receiver.pid"
RISK_WATCH_PID = LOGS / "risk_watch.pid"


def read_last_ndjson(path: Path) -> dict | None:
    if not path.exists():
        return None

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": lines[-1]}


def count_ndjson(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": "invalid_json"}
    

def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None

    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def pid_is_running(pid: int | None) -> bool:
    if pid is None:
        return False

    result = subprocess.run(
        ["bash", "-lc", f"ps -p {pid} > /dev/null 2>&1"],
        capture_output=True,
        text=True,
    )

    return result.returncode == 0


def health_check() -> dict:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": response.status == 200,
                "status_code": response.status,
                "body": json.loads(body),
            }
    except URLError as exc:
        return {
            "ok": False,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def build_status() -> dict:
    webhook_pid = read_pid(WEBHOOK_PID)
    risk_watch_pid = read_pid(RISK_WATCH_PID)

    return {
        "health": health_check(),
        "paused": PAUSED_FILE.exists(),
        "paused_details": read_json(PAUSED_FILE),
        "orders_count": count_ndjson(ORDERS_LOG),
        "alerts_count": count_ndjson(ALERTS_LOG),
        "last_order": read_last_ndjson(ORDERS_LOG),
        "last_alert": read_last_ndjson(ALERTS_LOG),
        "live_status": read_json(LIVE_STATUS),
        "processes": {
            "webhook_receiver": {
                "pid": webhook_pid,
                "running": pid_is_running(webhook_pid),
            },
            "risk_watch": {
                "pid": risk_watch_pid,
                "running": pid_is_running(risk_watch_pid),
            },
        },
    }


def main() -> None:
    status = build_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()