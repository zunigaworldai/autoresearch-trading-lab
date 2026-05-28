from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "entry_time",
        "exit_time",
        "side",
        "setup",
        "entry",
        "exit",
        "stop",
        "tp",
        "exit_reason",
        "gross_return",
        "net_return",
    }

    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: missing columns: {sorted(missing)}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["entry_hour"] = df["entry_time"].dt.hour
    df["entry_minute"] = df["entry_time"].dt.minute
    df["entry_date"] = df["entry_time"].dt.date
    df["exit_date"] = df["exit_time"].dt.date
    df["holding_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
    df["overnight"] = df["entry_date"] != df["exit_date"]

    return df


def summarize_basic(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "message": "No trades found",
        }

    winners = df[df["net_return"] > 0]
    losers = df[df["net_return"] <= 0]

    return {
        "trades": int(len(df)),
        "wins": int(len(winners)),
        "losses": int(len(losers)),
        "win_rate": float(len(winners) / len(df)),
        "avg_net_return": float(df["net_return"].mean()),
        "median_net_return": float(df["net_return"].median()),
        "best_trade": float(df["net_return"].max()),
        "worst_trade": float(df["net_return"].min()),
        "tp_hits": int((df["exit_reason"] == "TP").sum()),
        "stop_hits": int((df["exit_reason"] == "STOP").sum()),
        "overnight_trades": int(df["overnight"].sum()),
        "overnight_pct": float(df["overnight"].mean()),
        "avg_holding_minutes": float(df["holding_minutes"].mean()),
        "max_holding_minutes": float(df["holding_minutes"].max()),
    }


def summarize_by_entry_hour(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    rows = []

    for hour, group in df.groupby("entry_hour"):
        trades = len(group)
        wins = int((group["net_return"] > 0).sum())
        losses = int((group["net_return"] <= 0).sum())
        tp_hits = int((group["exit_reason"] == "TP").sum())
        stop_hits = int((group["exit_reason"] == "STOP").sum())

        rows.append(
            {
                "entry_hour": int(hour),
                "trades": int(trades),
                "wins": wins,
                "losses": losses,
                "win_rate": float(wins / trades) if trades else 0.0,
                "avg_net_return": float(group["net_return"].mean()),
                "median_net_return": float(group["net_return"].median()),
                "best_trade": float(group["net_return"].max()),
                "worst_trade": float(group["net_return"].min()),
                "tp_hits": tp_hits,
                "stop_hits": stop_hits,
                "overnight_trades": int(group["overnight"].sum()),
                "overnight_pct": float(group["overnight"].mean()),
                "avg_holding_minutes": float(group["holding_minutes"].mean()),
            }
        )

    return sorted(rows, key=lambda row: row["entry_hour"])


def summarize_by_exit_reason(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    rows = []

    for reason, group in df.groupby("exit_reason"):
        rows.append(
            {
                "exit_reason": str(reason),
                "trades": int(len(group)),
                "avg_net_return": float(group["net_return"].mean()),
                "median_net_return": float(group["net_return"].median()),
                "best_trade": float(group["net_return"].max()),
                "worst_trade": float(group["net_return"].min()),
                "overnight_trades": int(group["overnight"].sum()),
                "avg_holding_minutes": float(group["holding_minutes"].mean()),
            }
        )

    return sorted(rows, key=lambda row: row["trades"], reverse=True)


def build_recommendations(basic: dict, by_hour: list[dict]) -> list[str]:
    recommendations = []

    if basic.get("trades", 0) == 0:
        return ["No trades available for diagnostics."]

    if basic.get("overnight_trades", 0) > 0:
        recommendations.append(
            "Critical execution issue: strategy is holding trades overnight. "
            "For scalping/intraday validation, add forced end-of-day exit before optimizing entries."
        )

    if basic.get("stop_hits", 0) > basic.get("tp_hits", 0):
        recommendations.append(
            "Stops exceed TP hits. Diagnose entry quality, stop placement, and confirmation filters."
        )

    if basic.get("avg_net_return", 0.0) < 0:
        recommendations.append(
            "Average net return is negative. Current configuration should not be promoted."
        )

    bad_hours = [
        row for row in by_hour
        if row["trades"] >= 3 and row["avg_net_return"] < 0 and row["win_rate"] < 0.40
    ]

    if bad_hours:
        hours = ", ".join(str(row["entry_hour"]) for row in bad_hours)
        recommendations.append(
            f"Potential bad entry hours detected: {hours}. "
            "Test a time-of-day filter that excludes these windows."
        )

    good_hours = [
        row for row in by_hour
        if row["trades"] >= 3 and row["avg_net_return"] > 0 and row["win_rate"] >= 0.50
    ]

    if good_hours:
        hours = ", ".join(str(row["entry_hour"]) for row in good_hours)
        recommendations.append(
            f"Potential stronger entry hours detected: {hours}. "
            "Test a session filter that focuses on these windows."
        )

    if basic.get("avg_holding_minutes", 0.0) > 120:
        recommendations.append(
            "Average holding time is too long for scalping. "
            "Test max-bars-in-trade or forced session close logic."
        )

    if not recommendations:
        recommendations.append("No obvious diagnostic issue detected. Continue deeper feature analysis.")

    return recommendations


def analyze_trades(trades_path: Path) -> dict:
    df = load_trades(trades_path)

    basic = summarize_basic(df)
    by_hour = summarize_by_entry_hour(df)
    by_exit_reason = summarize_by_exit_reason(df)
    recommendations = build_recommendations(basic, by_hour)

    return {
        "ok": True,
        "trades_path": str(trades_path),
        "basic": basic,
        "by_entry_hour": by_hour,
        "by_exit_reason": by_exit_reason,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trade-level diagnostics")
    parser.add_argument("--trades", required=True, help="Path to trades CSV")
    parser.add_argument("--output", default=None, help="Optional output JSON path")

    args = parser.parse_args()

    report = analyze_trades(Path(args.trades))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()