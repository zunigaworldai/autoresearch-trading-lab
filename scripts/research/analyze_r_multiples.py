from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


R_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0]


def threshold_label(value: float) -> str:
    return str(value).replace(".", "_")


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: OHLCV file not found: {path}")

    df = pd.read_csv(path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: OHLCV missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: trades file not found: {path}")

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
        raise SystemExit(f"ERROR: trades missing columns: {sorted(missing)}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")

    for col in ["entry", "exit", "stop", "tp", "gross_return", "net_return"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["entry_time", "exit_time", "entry", "stop", "tp"])

    return df


def trade_path(
    ohlcv: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
) -> pd.DataFrame:
    path = ohlcv[
        (ohlcv["timestamp"] > entry_time)
        & (ohlcv["timestamp"] <= exit_time)
    ].copy()

    if path.empty:
        path = ohlcv[
            (ohlcv["timestamp"] >= entry_time)
            & (ohlcv["timestamp"] <= exit_time)
        ].copy()

    return path


def analyze_trade_r(row: pd.Series, path_df: pd.DataFrame) -> dict[str, Any]:
    if path_df.empty:
        return {
            "initial_risk": None,
            "planned_reward": None,
            "planned_rr": None,
            "mfe": None,
            "mae": None,
            "mfe_r": None,
            "mae_r": None,
            "exit_r": None,
        }

    side = str(row["side"]).upper()
    entry = float(row["entry"])
    stop = float(row["stop"])
    tp = float(row["tp"])
    exit_price = float(row["exit"])

    if side == "LONG":
        initial_risk = entry - stop
        planned_reward = tp - entry
        mfe = float(path_df["high"].max()) - entry
        mae = entry - float(path_df["low"].min())
        exit_r = (exit_price - entry) / initial_risk if initial_risk > 0 else None

    elif side == "SHORT":
        initial_risk = stop - entry
        planned_reward = entry - tp
        mfe = entry - float(path_df["low"].min())
        mae = float(path_df["high"].max()) - entry
        exit_r = (entry - exit_price) / initial_risk if initial_risk > 0 else None

    else:
        return {
            "initial_risk": None,
            "planned_reward": None,
            "planned_rr": None,
            "mfe": None,
            "mae": None,
            "mfe_r": None,
            "mae_r": None,
            "exit_r": None,
        }

    if initial_risk <= 0:
        return {
            "initial_risk": initial_risk,
            "planned_reward": planned_reward,
            "planned_rr": None,
            "mfe": mfe,
            "mae": mae,
            "mfe_r": None,
            "mae_r": None,
            "exit_r": None,
        }

    return {
        "initial_risk": initial_risk,
        "planned_reward": planned_reward,
        "planned_rr": planned_reward / initial_risk,
        "mfe": mfe,
        "mae": mae,
        "mfe_r": mfe / initial_risk,
        "mae_r": mae / initial_risk,
        "exit_r": exit_r,
    }


def annotate_trades_with_r(ohlcv: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for idx, row in trades.iterrows():
        path_df = trade_path(
            ohlcv=ohlcv,
            entry_time=row["entry_time"],
            exit_time=row["exit_time"],
        )

        r_stats = analyze_trade_r(row, path_df)

        enriched = row.to_dict()
        enriched["trade_index"] = int(idx)
        enriched["bars_in_trade"] = int(len(path_df))
        enriched.update(r_stats)

        for threshold in R_THRESHOLDS:
            label = threshold_label(threshold)
            mfe_r = enriched.get("mfe_r")
            enriched[f"reached_{label}r"] = bool(mfe_r is not None and mfe_r >= threshold)

        enriched["reached_1r_and_lost"] = bool(
            enriched.get("mfe_r") is not None
            and enriched["mfe_r"] >= 1.0
            and float(enriched.get("net_return", 0.0)) <= 0.0
        )

        enriched["reached_1r_and_stopped"] = bool(
            enriched.get("mfe_r") is not None
            and enriched["mfe_r"] >= 1.0
            and str(enriched.get("exit_reason", "")).upper() == "STOP"
        )

        enriched["reached_1r_and_eod"] = bool(
            enriched.get("mfe_r") is not None
            and enriched["mfe_r"] >= 1.0
            and str(enriched.get("exit_reason", "")).upper() == "EOD"
        )

        rows.append(enriched)

    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"trades": 0}

    valid = df.dropna(subset=["mfe_r", "mae_r", "exit_r"]).copy()

    if valid.empty:
        return {"trades": int(len(df)), "valid_r_trades": 0}

    summary: dict[str, Any] = {
        "trades": int(len(df)),
        "valid_r_trades": int(len(valid)),
        "avg_mfe_r": float(valid["mfe_r"].mean()),
        "median_mfe_r": float(valid["mfe_r"].median()),
        "avg_mae_r": float(valid["mae_r"].mean()),
        "median_mae_r": float(valid["mae_r"].median()),
        "avg_exit_r": float(valid["exit_r"].mean()),
        "median_exit_r": float(valid["exit_r"].median()),
        "avg_net_return": float(valid["net_return"].mean()),
        "win_rate": float((valid["net_return"] > 0).mean()),
        "tp_hits": int((valid["exit_reason"] == "TP").sum()),
        "stop_hits": int((valid["exit_reason"] == "STOP").sum()),
        "eod_exits": int((valid["exit_reason"] == "EOD").sum()),
    }

    for threshold in R_THRESHOLDS:
        label = threshold_label(threshold)
        col = f"reached_{label}r"
        summary[f"reached_{label}r_count"] = int(valid[col].sum())
        summary[f"reached_{label}r_pct"] = float(valid[col].mean())

    reached_1r = valid[valid["reached_1_0r"]]
    reached_1r_and_lost = valid[valid["reached_1r_and_lost"]]
    reached_1r_and_stopped = valid[valid["reached_1r_and_stopped"]]
    reached_1r_and_eod = valid[valid["reached_1r_and_eod"]]

    summary["reached_1r_count"] = int(len(reached_1r))
    summary["reached_1r_pct"] = float(len(reached_1r) / len(valid)) if len(valid) else 0.0
    summary["reached_1r_and_lost_count"] = int(len(reached_1r_and_lost))
    summary["reached_1r_and_lost_pct"] = float(len(reached_1r_and_lost) / len(valid)) if len(valid) else 0.0
    summary["reached_1r_and_stopped_count"] = int(len(reached_1r_and_stopped))
    summary["reached_1r_and_eod_count"] = int(len(reached_1r_and_eod))

    if len(reached_1r):
        summary["reached_1r_lost_after_reaching_1r_pct"] = float(
            len(reached_1r_and_lost) / len(reached_1r)
        )
    else:
        summary["reached_1r_lost_after_reaching_1r_pct"] = 0.0

    return summary


def summarize_by_group(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return []

    rows = []

    for value, group in df.groupby(group_col):
        valid = group.dropna(subset=["mfe_r", "mae_r", "exit_r"]).copy()

        if valid.empty:
            continue

        reached_1r = valid[valid["reached_1_0r"]]
        reached_1r_and_lost = valid[valid["reached_1r_and_lost"]]

        rows.append(
            {
                str(group_col): str(value),
                "trades": int(len(valid)),
                "avg_mfe_r": float(valid["mfe_r"].mean()),
                "median_mfe_r": float(valid["mfe_r"].median()),
                "avg_mae_r": float(valid["mae_r"].mean()),
                "avg_exit_r": float(valid["exit_r"].mean()),
                "avg_net_return": float(valid["net_return"].mean()),
                "win_rate": float((valid["net_return"] > 0).mean()),
                "reached_1r_count": int(len(reached_1r)),
                "reached_1r_pct": float(len(reached_1r) / len(valid)),
                "reached_1r_and_lost_count": int(len(reached_1r_and_lost)),
                "reached_1r_and_lost_pct": float(len(reached_1r_and_lost) / len(valid)),
                "tp_hits": int((valid["exit_reason"] == "TP").sum()),
                "stop_hits": int((valid["exit_reason"] == "STOP").sum()),
                "eod_exits": int((valid["exit_reason"] == "EOD").sum()),
            }
        )

    return sorted(rows, key=lambda row: row["trades"], reverse=True)


def build_recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations = []

    trades = int(summary.get("valid_r_trades", 0))
    if trades == 0:
        return ["No valid R-multiple trades found. Cannot recommend exit improvements."]

    reached_1r_pct = float(summary.get("reached_1r_pct", 0.0))
    reached_1r_lost_pct = float(summary.get("reached_1r_and_lost_pct", 0.0))
    lost_after_1r_pct = float(summary.get("reached_1r_lost_after_reaching_1r_pct", 0.0))
    eod_exits = int(summary.get("eod_exits", 0))
    avg_exit_r = float(summary.get("avg_exit_r", 0.0))
    avg_mfe_r = float(summary.get("avg_mfe_r", 0.0))

    if reached_1r_pct >= 0.25:
        recommendations.append(
            "Break-even candidate: at least 25% of trades reached +1R. "
            "Test moving stop to entry after +1R."
        )

    if reached_1r_lost_pct >= 0.10 or lost_after_1r_pct >= 0.30:
        recommendations.append(
            "Strong break-even evidence: a meaningful share of trades reached +1R and later closed negative. "
            "Test break-even stop before changing entry logic."
        )

    if reached_1r_pct >= 0.30:
        recommendations.append(
            "Partial profit candidate: enough trades reached +1R to test closing 50% at +1R "
            "and moving the remainder to break-even."
        )

    if eod_exits > 0 and avg_mfe_r > max(avg_exit_r, 0.0):
        recommendations.append(
            "Trailing-stop candidate: trades show favorable excursion before final exit. "
            "Test ATR trailing or R-based trailing after +1R."
        )

    if float(summary.get("reached_2_0r_pct", 0.0)) < 0.20:
        recommendations.append(
            "Current +2R target may be too ambitious for this setup/timeframe. "
            "Compare fixed TP versus partial profit and trailing exits."
        )

    if not recommendations:
        recommendations.append(
            "No strong exit-management signal detected. Prioritize entry filters and regime filters first."
        )

    return recommendations


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_for_json(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def analyze_r_multiples(
    ohlcv_path: Path,
    trades_path: Path,
    symbol: str,
    timeframe: str,
    output_dir: Path,
) -> dict[str, Any]:
    ohlcv = load_ohlcv(ohlcv_path)
    trades = load_trades(trades_path)

    annotated = annotate_trades_with_r(ohlcv, trades)

    if "entry_time" in annotated.columns:
        annotated["entry_hour"] = pd.to_datetime(
            annotated["entry_time"],
            errors="coerce",
        ).dt.hour

    summary = summarize_overall(annotated)
    by_exit_reason = summarize_by_group(annotated, "exit_reason")
    by_entry_hour = summarize_by_group(annotated, "entry_hour")
    recommendations = build_recommendations(summary)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{symbol.upper()}_{timeframe}_r_multiples"
    annotated_path = output_dir / f"{base_name}_annotated_trades.csv"
    report_path = output_dir / f"{base_name}_report.json"

    annotated.to_csv(annotated_path, index=False)

    report = {
        "ok": True,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "ohlcv_path": str(ohlcv_path),
        "trades_path": str(trades_path),
        "summary": summary,
        "by_exit_reason": by_exit_reason,
        "by_entry_hour": by_entry_hour,
        "recommendations": recommendations,
        "output_files": {
            "annotated_trades": str(annotated_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(clean_for_json(report), indent=2))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MFE/MAE/R multiples for strategy trades")
    parser.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV used in the backtest")
    parser.add_argument("--trades", required=True, help="Path to trades CSV generated by backtest")
    parser.add_argument("--symbol", required=True, help="Symbol, example: SPY")
    parser.add_argument("--timeframe", required=True, help="Timeframe, example: 5m")
    parser.add_argument("--output-dir", default="outputs/research", help="Output folder")

    args = parser.parse_args()

    report = analyze_r_multiples(
        ohlcv_path=Path(args.ohlcv),
        trades_path=Path(args.trades),
        symbol=args.symbol,
        timeframe=args.timeframe,
        output_dir=Path(args.output_dir),
    )

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()