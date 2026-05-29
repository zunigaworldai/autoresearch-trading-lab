from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("outputs/backtests")
DEFAULT_OUTPUT_DIR = Path("outputs/research")

METRIC_COLUMNS = [
    "global_score",
    "profit_factor",
    "expectancy",
    "win_rate",
    "max_drawdown",
    "sharpe",
    "sortino",
    "calmar",
    "avg_net_return",
    "best_trade",
    "worst_trade",
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: result file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")


def find_result_files(results_root: Path, symbol: str) -> list[Path]:
    symbol = symbol.upper()
    symbol_dir = results_root / symbol

    if not symbol_dir.exists():
        raise SystemExit(f"ERROR: symbol results folder not found: {symbol_dir}")

    files = sorted(symbol_dir.glob(f"{symbol}_*_zw_vwap_vol_keltner*_result.json"))

    if not files:
        raise SystemExit(f"ERROR: no result files found in {symbol_dir}")

    return files


def flatten_result_file(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)

    symbol = str(payload.get("symbol", "")).upper()
    timeframe = str(payload.get("timeframe", ""))
    strategy = str(payload.get("strategy", ""))
    exit_policy = str(payload.get("exit_policy", "fixed"))
    csv_path = str(payload.get("csv_path", ""))

    rows = []

    for cost_label, report in payload.get("cost_results", {}).items():
        row = {
            "result_file": str(path),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "exit_policy": str(report.get("exit_policy", exit_policy)),
            "cost_label": cost_label,
            "cost_per_side": float(report.get("cost_per_side", 0.0)),
            "csv_path": csv_path,
            "total_trades": int(report.get("total_trades", report.get("trades", 0))),
            "tp_hits": int(report.get("tp_hits", 0)),
            "stop_hits": int(report.get("stop_hits", 0)),
            "be_exits": int(report.get("be_exits", 0)),
            "eod_exits": int(report.get("eod_exits", 0)),
            "be_triggered_count": int(report.get("be_triggered_count", 0)),
            "setup_counts": json.dumps(report.get("setup_counts", {}), sort_keys=True),
            "trades_path": str(report.get("trades_path", "")),
            "equity_path": str(report.get("equity_path", "")),
        }

        for metric in METRIC_COLUMNS:
            row[metric] = float(report.get(metric, 0.0))

        rows.append(row)

    return rows


def load_policy_matrix(results_root: Path, symbol: str) -> pd.DataFrame:
    rows = []

    for path in find_result_files(results_root, symbol):
        rows.extend(flatten_result_file(path))

    if not rows:
        raise SystemExit("ERROR: no rows found in result files")

    df = pd.DataFrame(rows)

    cost_order = {"cost_1x": 1, "cost_2x": 2, "cost_3x": 3}
    df["_cost_order"] = df["cost_label"].map(cost_order).fillna(99)
    df = df.sort_values(["timeframe", "_cost_order", "exit_policy"])
    df = df.drop(columns=["_cost_order"])

    return df


def add_fixed_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    fixed = df[df["exit_policy"] == "fixed"].copy()

    if fixed.empty:
        for metric in METRIC_COLUMNS:
            df[f"fixed_{metric}"] = None
            df[f"delta_vs_fixed_{metric}"] = None
        return df

    baseline_cols = ["symbol", "timeframe", "cost_label"] + METRIC_COLUMNS
    fixed = fixed[baseline_cols].copy()

    fixed = fixed.rename(
        columns={metric: f"fixed_{metric}" for metric in METRIC_COLUMNS}
    )

    merged = df.merge(
        fixed,
        how="left",
        on=["symbol", "timeframe", "cost_label"],
    )

    for metric in METRIC_COLUMNS:
        merged[f"delta_vs_fixed_{metric}"] = (
            merged[metric] - merged[f"fixed_{metric}"]
        )

    return merged


def best_rows(df: pd.DataFrame) -> dict[str, Any]:
    cost_1x = df[df["cost_label"] == "cost_1x"].copy()

    if cost_1x.empty:
        return {}

    return {
        "best_by_global_score_cost_1x": cost_1x.sort_values(
            "global_score",
            ascending=False,
        ).head(1).to_dict("records")[0],
        "best_by_profit_factor_cost_1x": cost_1x.sort_values(
            "profit_factor",
            ascending=False,
        ).head(1).to_dict("records")[0],
        "best_by_expectancy_cost_1x": cost_1x.sort_values(
            "expectancy",
            ascending=False,
        ).head(1).to_dict("records")[0],
        "lowest_drawdown_cost_1x": cost_1x.sort_values(
            "max_drawdown",
            ascending=True,
        ).head(1).to_dict("records")[0],
    }


def compare_policy_vs_fixed(df: pd.DataFrame, policy: str) -> list[dict[str, Any]]:
    rows = []

    subset = df[
        (df["exit_policy"] == policy)
        & (df["cost_label"] == "cost_1x")
    ].copy()

    for _, row in subset.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "exit_policy": row["exit_policy"],
                "cost_label": row["cost_label"],
                "profit_factor": float(row["profit_factor"]),
                "fixed_profit_factor": safe_float(row.get("fixed_profit_factor")),
                "delta_profit_factor": safe_float(row.get("delta_vs_fixed_profit_factor")),
                "expectancy": float(row["expectancy"]),
                "fixed_expectancy": safe_float(row.get("fixed_expectancy")),
                "delta_expectancy": safe_float(row.get("delta_vs_fixed_expectancy")),
                "global_score": float(row["global_score"]),
                "fixed_global_score": safe_float(row.get("fixed_global_score")),
                "delta_global_score": safe_float(row.get("delta_vs_fixed_global_score")),
                "max_drawdown": float(row["max_drawdown"]),
                "fixed_max_drawdown": safe_float(row.get("fixed_max_drawdown")),
                "delta_max_drawdown": safe_float(row.get("delta_vs_fixed_max_drawdown")),
                "total_trades": int(row["total_trades"]),
                "tp_hits": int(row["tp_hits"]),
                "stop_hits": int(row["stop_hits"]),
                "be_exits": int(row["be_exits"]),
                "eod_exits": int(row["eod_exits"]),
                "be_triggered_count": int(row["be_triggered_count"]),
            }
        )

    return sorted(rows, key=lambda item: item["global_score"], reverse=True)


def build_recommendations(df: pd.DataFrame) -> list[str]:
    recommendations = []

    cost_1x = df[df["cost_label"] == "cost_1x"].copy()

    if cost_1x.empty:
        return ["No cost_1x rows found. Cannot compare exit policies."]

    best = cost_1x.sort_values("global_score", ascending=False).iloc[0]

    recommendations.append(
        f"Best current row is {best['symbol']} {best['timeframe']} "
        f"{best['exit_policy']} with PF={best['profit_factor']:.4f}, "
        f"expectancy={best['expectancy']:.6f}, "
        f"global_score={best['global_score']:.4f}."
    )

    be_rows = cost_1x[cost_1x["exit_policy"] == "break_even_1r"].copy()

    if not be_rows.empty:
        improved_score = be_rows[be_rows["delta_vs_fixed_global_score"] > 0]
        improved_pf = be_rows[be_rows["delta_vs_fixed_profit_factor"] > 0]
        improved_expectancy = be_rows[be_rows["delta_vs_fixed_expectancy"] > 0]

        if len(improved_score):
            tfs = ", ".join(improved_score["timeframe"].astype(str).tolist())
            recommendations.append(
                f"break_even_1r improved global_score versus fixed in: {tfs}."
            )

        if len(improved_pf):
            tfs = ", ".join(improved_pf["timeframe"].astype(str).tolist())
            recommendations.append(
                f"break_even_1r improved profit_factor versus fixed in: {tfs}."
            )

        if len(improved_expectancy):
            tfs = ", ".join(improved_expectancy["timeframe"].astype(str).tolist())
            recommendations.append(
                f"break_even_1r improved expectancy versus fixed in: {tfs}."
            )

        if float(be_rows["profit_factor"].max()) < 1.0:
            recommendations.append(
                "break_even_1r is only a marginal improvement. "
                "It does not create positive edge yet."
            )

        if int(be_rows["be_triggered_count"].sum()) > 0:
            recommendations.append(
                "BE triggers are active, so the policy is working technically. "
                "Next test should be partial_50_at_1r_be."
            )

    if float(cost_1x["profit_factor"].max()) < 1.0:
        recommendations.append(
            "No exit policy currently passes PF > 1.0. Do not promote to paper/live."
        )

    recommendations.append(
        "Recommended next experiment: implement partial_50_at_1r_be and compare it "
        "against fixed and break_even_1r."
    )

    return recommendations


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}

    if isinstance(value, list):
        return [clean_for_json(v) for v in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def save_outputs(
    df: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path,
    symbol: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / f"{symbol.upper()}_exit_policy_comparison_matrix.csv"
    report_path = output_dir / f"{symbol.upper()}_exit_policy_comparison_report.json"

    df.to_csv(matrix_path, index=False)
    report_path.write_text(json.dumps(clean_for_json(report), indent=2))

    return {
        "matrix_path": str(matrix_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed and alternative exit policies")
    parser.add_argument("--symbol", required=True, help="Symbol, example: SPY")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args()

    symbol = args.symbol.upper()

    matrix = load_policy_matrix(
        results_root=Path(args.results_root),
        symbol=symbol,
    )

    matrix = add_fixed_deltas(matrix)

    report = {
        "ok": True,
        "symbol": symbol,
        "rows_analyzed": int(len(matrix)),
        "timeframes": sorted(matrix["timeframe"].unique().tolist()),
        "exit_policies": sorted(matrix["exit_policy"].unique().tolist()),
        "cost_labels": sorted(matrix["cost_label"].unique().tolist()),
        "best_rows": best_rows(matrix),
        "break_even_1r_vs_fixed_cost_1x": compare_policy_vs_fixed(
            matrix,
            "break_even_1r",
        ),
        "recommendations": build_recommendations(matrix),
    }

    output_files = save_outputs(
        df=matrix,
        report=report,
        output_dir=Path(args.output_dir),
        symbol=symbol,
    )

    report["output_files"] = output_files

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()