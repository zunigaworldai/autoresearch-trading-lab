from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("outputs/backtests")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON in {path}: {exc}")


def find_result_files(results_root: Path, symbol: str) -> list[Path]:
    symbol_dir = results_root / symbol.upper()

    if not symbol_dir.exists():
        raise SystemExit(f"ERROR: results folder not found: {symbol_dir}")

    files = sorted(symbol_dir.glob(f"{symbol.upper()}_*_zw_vwap_vol_keltner_result.json"))

    if not files:
        raise SystemExit(f"ERROR: no result files found in {symbol_dir}")

    return files


def flatten_result_file(path: Path) -> list[dict]:
    payload = read_json(path)

    symbol = payload.get("symbol", "")
    timeframe = payload.get("timeframe", "")
    strategy = payload.get("strategy", "")
    csv_path = payload.get("csv_path", "")

    rows = []

    for cost_label, report in payload.get("cost_results", {}).items():
        tp_hits = int(report.get("tp_hits", 0))
        stop_hits = int(report.get("stop_hits", 0))
        total_trades = int(report.get("total_trades", report.get("trades", 0)))

        stop_tp_ratio = None
        if tp_hits > 0:
            stop_tp_ratio = stop_hits / tp_hits

        rows.append(
            {
                "result_file": str(path),
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy,
                "csv_path": csv_path,
                "cost_label": cost_label,
                "cost_per_side": float(report.get("cost_per_side", 0.0)),
                "global_score": float(report.get("global_score", 0.0)),
                "cagr": float(report.get("cagr", 0.0)),
                "max_drawdown": float(report.get("max_drawdown", 0.0)),
                "sharpe": float(report.get("sharpe", 0.0)),
                "sortino": float(report.get("sortino", 0.0)),
                "calmar": float(report.get("calmar", 0.0)),
                "profit_factor": float(report.get("profit_factor", 0.0)),
                "expectancy": float(report.get("expectancy", 0.0)),
                "win_rate": float(report.get("win_rate", 0.0)),
                "total_trades": total_trades,
                "tp_hits": tp_hits,
                "stop_hits": stop_hits,
                "stop_tp_ratio": stop_tp_ratio,
                "avg_net_return": float(report.get("avg_net_return", 0.0)),
                "best_trade": float(report.get("best_trade", 0.0)),
                "worst_trade": float(report.get("worst_trade", 0.0)),
            }
        )

    return rows


def load_all_results(results_root: Path, symbol: str) -> pd.DataFrame:
    rows = []

    for path in find_result_files(results_root, symbol):
        rows.extend(flatten_result_file(path))

    if not rows:
        raise SystemExit("ERROR: no cost results found")

    return pd.DataFrame(rows)


def classify_row(row: pd.Series) -> str:
    pf = float(row["profit_factor"])
    expectancy = float(row["expectancy"])
    score = float(row["global_score"])
    trades = int(row["total_trades"])

    if trades < 20:
        return "insufficient_trades"

    if pf >= 1.25 and expectancy > 0 and score > 0:
        return "candidate_keep"

    if pf >= 1.0 and expectancy >= 0:
        return "watchlist"

    if pf >= 0.75:
        return "improve_candidate"

    return "reject_current_config"


def build_summary(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["decision"] = df.apply(classify_row, axis=1)

    cost_1x = df[df["cost_label"] == "cost_1x"].copy()

    best_by_score = cost_1x.sort_values("global_score", ascending=False).head(1)
    best_by_pf = cost_1x.sort_values("profit_factor", ascending=False).head(1)
    worst_by_score = cost_1x.sort_values("global_score", ascending=True).head(1)

    summary = {
        "rows_analyzed": int(len(df)),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "timeframes": sorted(df["timeframe"].unique().tolist()),
        "cost_labels": sorted(df["cost_label"].unique().tolist()),
        "best_by_global_score_cost_1x": best_by_score.to_dict("records")[0] if len(best_by_score) else None,
        "best_by_profit_factor_cost_1x": best_by_pf.to_dict("records")[0] if len(best_by_pf) else None,
        "worst_by_global_score_cost_1x": worst_by_score.to_dict("records")[0] if len(worst_by_score) else None,
        "decision_counts": df["decision"].value_counts().to_dict(),
    }

    return summary


def build_recommendations(df: pd.DataFrame) -> list[str]:
    recommendations = []
    cost_1x = df[df["cost_label"] == "cost_1x"].copy()

    if cost_1x.empty:
        return ["No cost_1x rows found. Cannot generate recommendations."]

    best = cost_1x.sort_values("profit_factor", ascending=False).iloc[0]
    worst = cost_1x.sort_values("profit_factor", ascending=True).iloc[0]

    recommendations.append(
        f"Best preliminary timeframe by Profit Factor is {best['timeframe']} "
        f"with PF={best['profit_factor']:.4f}, expectancy={best['expectancy']:.6f}, "
        f"win_rate={best['win_rate']:.2%}, trades={int(best['total_trades'])}."
    )

    recommendations.append(
        f"Worst preliminary timeframe by Profit Factor is {worst['timeframe']} "
        f"with PF={worst['profit_factor']:.4f}, expectancy={worst['expectancy']:.6f}."
    )

    if float(best["profit_factor"]) < 1.0:
        recommendations.append(
            "No timeframe currently shows positive edge. Do not promote this configuration to paper/live."
        )

    if float(best["profit_factor"]) >= 0.75 and float(best["profit_factor"]) < 1.0:
        recommendations.append(
            "Best timeframe is close enough to diagnose. Create improvement tests around filters before rejecting strategy family."
        )

    if int(best["stop_hits"]) > int(best["tp_hits"]):
        recommendations.append(
            "Stops exceed TP hits in the best timeframe. Diagnose entry quality, stop placement, and time-of-day filters."
        )

    if float(best["expectancy"]) < 0:
        recommendations.append(
            "Expectancy is negative. Candidate must improve average trade result before further deployment."
        )

    return recommendations


def save_outputs(df: pd.DataFrame, summary: dict, recommendations: list[str], output_dir: Path, symbol: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / f"{symbol.upper()}_zw_research_matrix.csv"
    report_path = output_dir / f"{symbol.upper()}_zw_research_report.json"

    df.to_csv(matrix_path, index=False)

    report = {
        "symbol": symbol.upper(),
        "strategy": "zw_vwap_vol_keltner",
        "summary": summary,
        "recommendations": recommendations,
    }

    report_path.write_text(json.dumps(report, indent=2))

    return {
        "matrix_path": str(matrix_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze historical ZW backtest results")
    parser.add_argument("--symbol", required=True, help="Symbol, example: SPY")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default="outputs/research")

    args = parser.parse_args()

    df = load_all_results(
        results_root=Path(args.results_root),
        symbol=args.symbol,
    )

    summary = build_summary(df)
    recommendations = build_recommendations(df)

    output_files = save_outputs(
        df=df,
        summary=summary,
        recommendations=recommendations,
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
    )

    final_report = {
        "ok": True,
        "symbol": args.symbol.upper(),
        "rows_analyzed": int(len(df)),
        "summary": summary,
        "recommendations": recommendations,
        "output_files": output_files,
    }

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()