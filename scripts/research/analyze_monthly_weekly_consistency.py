from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRADES_DIR = Path("outputs/research/regime_diagnostics")
DEFAULT_OUTPUT_DIR = Path("outputs/research/monthly_weekly_consistency")


DEFAULT_CANDIDATES = [
    {
        "label": "NVDA_A_C_0930_1030_prev_day_return_lte_0.015038",
        "trades_file": "NVDA_5m_A_C_0930_1030_break_even_1r_trades_with_regime_features.csv",
        "condition": "prev_day_return <= 0.015038",
    },
    {
        "label": "NVDA_A_C_0930_1030_prev_day_return_lt_0",
        "trades_file": "NVDA_5m_A_C_0930_1030_break_even_1r_trades_with_regime_features.csv",
        "condition": "prev_day_return < 0",
    },
    {
        "label": "NVDA_A_C_0930_1030_prev_vol20_gte_0.028897",
        "trades_file": "NVDA_5m_A_C_0930_1030_break_even_1r_trades_with_regime_features.csv",
        "condition": "prev_vol20 >= 0.028897",
    },
    {
        "label": "NVDA_A_only_1600_prev_range20_gte_0.042155",
        "trades_file": "NVDA_5m_A_only_0930_1600_break_even_1r_trades_with_regime_features.csv",
        "condition": "prev_range20 >= 0.042155",
    },
    {
        "label": "NVDA_A_C_1600_dow_eq_1",
        "trades_file": "NVDA_5m_A_C_0930_1600_break_even_1r_trades_with_regime_features.csv",
        "condition": "dow == 1",
    },
]


def profit_factor(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def max_drawdown(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(abs(dd.min()))


def compound_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def largest_win_share(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    gross_profit = returns[returns > 0].sum()
    if gross_profit <= 0:
        return 0.0
    return float(returns.max() / gross_profit)


def apply_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    work = df.copy()

    for col in work.columns:
        if col not in {"date", "entry_time", "exit_time", "entry_timestamp", "exit_timestamp", "timestamp", "datetime", "setup", "side", "exit_reason"}:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    try:
        filtered = work.query(condition, engine="python").copy()
    except Exception as exc:
        raise SystemExit(f"ERROR applying condition '{condition}': {exc}")

    return filtered


def period_stats(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["period", "trades", "pf", "expectancy", "win_rate", "max_drawdown", "return"])

    grouped = df.groupby(df["date"].dt.to_period(period))
    rows = []

    for key, group in grouped:
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        rows.append(
            {
                "period": str(key),
                "trades": int(len(returns)),
                "pf": profit_factor(returns),
                "expectancy": float(returns.mean()) if len(returns) else 0.0,
                "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
                "max_drawdown": max_drawdown(returns),
                "return": compound_return(returns),
            }
        )

    return pd.DataFrame(rows)


def yearly_pass_count(yearly: pd.DataFrame) -> int:
    if yearly.empty:
        return 0
    passed = yearly[(yearly["trades"] > 0) & (yearly["pf"] > 1.0) & (yearly["expectancy"] > 0) & (yearly["return"] > 0)]
    return int(len(passed))


def summarize_candidate(label: str, condition: str, df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    filtered = apply_condition(df, condition)
    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered["net_return"] = pd.to_numeric(filtered["net_return"], errors="coerce")
    filtered = filtered.dropna(subset=["date", "net_return"]).copy()

    returns = filtered["net_return"]

    yearly = period_stats(filtered, "Y")
    monthly = period_stats(filtered, "M")
    weekly = period_stats(filtered, "W")

    yearly_path = output_dir / f"{label}_yearly.csv"
    monthly_path = output_dir / f"{label}_monthly.csv"
    weekly_path = output_dir / f"{label}_weekly.csv"
    trades_path = output_dir / f"{label}_filtered_trades.csv"

    yearly.to_csv(yearly_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    weekly.to_csv(weekly_path, index=False)
    filtered.to_csv(trades_path, index=False)

    negative_years = yearly[yearly["return"] < 0] if not yearly.empty else yearly
    negative_months = monthly[monthly["return"] < 0] if not monthly.empty else monthly
    negative_weeks = weekly[weekly["return"] < 0] if not weekly.empty else weekly

    summary = {
        "label": label,
        "condition": condition,
        "trades": int(len(filtered)),
        "pf": profit_factor(returns),
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "max_drawdown": max_drawdown(returns),
        "total_return": compound_return(returns),
        "largest_win_share": largest_win_share(returns),
        "years_total": int(len(yearly)),
        "years_pass": yearly_pass_count(yearly),
        "years_negative": int(len(negative_years)),
        "worst_year_return": float(yearly["return"].min()) if not yearly.empty else 0.0,
        "months_total": int(len(monthly)),
        "months_negative": int(len(negative_months)),
        "negative_month_pct": float(len(negative_months) / len(monthly)) if len(monthly) else 0.0,
        "worst_month_return": float(monthly["return"].min()) if not monthly.empty else 0.0,
        "avg_month_return": float(monthly["return"].mean()) if not monthly.empty else 0.0,
        "weeks_total": int(len(weekly)),
        "weeks_negative": int(len(negative_weeks)),
        "negative_week_pct": float(len(negative_weeks) / len(weekly)) if len(weekly) else 0.0,
        "worst_week_return": float(weekly["return"].min()) if not weekly.empty else 0.0,
        "avg_week_return": float(weekly["return"].mean()) if not weekly.empty else 0.0,
        "output_files": {
            "filtered_trades": str(trades_path),
            "yearly": str(yearly_path),
            "monthly": str(monthly_path),
            "weekly": str(weekly_path),
        },
    }

    summary["decision"] = classify_candidate(summary)
    return summary


def classify_candidate(row: dict[str, Any]) -> str:
    if row["trades"] < 50:
        return "reject_low_sample"
    if row["years_negative"] > 0:
        return "reject_negative_year"
    if row["years_pass"] < row["years_total"]:
        return "reject_yearly_consistency"
    if row["negative_month_pct"] > 0.30:
        return "reject_monthly_unstable"
    if row["worst_month_return"] < -0.04:
        return "reject_worst_month"
    if row["negative_week_pct"] > 0.45:
        return "watch_weekly_unstable"
    if row["max_drawdown"] > 0.12:
        return "watch_high_drawdown"
    if row["pf"] <= 1.20:
        return "watch_low_pf"
    return "candidate_walk_forward"


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    trades_dir = Path(args.trades_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for candidate in DEFAULT_CANDIDATES:
        path = trades_dir / candidate["trades_file"]
        if not path.exists():
            summaries.append(
                {
                    "label": candidate["label"],
                    "condition": candidate["condition"],
                    "error": f"Missing trades file: {path}",
                    "decision": "missing_input",
                }
            )
            continue

        df = pd.read_csv(path)

        if "date" not in df.columns:
            summaries.append(
                {
                    "label": candidate["label"],
                    "condition": candidate["condition"],
                    "error": f"Missing date column in {path}",
                    "decision": "invalid_input",
                }
            )
            continue

        summary = summarize_candidate(
            label=candidate["label"],
            condition=candidate["condition"],
            df=df,
            output_dir=output_dir,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "monthly_weekly_candidate_summary.csv"
    report_path = output_dir / "monthly_weekly_candidate_report.json"

    summary_df.to_csv(summary_path, index=False)

    report = {
        "ok": True,
        "candidates_tested": len(summaries),
        "decision_counts": summary_df["decision"].value_counts().to_dict() if "decision" in summary_df.columns else {},
        "top_summary": summaries,
        "output_files": {
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "rules": {
            "reject_negative_year": "Any negative yearly compounded return blocks production.",
            "reject_monthly_unstable": "More than 30% negative months blocks production.",
            "reject_worst_month": "Worst month below -4% blocks production.",
            "candidate_walk_forward": "No negative years, acceptable monthly/weekly stability, PF/DD pass.",
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze monthly and weekly consistency for filtered candidates.")
    parser.add_argument("--trades-dir", default=str(DEFAULT_TRADES_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    report = run_analysis(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
