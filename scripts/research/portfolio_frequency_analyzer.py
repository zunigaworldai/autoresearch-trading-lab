from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/research/portfolio_frequency")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    return value


def profit_factor(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def compound_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(abs(drawdown.min()))


def largest_win_share(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    gross_profit = float(returns[returns > 0].sum())
    if gross_profit <= 0:
        return 0.0
    return float(returns.max() / gross_profit)


def infer_symbol(path: Path, df: pd.DataFrame) -> str:
    if "symbol" in df.columns and df["symbol"].notna().any():
        return str(df["symbol"].dropna().iloc[0]).upper()
    match = re.match(r"([A-Z]+)_", path.name)
    return match.group(1).upper() if match else "UNKNOWN"


def infer_entry_time(df: pd.DataFrame) -> pd.Series:
    for col in ["entry_time", "entry_timestamp", "timestamp", "datetime", "date"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    raise SystemExit("ERROR: no timestamp/date column found in trades file")


def numericize_for_query(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    string_cols = {
        "symbol", "setup", "side", "exit_reason", "partial_exit_reason",
        "entry_time", "exit_time", "entry_timestamp", "exit_timestamp",
        "timestamp", "datetime", "date", "source_file",
    }
    for col in out.columns:
        if col in string_cols:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_trades(paths: list[Path], condition: str | None) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["symbol"] = infer_symbol(path, df)
        df["source_file"] = str(path)
        df["entry_time_normalized"] = infer_entry_time(df)
        df["date"] = df["entry_time_normalized"].dt.normalize()

        if "net_return" not in df.columns:
            raise SystemExit(f"ERROR: net_return column missing in {path}")

        df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
        df = df.dropna(subset=["date", "net_return"]).copy()

        if condition:
            qdf = numericize_for_query(df)
            try:
                keep_index = qdf.query(condition, engine="python").index
            except Exception as exc:
                raise SystemExit(f"ERROR applying condition '{condition}' to {path}: {exc}")
            df = df.loc[keep_index].copy()

        frames.append(df)

    if not frames:
        raise SystemExit("ERROR: no trades loaded")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["entry_time_normalized", "symbol"]).reset_index(drop=True)
    return out


def period_frame(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame(rows)

    for key, group in trades.groupby(trades["date"].dt.to_period(period)):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        symbols = sorted(group["symbol"].dropna().unique().tolist())
        rows.append(
            {
                "period": str(key),
                "trades": int(len(group)),
                "symbols_active": int(len(symbols)),
                "symbols": ",".join(symbols),
                "pf": profit_factor(returns),
                "expectancy": float(returns.mean()) if len(returns) else 0.0,
                "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
                "return": compound_return(returns),
                "max_drawdown": max_drawdown(returns),
            }
        )
    return pd.DataFrame(rows)


def symbol_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, group in trades.groupby("symbol"):
        returns = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        rows.append(
            {
                "symbol": symbol,
                "trades": int(len(group)),
                "active_days": int(group["date"].nunique()),
                "active_weeks": int(group["date"].dt.to_period("W").nunique()),
                "active_months": int(group["date"].dt.to_period("M").nunique()),
                "pf": profit_factor(returns),
                "expectancy": float(returns.mean()) if len(returns) else 0.0,
                "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
                "total_return": compound_return(returns),
                "max_drawdown": max_drawdown(returns),
                "largest_win_share": largest_win_share(returns),
            }
        )
    return pd.DataFrame(rows).sort_values(["trades", "pf"], ascending=[False, False]).reset_index(drop=True)


def complete_period_count(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return int(len(pd.period_range(start=start, end=end, freq=freq)))


def business_day_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    return int(len(pd.bdate_range(start=start.normalize(), end=end.normalize())))


def classify_portfolio(summary: dict[str, Any]) -> str:
    if summary["pf"] <= 1.0 or summary["expectancy"] <= 0:
        return "reject_no_edge"
    if summary["years_negative"] > 0:
        return "reject_negative_year"
    if summary["negative_month_pct_calendar"] > 0.35:
        return "reject_monthly_unstable"
    if summary["worst_month_return"] < -0.05:
        return "reject_worst_month"
    if summary["max_drawdown_trade_sequence"] > 0.15:
        return "watch_high_drawdown"
    if summary["pf"] <= 1.20:
        return "watch_low_pf"
    if summary["total_trades"] < 100:
        return "watch_low_total_sample"
    if summary["trades_per_week_calendar"] < 5:
        return "watch_low_portfolio_frequency"
    return "candidate_portfolio_walk_forward"


def analyze_portfolio(trades: pd.DataFrame, output_dir: Path, label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        raise SystemExit("ERROR: portfolio trades are empty after filtering")

    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    start = trades["date"].min()
    end = trades["date"].max()

    total_business_days = business_day_count(start, end)
    total_weeks = complete_period_count(start, end, "W")
    total_months = complete_period_count(start, end, "M")
    total_years = complete_period_count(start, end, "Y")

    active_days = int(trades["date"].nunique())
    active_weeks = int(trades["date"].dt.to_period("W").nunique())
    active_months = int(trades["date"].dt.to_period("M").nunique())

    daily = period_frame(trades, "D")
    weekly = period_frame(trades, "W")
    monthly = period_frame(trades, "M")
    yearly = period_frame(trades, "Y")
    symbols = symbol_summary(trades)

    negative_years = yearly[yearly["return"] < 0] if not yearly.empty else yearly
    negative_months = monthly[monthly["return"] < 0] if not monthly.empty else monthly
    negative_weeks = weekly[weekly["return"] < 0] if not weekly.empty else weekly

    day_trade_counts = trades.groupby("date").size()
    week_trade_counts = trades.groupby(trades["date"].dt.to_period("W")).size()

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:120]

    combined_path = output_dir / f"{safe}_combined_trades.csv"
    daily_path = output_dir / f"{safe}_daily.csv"
    weekly_path = output_dir / f"{safe}_weekly.csv"
    monthly_path = output_dir / f"{safe}_monthly.csv"
    yearly_path = output_dir / f"{safe}_yearly.csv"
    symbol_path = output_dir / f"{safe}_symbol_summary.csv"
    report_path = output_dir / f"{safe}_portfolio_report.json"

    trades.to_csv(combined_path, index=False)
    daily.to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    symbols.to_csv(symbol_path, index=False)

    summary = {
        "label": label,
        "symbols": sorted(trades["symbol"].dropna().unique().tolist()),
        "symbols_count": int(trades["symbol"].nunique()),
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "total_trades": int(len(trades)),
        "pf": profit_factor(returns),
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "total_return_trade_sequence": compound_return(returns),
        "max_drawdown_trade_sequence": max_drawdown(returns),
        "largest_win_share": largest_win_share(returns),
        "business_days_total": total_business_days,
        "active_days": active_days,
        "active_day_pct": float(active_days / total_business_days) if total_business_days else 0.0,
        "trades_per_business_day": float(len(trades) / total_business_days) if total_business_days else 0.0,
        "trades_per_active_day": float(len(trades) / active_days) if active_days else 0.0,
        "calendar_weeks_total": total_weeks,
        "active_weeks": active_weeks,
        "active_week_pct": float(active_weeks / total_weeks) if total_weeks else 0.0,
        "trades_per_week_calendar": float(len(trades) / total_weeks) if total_weeks else 0.0,
        "trades_per_active_week": float(len(trades) / active_weeks) if active_weeks else 0.0,
        "calendar_months_total": total_months,
        "active_months": active_months,
        "inactive_months": int(max(total_months - active_months, 0)),
        "active_month_pct": float(active_months / total_months) if total_months else 0.0,
        "trades_per_calendar_month": float(len(trades) / total_months) if total_months else 0.0,
        "trades_per_active_month": float(len(trades) / active_months) if active_months else 0.0,
        "calendar_years_total": total_years,
        "years_negative": int(len(negative_years)),
        "worst_year_return": float(yearly["return"].min()) if not yearly.empty else 0.0,
        "months_negative_active": int(len(negative_months)),
        "negative_month_pct_active": float(len(negative_months) / len(monthly)) if len(monthly) else 0.0,
        "negative_month_pct_calendar": float(len(negative_months) / total_months) if total_months else 0.0,
        "worst_month_return": float(monthly["return"].min()) if not monthly.empty else 0.0,
        "avg_month_return_active": float(monthly["return"].mean()) if not monthly.empty else 0.0,
        "weeks_negative_active": int(len(negative_weeks)),
        "negative_week_pct_active": float(len(negative_weeks) / len(weekly)) if len(weekly) else 0.0,
        "negative_week_pct_calendar": float(len(negative_weeks) / total_weeks) if total_weeks else 0.0,
        "worst_week_return": float(weekly["return"].min()) if not weekly.empty else 0.0,
        "avg_week_return_active": float(weekly["return"].mean()) if not weekly.empty else 0.0,
        "active_day_distribution": {
            "days_with_1_trade": int((day_trade_counts == 1).sum()),
            "days_with_2_to_5_trades": int(((day_trade_counts >= 2) & (day_trade_counts <= 5)).sum()),
            "days_with_6_plus_trades": int((day_trade_counts >= 6).sum()),
            "max_trades_single_day": int(day_trade_counts.max()) if len(day_trade_counts) else 0,
        },
        "week_trade_distribution": {
            "active_weeks_with_1_trade": int((week_trade_counts == 1).sum()),
            "active_weeks_with_2_to_4_trades": int(((week_trade_counts >= 2) & (week_trade_counts <= 4)).sum()),
            "active_weeks_with_5_plus_trades": int((week_trade_counts >= 5).sum()),
            "max_trades_single_week": int(week_trade_counts.max()) if len(week_trade_counts) else 0,
            "calendar_weeks_zero_trades": int(max(total_weeks - active_weeks, 0)),
        },
        "output_files": {
            "combined_trades": str(combined_path),
            "daily": str(daily_path),
            "weekly": str(weekly_path),
            "monthly": str(monthly_path),
            "yearly": str(yearly_path),
            "symbol_summary": str(symbol_path),
            "report": str(report_path),
        },
        "notes": [
            "Portfolio return is a trade-sequence return from net_return, not a final capital allocation model.",
            "Use this for frequency and coverage diagnostics. Later portfolio sizing must include exposure, correlation, and risk budget.",
        ],
    }

    summary["decision"] = classify_portfolio(summary)
    report_path.write_text(json.dumps(clean(summary), indent=2), encoding="utf-8")
    return summary


def expand_globs(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(x) for x in sorted(glob.glob(pattern)))

    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze portfolio-level trade frequency and stability.")
    parser.add_argument("--trades-glob", nargs="+", required=True)
    parser.add_argument("--condition", default="")
    parser.add_argument("--label", default="portfolio_frequency")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    paths = expand_globs(args.trades_glob)
    if not paths:
        raise SystemExit("ERROR: no files matched --trades-glob")

    trades = load_trades(paths, args.condition.strip() or None)
    report = analyze_portfolio(trades, Path(args.output_dir), args.label)
    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
