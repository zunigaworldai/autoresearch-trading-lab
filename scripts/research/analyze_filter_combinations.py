from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRADES_FILE = Path("outputs/research/regime_diagnostics/NVDA_5m_A_C_0930_1030_break_even_1r_trades_with_regime_features.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/filter_combinations")


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


def pf(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    gains = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def comp_ret(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def dd(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    return float(abs((eq / peak - 1.0).min()))


def lws(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    gross = r[r > 0].sum()
    return float(r.max() / gross) if gross > 0 else 0.0


def period_stats(df: pd.DataFrame, p: str) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for key, g in df.groupby(df["date"].dt.to_period(p)):
        r = pd.to_numeric(g["net_return"], errors="coerce").dropna()
        rows.append({
            "period": str(key),
            "trades": int(len(r)),
            "pf": pf(r),
            "expectancy": float(r.mean()) if len(r) else 0.0,
            "return": comp_ret(r),
            "max_drawdown": dd(r),
        })
    return pd.DataFrame(rows)


def classify(row: dict[str, Any]) -> str:
    if row["trades"] < 35:
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


def score(row: dict[str, Any]) -> float:
    s = 0.0
    s += min(max(row["pf"], 0.0), 2.0) / 2.0 * 0.18
    s += min(max(row["expectancy"], 0.0), 0.005) / 0.005 * 0.18
    s += min(max(row["total_return"], 0.0), 0.30) / 0.30 * 0.12
    s += max(0.0, 1.0 - min(row["max_drawdown"], 0.20) / 0.20) * 0.12
    s += max(0.0, 1.0 - min(row["negative_month_pct"], 1.0)) * 0.16
    s += max(0.0, 1.0 - min(row["negative_week_pct"], 1.0)) * 0.08
    s += max(0.0, 1.0 - min(abs(row["worst_month_return"]), 0.10) / 0.10) * 0.08
    s += max(0.0, 1.0 - min(row["largest_win_share"], 1.0)) * 0.08
    if row["years_negative"] > 0:
        s -= 0.35
    if row["negative_month_pct"] > 0.30:
        s -= 0.20
    if row["worst_month_return"] < -0.04:
        s -= 0.15
    if row["trades"] < 50:
        s -= 0.10
    return float(s)


def evaluate(label: str, df: pd.DataFrame) -> dict[str, Any]:
    r = pd.to_numeric(df["net_return"], errors="coerce").dropna()
    yearly = period_stats(df, "Y")
    monthly = period_stats(df, "M")
    weekly = period_stats(df, "W")
    yn = yearly[yearly["return"] < 0] if not yearly.empty else yearly
    mn = monthly[monthly["return"] < 0] if not monthly.empty else monthly
    wn = weekly[weekly["return"] < 0] if not weekly.empty else weekly
    years_pass = 0
    if not yearly.empty:
        years_pass = int(len(yearly[(yearly["trades"] > 0) & (yearly["pf"] > 1.0) & (yearly["expectancy"] > 0) & (yearly["return"] > 0)]))
    row = {
        "label": label,
        "trades": int(len(df)),
        "pf": pf(r),
        "expectancy": float(r.mean()) if len(r) else 0.0,
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "max_drawdown": dd(r),
        "total_return": comp_ret(r),
        "largest_win_share": lws(r),
        "years_total": int(len(yearly)),
        "years_pass": years_pass,
        "years_negative": int(len(yn)),
        "worst_year_return": float(yearly["return"].min()) if not yearly.empty else 0.0,
        "months_total": int(len(monthly)),
        "months_negative": int(len(mn)),
        "negative_month_pct": float(len(mn) / len(monthly)) if len(monthly) else 0.0,
        "worst_month_return": float(monthly["return"].min()) if not monthly.empty else 0.0,
        "avg_month_return": float(monthly["return"].mean()) if not monthly.empty else 0.0,
        "weeks_total": int(len(weekly)),
        "weeks_negative": int(len(wn)),
        "negative_week_pct": float(len(wn) / len(weekly)) if len(weekly) else 0.0,
        "worst_week_return": float(weekly["return"].min()) if not weekly.empty else 0.0,
        "avg_week_return": float(weekly["return"].mean()) if not weekly.empty else 0.0,
    }
    row["decision"] = classify(row)
    row["stability_score"] = score(row)
    return row


def atoms(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out = []
    if "prev_day_return" in df.columns:
        s = pd.to_numeric(df["prev_day_return"], errors="coerce")
        out.append(("prev_day_return < 0", s < 0))
        out.append(("prev_day_return <= 0.015038", s <= 0.015038))
    cols = ["prev_day_return", "prev_vol20", "prev_range_pct", "prev_range20", "prev_atr20_pct", "prev_trend_20", "prev_trend_50", "gap_pct"]
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().nunique() < 5:
            continue
        for q in [0.20, 0.33, 0.50, 0.67, 0.80]:
            val = float(s.quantile(q))
            tag = int(round(q * 100))
            out.append((f"{col} <= q{tag} ({val:.6f})", s <= val))
            out.append((f"{col} >= q{tag} ({val:.6f})", s >= val))
        if "trend" in col or "gap" in col:
            out.append((f"{col} < 0", s < 0))
            out.append((f"{col} > 0", s > 0))
    if "dow" in df.columns:
        d = pd.to_numeric(df["dow"], errors="coerce")
        for v in sorted(d.dropna().unique()):
            iv = int(v)
            out.append((f"dow == {iv}", d == v))
            out.append((f"dow != {iv}", d != v))
    seen, unique = set(), []
    for label, mask in out:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, mask.fillna(False)))
    return unique


def safe_name(label: str) -> str:
    label = label.replace(" ", "_").replace("<=", "lte").replace(">=", "gte").replace("<", "lt").replace(">", "gt")
    return "".join(c if c.isalnum() or c in "._-=" else "_" for c in label)[:180]


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades_file = Path(args.trades_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not trades_file.exists():
        raise SystemExit(f"ERROR: trades file not found: {trades_file}")
    df = pd.read_csv(trades_file)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df = df.dropna(subset=["date", "net_return"]).copy()
    atom_list = atoms(df)
    rows = []
    tested = 0
    for size in range(1, args.max_filters + 1):
        for combo in itertools.combinations(atom_list, size):
            label = " AND ".join(x[0] for x in combo)
            mask = combo[0][1].copy()
            for _, m in combo[1:]:
                mask = mask & m
            kept = df[mask.fillna(False)].copy()
            if len(kept) < args.min_trades:
                continue
            rows.append(evaluate(label, kept))
            tested += 1
    if not rows:
        raise SystemExit("ERROR: no combinations met min_trades")
    results = pd.DataFrame(rows).sort_values(
        ["stability_score", "years_negative", "negative_month_pct", "pf"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    results_path = output_dir / "filter_combination_results.csv"
    report_path = output_dir / "filter_combination_report.json"
    results.to_csv(results_path, index=False)
    top = results.head(args.top_n).to_dict("records")
    for row in top[:10]:
        label = row["label"]
        mask = pd.Series(True, index=df.index)
        for part in label.split(" AND "):
            match = [m for l, m in atom_list if l == part]
            if match:
                mask = mask & match[0]
        kept = df[mask.fillna(False)].copy()
        kept.to_csv(output_dir / f"{safe_name(label)}_filtered_trades.csv", index=False)
        period_stats(kept, "Y").to_csv(output_dir / f"{safe_name(label)}_yearly.csv", index=False)
        period_stats(kept, "M").to_csv(output_dir / f"{safe_name(label)}_monthly.csv", index=False)
        period_stats(kept, "W").to_csv(output_dir / f"{safe_name(label)}_weekly.csv", index=False)
    report = {
        "ok": True,
        "source_trades_file": str(trades_file),
        "atoms": len(atom_list),
        "combinations_tested": tested,
        "min_trades": args.min_trades,
        "max_filters": args.max_filters,
        "decision_counts": results["decision"].value_counts().to_dict(),
        "top_by_score": top,
        "output_files": {"results": str(results_path), "report": str(report_path)},
        "warning": "Diagnostic only. Must pass walk-forward and cost stress before paper/live.",
    }
    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Search regime filter combinations on trades-with-features.")
    parser.add_argument("--trades-file", default=str(DEFAULT_TRADES_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-trades", type=int, default=35)
    parser.add_argument("--max-filters", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(clean(run(args)), indent=2))


if __name__ == "__main__":
    main()
