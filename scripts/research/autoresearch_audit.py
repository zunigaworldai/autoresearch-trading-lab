from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/research/autoresearch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD = OUT_DIR / "latest_research_audit.md"
REPORT_JSON = OUT_DIR / "latest_research_audit.json"


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, pd.Period)):
        return str(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if hasattr(x, "item"):
        try:
            return clean(x.item())
        except Exception:
            return str(x)
    return x


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fnum(x: Any, n: int = 4) -> str:
    try:
        y = float(x)
    except Exception:
        return "NA"
    if math.isnan(y) or math.isinf(y):
        return "NA"
    return f"{y:.{n}f}"


def add_json_rankings(lines: list[str]) -> None:
    paths = [
        ("Strict 5Y ranking", ROOT / "outputs/research/rankings_5y/multi_symbol_5m_break_even_1r_setup_ranking_report.json"),
        ("Diagnostic 5Y ranking", ROOT / "outputs/research/rankings_5y_diagnostic/multi_symbol_5m_break_even_1r_setup_ranking_report.json"),
        ("Monthly/weekly consistency", ROOT / "outputs/research/monthly_weekly_consistency/monthly_weekly_candidate_report.json"),
    ]

    for label, path in paths:
        data = read_json(path)
        if not data:
            continue
        counts = data.get("decision_counts_cost_1x") or data.get("decision_counts") or {}
        lines.append(f"- {label}: {counts}")

        top = data.get("top_candidates_cost_1x") or data.get("top_all_cost_1x") or data.get("top_summary") or []
        for r in top[:5]:
            lines.append(
                "  - "
                f"{r.get('symbol', r.get('label', 'NA'))} "
                f"{r.get('setup_combo', '')} {r.get('window_label', '')} | "
                f"trades={r.get('total_trades', r.get('trades', 'NA'))} "
                f"PF={fnum(r.get('profit_factor', r.get('pf')))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"Decision={r.get('decision', 'NA')}"
            )


def collect_csv_family(
    lines: list[str],
    family_name: str,
    folder: str,
    pattern: str,
    score_col: str = "strategy_score",
) -> pd.DataFrame:
    files = sorted((ROOT / folder).glob(pattern))
    frames = [read_csv(p) for p in files]
    frames = [df for df in frames if not df.empty]

    if not frames:
        lines.append(f"- {family_name}: no CSV outputs found.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    cost1 = df[df["cost_label"] == "cost_1x"].copy() if "cost_label" in df.columns else df.copy()

    lines.append(f"- {family_name} files: {len(files)}")
    lines.append(f"- {family_name} rows cost_1x: {len(cost1)}")

    if "decision" in cost1.columns:
        lines.append(f"- {family_name} decision counts: {cost1['decision'].value_counts().to_dict()}")

    if {"pf", "expectancy"}.issubset(cost1.columns):
        positive = cost1[(cost1["pf"] > 1.0) & (cost1["expectancy"] > 0)].copy()
        lines.append(f"- {family_name} positive-edge rows before yearly/monthly filters: {len(positive)}")
        if "years_negative" in positive.columns:
            zero_neg_years = positive[positive["years_negative"] == 0]
            lines.append(f"- {family_name} positive-edge rows with zero negative years: {len(zero_neg_years)}")

    sort_cols = [c for c in [score_col, "pf", "expectancy", "trades"] if c in cost1.columns]
    if sort_cols:
        top = cost1.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(12)
        for _, r in top.iterrows():
            variant = r.get("variant", r.get("setup_combo", "NA"))
            window = r.get("entry_window_label", r.get("window_label", "NA"))
            or_part = f" OR={r.get('or_minutes')}" if "or_minutes" in r.index else ""
            lines.append(
                f"- {family_name} "
                f"{r.get('symbol', 'NA')} {variant}{or_part} {window} "
                f"exit={r.get('exit_policy', 'NA')} | "
                f"trades={r.get('trades', r.get('total_trades', 'NA'))} "
                f"TPM={fnum(r.get('trades_per_calendar_month', r.get('trades_per_month')), 2)} "
                f"PF={fnum(r.get('pf', r.get('profit_factor')))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"YearsNeg={r.get('years_negative', 'NA')} "
                f"MonthsNeg={r.get('months_negative', 'NA')}/{r.get('months_total_active', 'NA')} "
                f"Decision={r.get('decision', 'NA')}"
            )

    return cost1


def collect_portfolios(lines: list[str]) -> list[dict[str, Any]]:
    reports = []
    for path in sorted((ROOT / "outputs/research/portfolio_frequency").glob("*_portfolio_report.json")):
        data = read_json(path)
        if data:
            reports.append(data)

    if not reports:
        lines.append("- Portfolio reports: none found.")
        return reports

    lines.append(f"- Portfolio reports: {len(reports)}")
    lines.append(f"- Portfolio decision counts: {dict(Counter([r.get('decision', 'NA') for r in reports]))}")

    for r in reports:
        lines.append(
            "- Portfolio "
            f"{r.get('label', 'NA')} | "
            f"symbols={r.get('symbols_count', 'NA')} "
            f"trades={r.get('total_trades', 'NA')} "
            f"TPW={fnum(r.get('trades_per_week_calendar'), 2)} "
            f"TPM={fnum(r.get('trades_per_calendar_month'), 2)} "
            f"PF={fnum(r.get('pf'))} "
            f"Exp={fnum(r.get('expectancy'), 6)} "
            f"DD={fnum(r.get('max_drawdown_trade_sequence'))} "
            f"YearsNeg={r.get('years_negative', 'NA')} "
            f"WorstMonth={fnum(r.get('worst_month_return'))} "
            f"Decision={r.get('decision', 'NA')}"
        )

    return reports


def block_if_no_candidate(blocked: set[str], name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return

    if "decision" in df.columns and not (df["decision"] == "candidate_review").any():
        blocked.add(name)

    if {"pf", "expectancy", "years_negative"}.issubset(df.columns):
        clean_candidate = df[
            (df["pf"] > 1.0)
            & (df["expectancy"] > 0)
            & (df["years_negative"] == 0)
        ]
        if clean_candidate.empty:
            blocked.add(f"{name}_no_zero_negative_year_candidate")


def main() -> None:
    findings: list[str] = []
    add_json_rankings(findings)

    time_df = collect_csv_family(
        findings,
        "Time-window",
        "outputs/research/time_windows_setup_combo",
        "*_time_windows_setup_combo.csv",
        score_col="window_score",
    )

    orb_full = collect_csv_family(
        findings,
        "Opening Range Breakout",
        "outputs/research/opening_range_breakout",
        "*_opening_range_breakout.csv",
        score_col="strategy_score",
    )

    orb_fast = collect_csv_family(
        findings,
        "Opening Range Breakout Fast",
        "outputs/research/opening_range_breakout_fast",
        "*_opening_range_breakout_fast.csv",
        score_col="strategy_score",
    )

    vwap_mr = collect_csv_family(
        findings,
        "VWAP Mean Reversion",
        "outputs/research/vwap_mean_reversion",
        "*_vwap_mean_reversion.csv",
        score_col="strategy_score",
    )

    portfolios = collect_portfolios(findings)

    blocked: set[str] = {"current_vwap_vol_keltner_family"}
    block_if_no_candidate(blocked, "vwap_vol_keltner_time_window_expansion", time_df)
    block_if_no_candidate(blocked, "opening_range_breakout_v1", orb_full)
    block_if_no_candidate(blocked, "opening_range_breakout_fast_v1", orb_fast)
    block_if_no_candidate(blocked, "vwap_mean_reversion_v1", vwap_mr)

    for r in portfolios:
        if float(r.get("pf", 0) or 0) <= 1.0 or float(r.get("expectancy", 0) or 0) <= 0:
            blocked.add(str(r.get("label", "portfolio_unknown")))

    recommendations = [
        "Keep live money BLOCKED.",
        "Keep paper trading BLOCKED until 5Y, monthly/weekly, cost stress, walk-forward, and portfolio-level checks pass.",
        "Current VWAP/VOL/Keltner family remains blocked as a production candidate.",
        "Opening Range Breakout v1/fast remains blocked unless a new regime-specific hypothesis is defined.",
        "VWAP Mean Reversion v1 remains blocked; QQQ evidence is strongly negative with both partial_50_at_1r_be and fixed exits.",
        "Next recommended experiment: RSI + Bollinger Bands Mean Reversion with sideways-regime filter.",
        "Alternative diagnostic: inverse-signal audit for VWAP MR to check whether the current conditions capture continuation instead of reversion.",
    ]

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "status": {
            "live_money": "BLOCKED",
            "paper_trading": "BLOCKED",
            "karpathy_submodule": "vendor/karpathy-autoresearch",
            "program_file": "AUTORESEARCH_PROGRAM_TRADING.md",
        },
        "findings": findings,
        "recommendations": recommendations,
        "blocked_promotions": sorted(blocked),
    }

    md = [
        "# Karpathy AutoResearch Trading Audit",
        "",
        f"Created: {report['created']}",
        "",
        "## Status",
        "",
        "- Live money: **BLOCKED**",
        "- Paper trading: **BLOCKED**",
        "- Karpathy submodule: `vendor/karpathy-autoresearch`",
        "- Program file: `AUTORESEARCH_PROGRAM_TRADING.md`",
        "",
        "## Findings",
        "",
    ]
    md.extend(findings or ["- No research outputs found."])
    md.extend(["", "## Recommendations", ""])
    md.extend([f"- {x}" for x in recommendations])
    md.extend(["", "## Blocked Promotions", ""])
    md.extend([f"- {x}" for x in sorted(blocked)])

    REPORT_JSON.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
