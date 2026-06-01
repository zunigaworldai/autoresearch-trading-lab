from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/research/autoresearch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_MD = OUTPUT_DIR / "latest_research_audit.md"
REPORT_JSON = OUTPUT_DIR / "latest_research_audit.json"


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


def fnum(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(x) or math.isinf(x):
        return "NA"
    return f"{x:.{digits}f}"


def add_old_rankings(lines: list[str]) -> None:
    paths = [
        ("Strict 5Y ranking", ROOT / "outputs/research/rankings_5y/multi_symbol_5m_break_even_1r_setup_ranking_report.json"),
        ("Diagnostic 5Y ranking", ROOT / "outputs/research/rankings_5y_diagnostic/multi_symbol_5m_break_even_1r_setup_ranking_report.json"),
        ("Legacy ranking", ROOT / "outputs/research/rankings/multi_symbol_5m_break_even_1r_setup_ranking_report.json"),
    ]
    for name, path in paths:
        data = read_json(path)
        if not data:
            continue
        lines.append(f"- {name}: {data.get('decision_counts_cost_1x', {})}")
        top = data.get("top_candidates_cost_1x") or data.get("top_all_cost_1x") or []
        if top:
            r = top[0]
            lines.append(
                "- Top ranking row: "
                f"{r.get('symbol', 'NA')} {r.get('setup_combo', 'NA')} {r.get('window_label', 'NA')} | "
                f"trades={r.get('total_trades', r.get('trades', 'NA'))} "
                f"PF={fnum(r.get('profit_factor', r.get('pf')))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"Decision={r.get('decision', 'NA')}"
            )


def add_monthly_weekly(lines: list[str]) -> None:
    data = read_json(ROOT / "outputs/research/monthly_weekly_consistency/monthly_weekly_candidate_report.json")
    if not data:
        return
    lines.append(f"- Monthly/weekly candidates tested: {data.get('candidates_tested', 'NA')}")
    lines.append(f"- Monthly/weekly decision counts: {data.get('decision_counts', {})}")
    for r in (data.get("top_summary") or [])[:8]:
        lines.append(
            "- "
            f"{r.get('label', 'NA')} | "
            f"trades={r.get('trades', 'NA')} "
            f"PF={fnum(r.get('pf'))} "
            f"Exp={fnum(r.get('expectancy'), 6)} "
            f"DD={fnum(r.get('max_drawdown'))} "
            f"YearsNeg={r.get('years_negative', 'NA')} "
            f"MonthsNeg={r.get('months_negative', 'NA')}/{r.get('months_total', 'NA')} "
            f"WorstMonth={fnum(r.get('worst_month_return'))} "
            f"Decision={r.get('decision', 'NA')}"
        )


def add_filter_combinations(lines: list[str]) -> None:
    for path in sorted((ROOT / "outputs/research").glob("filter_combinations*/filter_combination_report.json")):
        data = read_json(path)
        if not data:
            continue
        lines.append(
            f"- {path.parent.name}: atoms={data.get('atoms', 'NA')} "
            f"combinations={data.get('combinations_tested', 'NA')} "
            f"decisions={data.get('decision_counts', {})}"
        )
        for r in (data.get("top_by_score") or [])[:5]:
            lines.append(
                "  - "
                f"{str(r.get('label', 'NA'))[:120]} | "
                f"trades={r.get('trades', 'NA')} "
                f"PF={fnum(r.get('pf'))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"YearsNeg={r.get('years_negative', 'NA')} "
                f"MonthsNeg={r.get('months_negative', 'NA')}/{r.get('months_total', 'NA')} "
                f"WorstMonth={fnum(r.get('worst_month_return'))} "
                f"Decision={r.get('decision', 'NA')}"
            )


def collect_time_windows(lines: list[str]) -> pd.DataFrame:
    files = sorted((ROOT / "outputs/research/time_windows_setup_combo").glob("*_time_windows_setup_combo.csv"))
    frames = [read_csv(p) for p in files]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    cost1 = df[df["cost_label"] == "cost_1x"].copy() if "cost_label" in df.columns else df.copy()
    lines.append(f"- Time-window files: {len(files)}")
    lines.append(f"- Time-window rows cost_1x: {len(cost1)}")
    if "decision" in cost1.columns:
        lines.append(f"- Time-window decision counts: {cost1['decision'].value_counts().to_dict()}")
    if {"pf", "expectancy"}.issubset(cost1.columns):
        positive = cost1[(cost1["pf"] > 1.0) & (cost1["expectancy"] > 0)]
        lines.append(f"- Time-window positive-edge rows before yearly/monthly filters: {len(positive)}")
    sort_cols = [c for c in ["window_score", "pf", "expectancy", "trades"] if c in cost1.columns]
    if sort_cols:
        top = cost1.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(12)
        for _, r in top.iterrows():
            lines.append(
                "- "
                f"{r.get('symbol', 'NA')} {r.get('setup_combo', 'NA')} {r.get('window_label', 'NA')} | "
                f"trades={r.get('trades', 'NA')} "
                f"TPM={fnum(r.get('trades_per_calendar_month'), 2)} "
                f"PF={fnum(r.get('pf'))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"YearsNeg={r.get('years_negative', 'NA')} "
                f"MonthsNeg={r.get('months_negative', 'NA')}/{r.get('months_total_active', 'NA')} "
                f"Decision={r.get('decision', 'NA')}"
            )
    return cost1


def collect_orb_outputs(lines: list[str]) -> pd.DataFrame:
    files = []
    files.extend(sorted((ROOT / "outputs/research/opening_range_breakout").glob("*_opening_range_breakout.csv")))
    files.extend(sorted((ROOT / "outputs/research/opening_range_breakout_fast").glob("*_opening_range_breakout_fast.csv")))

    frames = [read_csv(p) for p in files]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    cost1 = df[df["cost_label"] == "cost_1x"].copy() if "cost_label" in df.columns else df.copy()

    lines.append(f"- ORB files: {len(files)}")
    lines.append(f"- ORB rows cost_1x: {len(cost1)}")
    if "decision" in cost1.columns:
        lines.append(f"- ORB decision counts: {cost1['decision'].value_counts().to_dict()}")

    if {"pf", "expectancy"}.issubset(cost1.columns):
        positive = cost1[(cost1["pf"] > 1.0) & (cost1["expectancy"] > 0)].copy()
        lines.append(f"- ORB positive-edge rows before yearly/monthly filters: {len(positive)}")
        if not positive.empty and "years_negative" in positive.columns:
            clean_positive = positive[positive["years_negative"] == 0]
            lines.append(f"- ORB positive-edge rows with zero negative years: {len(clean_positive)}")

    sort_cols = [c for c in ["strategy_score", "pf", "expectancy", "trades"] if c in cost1.columns]
    if sort_cols:
        top = cost1.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(12)
        for _, r in top.iterrows():
            label = r.get("entry_window_label", r.get("window_label", "NA"))
            lines.append(
                "- ORB "
                f"{r.get('symbol', 'NA')} {r.get('variant', 'NA')} "
                f"OR={r.get('or_minutes', 'NA')} {label} | "
                f"trades={r.get('trades', 'NA')} "
                f"TPM={fnum(r.get('trades_per_calendar_month'), 2)} "
                f"PF={fnum(r.get('pf'))} "
                f"Exp={fnum(r.get('expectancy'), 6)} "
                f"DD={fnum(r.get('max_drawdown'))} "
                f"YearsNeg={r.get('years_negative', 'NA')} "
                f"MonthsNeg={r.get('months_negative', 'NA')}/{r.get('months_total_active', 'NA')} "
                f"Decision={r.get('decision', 'NA')}"
            )

    return cost1


def collect_portfolios(lines: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs/research/portfolio_frequency").glob("*_portfolio_report.json")):
        data = read_json(path)
        if data:
            reports.append(data)
    if not reports:
        return reports
    lines.append(f"- Portfolio reports: {len(reports)}")
    lines.append(f"- Portfolio decision counts: {dict(Counter([r.get('decision', 'NA') for r in reports]))}")
    for r in reports:
        lines.append(
            "- "
            f"{r.get('label', 'NA')} | "
            f"symbols={r.get('symbols_count', 'NA')} "
            f"trades={r.get('total_trades', 'NA')} "
            f"TPW={fnum(r.get('trades_per_week_calendar'), 2)} "
            f"TPM={fnum(r.get('trades_per_calendar_month'), 2)} "
            f"PF={fnum(r.get('pf'))} "
            f"Exp={fnum(r.get('expectancy'), 6)} "
            f"DD={fnum(r.get('max_drawdown_trade_sequence'))} "
            f"YearsNeg={r.get('years_negative', 'NA')} "
            f"MonthNegCal={fnum(r.get('negative_month_pct_calendar'), 4)} "
            f"WorstMonth={fnum(r.get('worst_month_return'))} "
            f"Decision={r.get('decision', 'NA')}"
        )
    return reports


def blocked_promotions(time_df: pd.DataFrame, orb_df: pd.DataFrame, portfolios: list[dict[str, Any]]) -> list[str]:
    blocked = {"current_vwap_vol_keltner_family"}

    if not time_df.empty and {"pf", "expectancy"}.issubset(time_df.columns):
        bad = time_df[(time_df["pf"] <= 1.0) | (time_df["expectancy"] <= 0)]
        if len(bad):
            blocked.add("vwap_vol_keltner_time_window_expansion")

    if not orb_df.empty:
        if "decision" in orb_df.columns:
            if not (orb_df["decision"] == "candidate_review").any():
                blocked.add("opening_range_breakout_v1")
        if {"pf", "expectancy", "years_negative"}.issubset(orb_df.columns):
            clean = orb_df[(orb_df["pf"] > 1.0) & (orb_df["expectancy"] > 0) & (orb_df["years_negative"] == 0)]
            if clean.empty:
                blocked.add("opening_range_breakout_v1_no_zero_negative_year_candidate")

    for r in portfolios:
        if float(r.get("pf", 0) or 0) <= 1.0 or float(r.get("expectancy", 0) or 0) <= 0:
            blocked.add(str(r.get("label", "portfolio_unknown")))

    return sorted(blocked)


def main() -> None:
    findings: list[str] = []
    add_old_rankings(findings)
    add_monthly_weekly(findings)
    add_filter_combinations(findings)
    time_df = collect_time_windows(findings)
    orb_df = collect_orb_outputs(findings)
    portfolios = collect_portfolios(findings)

    blocked = blocked_promotions(time_df, orb_df, portfolios)

    recommendations = [
        "Keep live money BLOCKED.",
        "Keep paper trading BLOCKED until 5Y, monthly/weekly, cost stress, walk-forward, and portfolio-level checks pass.",
        "Current VWAP/VOL/Keltner family remains blocked as a production candidate.",
        "Opening Range Breakout v1/fast remains blocked because no zero-negative-year candidate was found.",
        "Do not continue broad ORB brute-force without a new hypothesis; consider regime-specific ORB only if explicitly scoped.",
        "Start the next family experiment: VWAP Mean Reversion.",
        "For any new family, evaluate full RTH time windows per symbol first, then portfolio frequency and stability.",
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
        "blocked_promotions": blocked,
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
        *(findings or ["- No research outputs found."]),
        "",
        "## Recommendations",
        "",
    ]
    md.extend([f"- {x}" for x in recommendations])
    md.extend(["", "## Blocked Promotions", ""])
    md.extend([f"- {x}" for x in blocked])

    REPORT_JSON.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
