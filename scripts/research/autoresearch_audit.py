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
            return str(value)
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
        number = float(value)
    except Exception:
        return "NA"
    if math.isnan(number) or math.isinf(number):
        return "NA"
    return f"{number:.{digits}f}"


def add_json_report(lines: list[str], label: str, path: Path) -> None:
    data = read_json(path)
    if not data:
        return

    counts = data.get("decision_counts_cost_1x") or data.get("decision_counts") or {}
    lines.append(f"- {label}: {counts}")

    top = data.get("top_candidates_cost_1x") or data.get("top_all_cost_1x") or data.get("top_summary") or []
    for row in top[:5]:
        lines.append(
            "  - "
            f"{row.get('symbol', row.get('label', 'NA'))} "
            f"{row.get('variant', row.get('setup_combo', ''))} "
            f"{row.get('window_label', row.get('entry_window_label', ''))} | "
            f"trades={row.get('total_trades', row.get('trades', 'NA'))} "
            f"PF={fnum(row.get('profit_factor', row.get('pf')))} "
            f"Exp={fnum(row.get('expectancy'), 6)} "
            f"DD={fnum(row.get('max_drawdown'))} "
            f"YearsNeg={row.get('years_negative', 'NA')} "
            f"Decision={row.get('decision', 'NA')}"
        )


def collect_csv_family(
    lines: list[str],
    family_name: str,
    folder: str,
    pattern: str,
    score_col: str = "strategy_score",
) -> pd.DataFrame:
    files = sorted((ROOT / folder).glob(pattern))
    frames = [read_csv(file) for file in files]
    frames = [frame for frame in frames if not frame.empty]

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
            zero_negative_years = positive[positive["years_negative"] == 0]
            lines.append(f"- {family_name} positive-edge rows with zero negative years: {len(zero_negative_years)}")

    sort_cols = [col for col in [score_col, "pf", "expectancy", "trades"] if col in cost1.columns]
    if sort_cols:
        top = cost1.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(12)
        for _, row in top.iterrows():
            variant = row.get("variant", row.get("setup_combo", "NA"))
            window = row.get("entry_window_label", row.get("window_label", "NA"))
            or_part = f" OR={row.get('or_minutes')}" if "or_minutes" in row.index and pd.notna(row.get("or_minutes")) else ""
            lines.append(
                f"- {family_name} "
                f"{row.get('symbol', 'NA')} {variant}{or_part} {window} "
                f"exit={row.get('exit_policy', 'NA')} | "
                f"trades={row.get('trades', row.get('total_trades', 'NA'))} "
                f"TPM={fnum(row.get('trades_per_calendar_month', row.get('trades_per_month')), 2)} "
                f"PF={fnum(row.get('pf', row.get('profit_factor')))} "
                f"Exp={fnum(row.get('expectancy'), 6)} "
                f"DD={fnum(row.get('max_drawdown'))} "
                f"YearsNeg={row.get('years_negative', 'NA')} "
                f"MonthsNeg={row.get('months_negative', 'NA')}/{row.get('months_total_active', 'NA')} "
                f"Decision={row.get('decision', 'NA')}"
            )

    return cost1


def collect_portfolios(lines: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    for path in sorted((ROOT / "outputs/research/portfolio_frequency").glob("*_portfolio_report.json")):
        data = read_json(path)
        if data:
            reports.append(data)

    if not reports:
        lines.append("- Portfolio reports: none found.")
        return reports

    lines.append(f"- Portfolio reports: {len(reports)}")
    lines.append(f"- Portfolio decision counts: {dict(Counter([r.get('decision', 'NA') for r in reports]))}")

    for row in reports:
        lines.append(
            "- Portfolio "
            f"{row.get('label', 'NA')} | "
            f"symbols={row.get('symbols_count', 'NA')} "
            f"trades={row.get('total_trades', 'NA')} "
            f"TPW={fnum(row.get('trades_per_week_calendar'), 2)} "
            f"TPM={fnum(row.get('trades_per_calendar_month'), 2)} "
            f"PF={fnum(row.get('pf'))} "
            f"Exp={fnum(row.get('expectancy'), 6)} "
            f"DD={fnum(row.get('max_drawdown_trade_sequence'))} "
            f"YearsNeg={row.get('years_negative', 'NA')} "
            f"WorstMonth={fnum(row.get('worst_month_return'))} "
            f"Decision={row.get('decision', 'NA')}"
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

    add_json_report(
        findings,
        "Strict 5Y ranking",
        ROOT / "outputs/research/rankings_5y/multi_symbol_5m_break_even_1r_setup_ranking_report.json",
    )
    add_json_report(
        findings,
        "Diagnostic 5Y ranking",
        ROOT / "outputs/research/rankings_5y_diagnostic/multi_symbol_5m_break_even_1r_setup_ranking_report.json",
    )
    add_json_report(
        findings,
        "Monthly/weekly consistency",
        ROOT / "outputs/research/monthly_weekly_consistency/monthly_weekly_candidate_report.json",
    )

    families = [
        ("Time-window", "outputs/research/time_windows_setup_combo", "*_time_windows_setup_combo.csv", "window_score", "vwap_vol_keltner_time_window_expansion"),
        ("Opening Range Breakout", "outputs/research/opening_range_breakout", "*_opening_range_breakout.csv", "strategy_score", "opening_range_breakout_v1"),
        ("Opening Range Breakout Fast", "outputs/research/opening_range_breakout_fast", "*_opening_range_breakout_fast.csv", "strategy_score", "opening_range_breakout_fast_v1"),
        ("VWAP Mean Reversion", "outputs/research/vwap_mean_reversion", "*_vwap_mean_reversion.csv", "strategy_score", "vwap_mean_reversion_v1"),
        ("RSI Bollinger Mean Reversion", "outputs/research/rsi_bollinger_mean_reversion", "*_rsi_bollinger_mean_reversion.csv", "strategy_score", "rsi_bollinger_mean_reversion_v1"),
        ("Inverse Signal Audit", "outputs/research/inverse_signal_audit", "*_inverse_signal_audit.csv", "strategy_score", "inverse_signal_audit_v1"),
        ("VWAP Reclaim Continuation", "outputs/research/vwap_reclaim_continuation", "*_vwap_reclaim_continuation.csv", "strategy_score", "vwap_reclaim_continuation_v1"),
    ]

    blocked: set[str] = {"current_vwap_vol_keltner_family"}

    for family_name, folder, pattern, score_col, block_name in families:
        df = collect_csv_family(findings, family_name, folder, pattern, score_col)
        block_if_no_candidate(blocked, block_name, df)

    portfolios = collect_portfolios(findings)
    for row in portfolios:
        if float(row.get("pf", 0) or 0) <= 1.0 or float(row.get("expectancy", 0) or 0) <= 0:
            blocked.add(str(row.get("label", "portfolio_unknown")))

    recommendations = [
        "Keep live money BLOCKED.",
        "Keep paper trading BLOCKED until 5Y, monthly/weekly, cost stress, walk-forward, and portfolio-level checks pass.",
        "Current VWAP/VOL/Keltner family remains blocked as a production candidate.",
        "Opening Range Breakout v1/fast remains blocked unless a new regime-specific hypothesis is defined.",
        "VWAP Mean Reversion v1 remains blocked; QQQ evidence is strongly negative with both partial_50_at_1r_be and fixed exits.",
        "RSI + Bollinger Mean Reversion v1 remains blocked; QQQ evidence has no positive expectancy.",
        "Inverse-signal audit remains blocked; inversion did not convert failed mean-reversion signals into profitable continuation.",
        "VWAP Reclaim / Continuation v1 remains blocked at multi-symbol level; no candidate_review rows were found.",
        "Next recommended family: Opening Drive Continuation.",
        "Alternative next family: EMA Pullback Trend Scalping.",
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
    md.extend([f"- {item}" for item in recommendations])
    md.extend(["", "## Blocked Promotions", ""])
    md.extend([f"- {item}" for item in sorted(blocked)])

    REPORT_JSON.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
