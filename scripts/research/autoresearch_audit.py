from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUT_DIR = Path("outputs/research/autoresearch")


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "path": str(path)}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_audit():
    strict_5y_path = Path("outputs/research/rankings_5y/multi_symbol_5m_break_even_1r_setup_ranking_report.json")
    diag_5y_path = Path("outputs/research/rankings_5y_diagnostic/multi_symbol_5m_break_even_1r_setup_ranking_report.json")
    yearly_report_path = Path("outputs/research/yearly_diagnostics/yearly_setup_performance_report.json")
    yearly_summary_path = Path("outputs/research/yearly_diagnostics/yearly_setup_performance_summary.csv")

    strict_5y = read_json(strict_5y_path)
    diag_5y = read_json(diag_5y_path)
    yearly_report = read_json(yearly_report_path)
    yearly_summary = read_csv(yearly_summary_path)

    findings = []
    blocked = []
    rescue = []

    if strict_5y:
        counts = strict_5y.get("decision_counts_cost_1x", {})
        top = strict_5y.get("top_candidates_cost_1x", [])
        findings.append(f"Strict 5Y ranking: {counts}")
        if not top:
            findings.append("Strict 5Y ranking did not approve any production candidate.")
            blocked.append("current_vwap_vol_keltner_family")

    if diag_5y:
        findings.append(f"Diagnostic 5Y ranking: {diag_5y.get('decision_counts_cost_1x', {})}")

    if yearly_report:
        findings.append(f"Yearly diagnostics: {yearly_report.get('decision_counts', {})}")

    if not yearly_summary.empty and "decision" in yearly_summary.columns:
        watch = yearly_summary[yearly_summary["decision"].astype(str) == "watch_regime_filter"].copy()
        if not watch.empty:
            watch = watch.sort_values(
                ["years_pass_edge_1x", "avg_expectancy_1x", "total_trades_1x"],
                ascending=[False, False, False],
            )
            for _, row in watch.head(10).iterrows():
                rescue.append({
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "setup_combo": row.get("setup_combo"),
                    "window_label": row.get("window_label"),
                    "years_pass": row.get("years_pass_edge_1x"),
                    "years_total": row.get("years_total"),
                    "trades": row.get("total_trades_1x"),
                    "median_pf": row.get("median_pf_1x"),
                    "min_pf": row.get("min_pf_1x"),
                    "avg_expectancy": row.get("avg_expectancy_1x"),
                    "max_dd": row.get("max_yearly_dd_1x"),
                })

    if rescue:
        findings.append("Main rescue path: NVDA 5m A_only/A_C 09:30-10:30 with regime filters and improved exits.")
    else:
        findings.append("No strong rescue candidate found. A new strategy family is recommended.")

    recommendations = [
        "Run regime-filter diagnostics on NVDA A_only/A_C 09:30-10:30.",
        "Build monthly and weekly consistency analyzer.",
        "Then test RSI, MACD, Bollinger Bands, moving averages, and ATR filters.",
        "Keep paper/live blocked until 5Y, monthly, weekly, walk-forward, and cost_2x/cost_3x validation pass.",
    ]

    return {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "karpathy_submodule": "vendor/karpathy-autoresearch",
        "program_file": "AUTORESEARCH_PROGRAM_TRADING.md",
        "live_money_status": "BLOCKED",
        "paper_status": "BLOCKED",
        "findings": findings,
        "blocked_promotions": blocked,
        "rescue_candidates": rescue,
        "recommendations": recommendations,
    }


def write_markdown(report, path: Path):
    lines = []
    lines.append("# Karpathy AutoResearch Trading Audit")
    lines.append("")
    lines.append(f"Created: {report['created_at']}")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"- Live money: **{report['live_money_status']}**")
    lines.append(f"- Paper trading: **{report['paper_status']}**")
    lines.append(f"- Karpathy submodule: `{report['karpathy_submodule']}`")
    lines.append(f"- Program file: `{report['program_file']}`")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    for item in report["findings"]:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Rescue Candidates")
    lines.append("")
    if report["rescue_candidates"]:
        for c in report["rescue_candidates"]:
            lines.append(
                f"- {c.get('symbol')} {c.get('timeframe')} {c.get('setup_combo')} "
                f"{c.get('window_label')} | years_pass={c.get('years_pass')}/{c.get('years_total')} "
                f"| trades={c.get('trades')} | medianPF={c.get('median_pf')} "
                f"| minPF={c.get('min_pf')} | maxDD={c.get('max_dd')}"
            )
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")

    lines.append("")
    lines.append("## Blocked Promotions")
    lines.append("")
    if report["blocked_promotions"]:
        for item in report["blocked_promotions"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report = build_audit()

    json_path = OUT_DIR / "latest_research_audit.json"
    md_path = OUT_DIR / "latest_research_audit.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps({
        "ok": True,
        "json": str(json_path),
        "markdown": str(md_path),
        "live_money_status": report["live_money_status"],
        "paper_status": report["paper_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
