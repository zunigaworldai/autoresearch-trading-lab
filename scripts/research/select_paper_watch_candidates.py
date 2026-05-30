from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SUMMARY = Path("outputs/research/validation/is_oos_validation_summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/validation")

SETUP_COMPLEXITY = {
    "A_only": 1,
    "B_only": 1,
    "C_only": 1,
    "A_B": 2,
    "A_C": 2,
    "B_C": 2,
    "A_B_C": 3,
}


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return clean_for_json(value.item())
        except Exception:
            pass
    return value


def norm(value: float, cap: float, floor: float = 0.0) -> float:
    if cap <= floor:
        return 0.0
    value = max(floor, min(float(value), cap))
    return (value - floor) / (cap - floor)


def drawdown_score(dd: float, max_dd: float = 0.08) -> float:
    return max(0.0, 1.0 - min(max(float(dd), 0.0), max_dd) / max_dd)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: summary file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"ERROR: summary file is empty: {path}")
    return df


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "IS_trades", "IS_pf", "IS_expectancy", "IS_dd",
        "OOS_trades", "OOS_trades_per_month", "OOS_pf", "OOS_expectancy",
        "OOS_dd", "OOS_largest_win_share",
        "OOS_2x_pf", "OOS_2x_expectancy", "OOS_3x_pf", "OOS_3x_expectancy",
    ]

    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["setup_complexity"] = out["setup_combo"].map(SETUP_COMPLEXITY).fillna(9).astype(int)

    # Total-return logic: a lower expectancy strategy with many more trades can be superior
    # if total expected return is higher and it survives risk/cost filters.
    out["OOS_total_expected_return"] = out["OOS_expectancy"] * out["OOS_trades"]
    out["OOS_expected_monthly_return"] = out["OOS_expectancy"] * out["OOS_trades_per_month"]

    out["OOS_2x_total_expected_return"] = out["OOS_2x_expectancy"] * out["OOS_trades"]
    out["OOS_3x_total_expected_return"] = out["OOS_3x_expectancy"] * out["OOS_trades"]

    out["cost_2x_survival"] = (out["OOS_2x_pf"] > 1.0) & (out["OOS_2x_expectancy"] > 0.0)
    out["cost_3x_survival"] = (out["OOS_3x_pf"] > 1.0) & (out["OOS_3x_expectancy"] > 0.0)

    out["edge_quality_score"] = (
        0.45 * out["OOS_pf"].apply(lambda x: norm(x, 3.0))
        + 0.35 * out["OOS_expectancy"].apply(lambda x: norm(x, 0.006))
        + 0.20 * out["OOS_2x_pf"].apply(lambda x: norm(x, 2.5))
    )

    out["frequency_score"] = out["OOS_trades_per_month"].apply(lambda x: norm(x, 15.0))

    out["total_return_score"] = (
        0.55 * out["OOS_expected_monthly_return"].apply(lambda x: norm(x, 0.04))
        + 0.45 * out["OOS_total_expected_return"].apply(lambda x: norm(x, 0.08))
    )

    out["risk_score"] = (
        0.70 * out["OOS_dd"].apply(drawdown_score)
        + 0.30 * (1.0 - out["OOS_largest_win_share"].clip(lower=0.0, upper=1.0))
    )

    out["cost_stress_score"] = (
        0.60 * out["OOS_2x_pf"].apply(lambda x: norm(x, 2.5))
        + 0.40 * out["OOS_3x_pf"].apply(lambda x: norm(x, 2.0))
    )

    out["paper_priority"] = out["decision"].map({
        "candidate_paper_watch": 1,
        "watch_cost_sensitive": 2,
        "fail_oos": 3,
        "fail_is": 4,
    }).fillna(9).astype(int)

    # Final score: not just average gain, not just total gain.
    # It favors total robust profitability with edge, risk, frequency, and cost stress.
    out["portfolio_score"] = (
        0.30 * out["total_return_score"]
        + 0.25 * out["edge_quality_score"]
        + 0.20 * out["risk_score"]
        + 0.15 * out["frequency_score"]
        + 0.10 * out["cost_stress_score"]
    )

    return out


def dedupe(df: pd.DataFrame, include_cost_sensitive: bool) -> pd.DataFrame:
    allowed = ["candidate_paper_watch"]
    if include_cost_sensitive:
        allowed.append("watch_cost_sensitive")

    work = df[df["decision"].isin(allowed)].copy()

    if work.empty:
        return work

    # Deduplicate equivalent candidates:
    # A_B vs A_B_C or B_only vs B_C often generate identical trades.
    work["fingerprint"] = (
        work["symbol"].astype(str) + "|"
        + work["timeframe"].astype(str) + "|"
        + work["window_label"].astype(str) + "|"
        + work["session_start"].astype(str) + "|"
        + work["session_end"].astype(str) + "|"
        + work["OOS_trades"].round(0).astype(str) + "|"
        + work["OOS_pf"].round(4).astype(str) + "|"
        + work["OOS_expectancy"].round(6).astype(str) + "|"
        + work["OOS_dd"].round(4).astype(str)
    )

    work = work.sort_values(
        [
            "fingerprint",
            "paper_priority",
            "portfolio_score",
            "setup_complexity",
            "OOS_expected_monthly_return",
            "OOS_pf",
            "OOS_2x_pf",
        ],
        ascending=[True, True, False, True, False, False, False],
    )

    out = work.drop_duplicates("fingerprint", keep="first").copy()

    out = out.sort_values(
        [
            "paper_priority",
            "portfolio_score",
            "OOS_expected_monthly_return",
            "OOS_total_expected_return",
            "OOS_pf",
            "OOS_2x_pf",
            "OOS_trades_per_month",
        ],
        ascending=[True, False, False, False, False, False, False],
    ).reset_index(drop=True)

    out["rank"] = range(1, len(out) + 1)
    return out


def build_report(raw: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    report = {
        "ok": True,
        "raw_rows": int(len(raw)),
        "selected_rows": int(len(selected)),
        "decision_counts_raw": raw["decision"].value_counts().to_dict() if "decision" in raw else {},
        "decision_counts_selected": selected["decision"].value_counts().to_dict() if not selected.empty else {},
        "top_selected": selected.head(25).to_dict("records"),
        "best_by_symbol": {},
        "ranking_logic": {
            "OOS_total_expected_return": "OOS_expectancy * OOS_trades",
            "OOS_expected_monthly_return": "OOS_expectancy * OOS_trades_per_month",
            "portfolio_score": "weighted blend of total return, edge quality, drawdown/miracle-trade risk, frequency, and cost-stress survival",
            "dedupe": "deduplicates equivalent candidates and prefers simpler setup when metrics are equivalent",
        },
        "recommendations": [],
    }

    if not selected.empty:
        for symbol, group in selected.groupby("symbol"):
            report["best_by_symbol"][symbol] = group.head(5).to_dict("records")

        best = selected.iloc[0]
        report["recommendations"].append(
            f"Top paper-watch candidate: {best['symbol']} {best['timeframe']} "
            f"{best['setup_combo']} {best['window_label']} "
            f"OOS_PF={best['OOS_pf']:.4f}, OOS_Exp={best['OOS_expectancy']:.6f}, "
            f"OOS_TPM={best['OOS_trades_per_month']:.2f}, "
            f"OOS_expected_monthly_return={best['OOS_expected_monthly_return']:.6f}, "
            f"portfolio_score={best['portfolio_score']:.4f}."
        )

        freq = selected.sort_values(
            ["OOS_expected_monthly_return", "OOS_trades_per_month", "OOS_pf"],
            ascending=[False, False, False],
        ).iloc[0]

        report["recommendations"].append(
            f"Best total-return/frequency balance: {freq['symbol']} {freq['setup_combo']} "
            f"{freq['window_label']} with OOS_expected_monthly_return="
            f"{freq['OOS_expected_monthly_return']:.6f} and OOS_TPM={freq['OOS_trades_per_month']:.2f}."
        )

        report["recommendations"].append(
            "Next step: create paper execution watchlist and forward-test with real-time alerts, no real money."
        )
    else:
        report["recommendations"].append("No candidates selected. Expand universe/timeframes or relax thresholds.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select deduplicated paper-watch candidates.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include-cost-sensitive", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_summary(Path(args.summary))
    scored = add_scores(raw)
    selected = dedupe(scored, include_cost_sensitive=args.include_cost_sensitive)

    scored_path = output_dir / "paper_watch_scored_all.csv"
    selected_path = output_dir / "paper_watch_candidates.csv"
    report_path = output_dir / "paper_watch_candidates_report.json"

    scored.to_csv(scored_path, index=False)
    selected.to_csv(selected_path, index=False)

    report = build_report(scored, selected)
    report["output_files"] = {
        "scored_all": str(scored_path),
        "selected": str(selected_path),
        "report": str(report_path),
    }

    report_path.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
