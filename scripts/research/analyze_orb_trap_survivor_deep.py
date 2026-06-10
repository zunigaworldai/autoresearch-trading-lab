from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
file = ROOT / "outputs/research/opening_failed_breakout_reversal/survivor_trades/NVDA_both_or15_fail_rr15_extreme_afternoon_1300_1500_cost_1x_2x_3x_combined_trades.csv"

df = pd.read_csv(file)

time_col = "entry_time"
df["entry_dt"] = pd.to_datetime(df[time_col], errors="coerce")
df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")
df["year"] = df["entry_dt"].dt.year
df["month"] = df["entry_dt"].dt.month
df["month_label"] = df["entry_dt"].dt.to_period("M").astype(str)
df["entry_hhmm"] = df["entry_dt"].dt.strftime("%H:%M")
df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
df["gross_return"] = pd.to_numeric(df["gross_return"], errors="coerce")


def pf(s: pd.Series) -> float:
    gains = s[s > 0].sum()
    losses = s[s < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def stats(group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols)
        .agg(
            trades=("net_return", "count"),
            total_return=("net_return", lambda x: (1 + x).prod() - 1),
            expectancy=("net_return", "mean"),
            pf=("net_return", pf),
            win_rate=("net_return", lambda x: (x > 0).mean()),
            avg_gross=("gross_return", "mean"),
        )
        .reset_index()
    )


fmt = {
    "total_return": "{:.4f}".format,
    "expectancy": "{:.6f}".format,
    "pf": "{:.4f}".format,
    "win_rate": "{:.2%}".format,
    "avg_gross": "{:.6f}".format,
}

print("\n=== BY COST / SIDE ===")
print(stats(["cost_label", "side"]).to_string(index=False, formatters=fmt))

print("\n=== BY COST / EXIT REASON ===")
print(stats(["cost_label", "exit_reason"]).to_string(index=False, formatters=fmt))

print("\n=== BY COST / PARTIAL TAKEN ===")
print(stats(["cost_label", "partial_taken"]).to_string(index=False, formatters=fmt))

print("\n=== BY COST / MONTH OF YEAR ===")
moy = stats(["cost_label", "month"]).sort_values(["cost_label", "month"])
print(moy.to_string(index=False, formatters=fmt))

print("\n=== BY COST / ENTRY TIME ===")
et = stats(["cost_label", "entry_hhmm"]).sort_values(["cost_label", "entry_hhmm"])
print(et.to_string(index=False, formatters=fmt))

print("\n=== COST_3X WORST TRADES ===")
worst = df[df["cost_label"] == "cost_3x"].sort_values("net_return").head(15)
cols = [
    "entry_time", "exit_time", "side", "entry", "exit", "stop", "tp",
    "exit_reason", "partial_taken", "gross_return", "net_return"
]
print(worst[cols].to_string(index=False, formatters={
    "entry": "{:.4f}".format,
    "exit": "{:.4f}".format,
    "stop": "{:.4f}".format,
    "tp": "{:.4f}".format,
    "gross_return": "{:.6f}".format,
    "net_return": "{:.6f}".format,
}))

print("\n=== COST_3X BEST TRADES ===")
best = df[df["cost_label"] == "cost_3x"].sort_values("net_return", ascending=False).head(15)
print(best[cols].to_string(index=False, formatters={
    "entry": "{:.4f}".format,
    "exit": "{:.4f}".format,
    "stop": "{:.4f}".format,
    "tp": "{:.4f}".format,
    "gross_return": "{:.6f}".format,
    "net_return": "{:.6f}".format,
}))
