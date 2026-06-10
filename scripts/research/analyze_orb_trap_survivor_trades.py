from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
file = ROOT / "outputs/research/opening_failed_breakout_reversal/survivor_trades/NVDA_both_or15_fail_rr15_extreme_afternoon_1300_1500_cost_1x_2x_3x_combined_trades.csv"
df = pd.read_csv(file)

time_col = None
for col in ["entry_time", "entry_timestamp", "timestamp", "datetime", "date"]:
    if col in df.columns:
        time_col = col
        break

if time_col is None:
    raise SystemExit(f"No time column found. Columns: {list(df.columns)}")

df["entry_dt"] = pd.to_datetime(df[time_col], errors="coerce")
df["year"] = df["entry_dt"].dt.year
df["month"] = df["entry_dt"].dt.to_period("M").astype(str)
df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")


def pf(s: pd.Series) -> float:
    gains = s[s > 0].sum()
    losses = s[s < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


print("\n=== YEARLY SURVIVOR TRADES ===")
yearly = df.groupby(["cost_label", "year"]).agg(
    trades=("net_return", "count"),
    total_return=("net_return", lambda x: (1 + x).prod() - 1),
    expectancy=("net_return", "mean"),
    pf=("net_return", pf),
    win_rate=("net_return", lambda x: (x > 0).mean()),
).reset_index()

print(yearly.to_string(index=False, formatters={
    "total_return": "{:.4f}".format,
    "expectancy": "{:.6f}".format,
    "pf": "{:.4f}".format,
    "win_rate": "{:.2%}".format,
}))

print("\n=== WORST MONTHS ===")
monthly = df.groupby(["cost_label", "month"]).agg(
    trades=("net_return", "count"),
    total_return=("net_return", lambda x: (1 + x).prod() - 1),
    expectancy=("net_return", "mean"),
    pf=("net_return", pf),
).reset_index()

monthly = monthly.sort_values(["cost_label", "total_return"], ascending=[True, True])
print(monthly.groupby("cost_label").head(10).to_string(index=False, formatters={
    "total_return": "{:.4f}".format,
    "expectancy": "{:.6f}".format,
    "pf": "{:.4f}".format,
}))
