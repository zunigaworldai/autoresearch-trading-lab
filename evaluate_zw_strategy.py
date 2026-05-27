from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.metrics import score_report
from strategies.zw_vwap_vol_keltner import generate_signals, DEFAULT_PARAMS


def load_sample_ohlcv() -> pd.DataFrame:
    """
    Temporary synthetic intraday OHLCV data.

    Later we replace this with real OHLCV exported from TradingView,
    Polygon, Alpaca, Schwab, Yahoo Finance, Binance, or another data source.
    """
    np.random.seed(42)

    periods = 2500
    index = pd.date_range(
        start="2025-01-02 09:30",
        periods=periods,
        freq="5min",
    )

    returns = np.random.normal(0.00008, 0.0035, periods)
    close = 100 * (1 + pd.Series(returns)).cumprod()

    open_ = close.shift(1).fillna(close.iloc[0])
    spread = np.random.uniform(0.001, 0.006, periods)

    high = pd.concat([open_, close], axis=1).max(axis=1) * (1 + spread)
    low = pd.concat([open_, close], axis=1).min(axis=1) * (1 - spread)

    volume_base = np.random.normal(100_000, 20_000, periods)
    volume_spike = np.random.choice([1.0, 1.5, 2.0, 3.0], size=periods, p=[0.82, 0.10, 0.06, 0.02])
    volume = np.maximum(volume_base * volume_spike, 1_000)

    return pd.DataFrame(
        {
            "datetime": index,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": volume,
        }
    )


def run_trade_simulation(signals: pd.DataFrame, cost_per_side: float = 0.0005) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Conservative single-position backtest.

    Execution model:
    - Entry at signal bar close.
    - Stop/TP checked from the next bars using high/low.
    - If stop and TP are touched in the same bar, stop is assumed first.
    - Cost is charged both entry and exit.
    """

    equity_values = []
    trade_returns = []
    trades = []

    cash_equity = 1.0

    position = 0
    entry_price = None
    entry_time = None
    stop_price = None
    tp_price = None
    setup = None

    for i, row in signals.iterrows():
        current_equity = cash_equity

        # Manage open position first.
        if position != 0:
            close_price = float(row["close"])

            if position == 1:
                unrealized = close_price / entry_price - 1.0
                current_equity = cash_equity * (1 + unrealized)

                stop_hit = float(row["low"]) <= stop_price
                tp_hit = float(row["high"]) >= tp_price

                if stop_hit or tp_hit:
                    exit_reason = "STOP" if stop_hit else "TP"
                    exit_price = stop_price if stop_hit else tp_price

                    gross_return = exit_price / entry_price - 1.0
                    net_return = gross_return - (cost_per_side * 2)

                    cash_equity *= 1 + net_return
                    current_equity = cash_equity

                    trade_returns.append(net_return)
                    trades.append(
                        {
                            "entry_time": str(entry_time),
                            "exit_time": str(row.get("datetime", i)),
                            "side": "LONG",
                            "setup": setup,
                            "entry": entry_price,
                            "exit": exit_price,
                            "stop": stop_price,
                            "tp": tp_price,
                            "exit_reason": exit_reason,
                            "gross_return": gross_return,
                            "net_return": net_return,
                        }
                    )

                    position = 0
                    entry_price = None
                    entry_time = None
                    stop_price = None
                    tp_price = None
                    setup = None

            elif position == -1:
                unrealized = entry_price / close_price - 1.0
                current_equity = cash_equity * (1 + unrealized)

                stop_hit = float(row["high"]) >= stop_price
                tp_hit = float(row["low"]) <= tp_price

                if stop_hit or tp_hit:
                    exit_reason = "STOP" if stop_hit else "TP"
                    exit_price = stop_price if stop_hit else tp_price

                    gross_return = entry_price / exit_price - 1.0
                    net_return = gross_return - (cost_per_side * 2)

                    cash_equity *= 1 + net_return
                    current_equity = cash_equity

                    trade_returns.append(net_return)
                    trades.append(
                        {
                            "entry_time": str(entry_time),
                            "exit_time": str(row.get("datetime", i)),
                            "side": "SHORT",
                            "setup": setup,
                            "entry": entry_price,
                            "exit": exit_price,
                            "stop": stop_price,
                            "tp": tp_price,
                            "exit_reason": exit_reason,
                            "gross_return": gross_return,
                            "net_return": net_return,
                        }
                    )

                    position = 0
                    entry_price = None
                    entry_time = None
                    stop_price = None
                    tp_price = None
                    setup = None

        # Enter only if flat after position management.
        if position == 0 and int(row["entry_signal"]) != 0:
            if not pd.isna(row["exit_stop"]) and not pd.isna(row["exit_tp"]):
                position = int(row["entry_signal"])
                entry_price = float(row["close"])
                entry_time = row.get("datetime", i)
                stop_price = float(row["exit_stop"])
                tp_price = float(row["exit_tp"])
                setup = row.get("setup", "")

        equity_values.append(current_equity)

    equity = pd.Series(equity_values, index=signals.index, name="equity")
    returns = equity.pct_change().fillna(0.0)
    trade_returns_series = pd.Series(trade_returns, name="trade_returns")
    trades_df = pd.DataFrame(trades)

    return equity, trade_returns_series, trades_df


def evaluate(params: dict | None = None, cost_per_side: float = 0.0005) -> dict:
    df = load_sample_ohlcv()
    signals = generate_signals(df, params or DEFAULT_PARAMS)

    equity, trade_returns, trades = run_trade_simulation(signals, cost_per_side=cost_per_side)

    report = score_report(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        trade_returns=trade_returns,
    )

    report["cost_per_side"] = cost_per_side
    report["total_trades"] = int(len(trades))

    if len(trades):
        report["setup_counts"] = trades["setup"].value_counts().to_dict()
        report["tp_hits"] = int((trades["exit_reason"] == "TP").sum())
        report["stop_hits"] = int((trades["exit_reason"] == "STOP").sum())
    else:
        report["setup_counts"] = {}
        report["tp_hits"] = 0
        report["stop_hits"] = 0

    return report, trades


def main():
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/trade_logs").mkdir(parents=True, exist_ok=True)

    params = DEFAULT_PARAMS.copy()

    results = {}

    for label, cost in {
        "cost_1x": 0.0005,
        "cost_2x": 0.0010,
        "cost_3x": 0.0015,
    }.items():
        report, trades = evaluate(params=params, cost_per_side=cost)
        results[label] = report

        trades.to_csv(f"experiments/trade_logs/zw_vwap_vol_keltner_{label}.csv", index=False)

    print(json.dumps(results, indent=2))

    with open("experiments/zw_vwap_vol_keltner_latest_result.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()