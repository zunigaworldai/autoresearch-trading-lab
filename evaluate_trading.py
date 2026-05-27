import json
import pandas as pd
import numpy as np

from strategies.candidate_strategy import generate_signals, PARAMS
from engine.metrics import score_report


def load_sample_data() -> pd.DataFrame:
    """
    Temporary synthetic market data.
    Later we replace this with real TradingView/exported OHLCV data.
    """
    np.random.seed(42)
    n = 1000
    returns = np.random.normal(0.0003, 0.01, n)
    close = 100 * (1 + pd.Series(returns)).cumprod()

    df = pd.DataFrame({
        "close": close
    })

    return df


def run_backtest(df: pd.DataFrame, cost_per_trade: float = 0.0005) -> dict:
    signals = generate_signals(df, PARAMS)

    signals["market_return"] = signals["close"].pct_change().fillna(0)
    signals["position"] = signals["signal"].shift(1).fillna(0)
    signals["turnover"] = signals["position"].diff().abs().fillna(0)

    signals["strategy_return"] = (
        signals["position"] * signals["market_return"]
        - signals["turnover"] * cost_per_trade
    )

    signals["equity"] = (1 + signals["strategy_return"]).cumprod()

    trade_returns = signals.loc[signals["turnover"] > 0, "strategy_return"]

    return score_report(
        equity=signals["equity"],
        returns=signals["strategy_return"],
        trade_returns=trade_returns,
    )


def main():
    df = load_sample_data()

    results = {
        "cost_1x": run_backtest(df, 0.0005),
        "cost_2x": run_backtest(df, 0.0010),
        "cost_3x": run_backtest(df, 0.0015),
    }

    print(json.dumps(results, indent=2))

    with open("experiments/latest_result.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
