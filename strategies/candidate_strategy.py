import pandas as pd


PARAMS = {
    "fast_ma": 20,
    "slow_ma": 50,
    "risk_per_trade": 0.01,
}


def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Candidate strategy.

    Required columns:
    - close

    Output:
    - signal: 1 long, 0 flat
    """
    params = params or PARAMS.copy()
    out = df.copy()

    fast = int(params["fast_ma"])
    slow = int(params["slow_ma"])

    out["fast_ma"] = out["close"].rolling(fast).mean()
    out["slow_ma"] = out["close"].rolling(slow).mean()

    out["signal"] = 0
    out.loc[out["fast_ma"] > out["slow_ma"], "signal"] = 1

    return out
